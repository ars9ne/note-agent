import asyncio
import getpass
import hashlib
import hmac
import json
import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from config import get_settings


logger = logging.getLogger("notes-bot")

USER_ROLE = "writer"
NOTES_ACCESS_MODES = {"none", "read", "edit"}


class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.role = USER_ROLE


class Intent:
    def __init__(self, kind, note_text=None, note_id=None):
        self.kind = kind
        self.note_text = note_text
        self.note_id = note_id

    def __repr__(self):
        return (
            "Intent("
            f"kind={self.kind!r}, "
            f"note_text={self.note_text!r}, "
            f"note_id={self.note_id!r}"
            ")"
        )


SYSTEM_PROMPT = """
Ты полезный консольный ассистент.
Отвечай на языке пользователя.
""".strip()


CLASSIFIER_PROMPT = """
Ты классификатор сообщений для консольного бота с заметками.

Верни только JSON без markdown:
{
  "kind": "add_note | list_notes | delete_note | update_note | chat",
  "note_text": "текст заметки или null",
  "note_id": число или null
}

Правила:
- add_note: пользователь явно просит добавить, записать, сохранить или создать заметку.
- list_notes: пользователь явно просит показать или перечислить заметки.
- delete_note: пользователь явно просит удалить заметку по id.
- update_note: пользователь явно просит изменить заметку по id.
- chat: обычные вопросы, советы, рецепты, объяснения и просьбы рассуждать.
- Вопросы вроде "чего не хватает в моём списке покупок" классифицируй как chat.
- Если данных для note_id нет, ставь null.
- Для add_note в note_text положи то, что нужно сохранить.
- Для update_note в note_text положи новый текст или суть изменения.
""".strip()


NOTES_READ_PROMPT = """
У тебя есть доступ только на чтение к заметкам пользователя.
Используй заметки как контекст, но не обещай, что можешь их изменить.
""".strip()


NOTES_EDIT_PROMPT = """
У тебя есть доступ к заметкам пользователя через код бота.
Ты НЕ вызываешь инструменты сам. Вместо этого ты можешь предложить действие.

Верни только JSON без markdown:
{
  "answer": "ответ пользователю",
  "proposed_action": null или {
    "type": "add_note | update_note | delete_note",
    "id": число или null,
    "text": "полный новый текст заметки для add/update или null",
    "reason": "почему это действие нужно"
  }
}

Правила:
- Если предлагаешь update_note, text должен быть полным новым содержимым заметки после правки.
- Если пользователь спрашивает, чего не хватает для рецепта, сравни рецепт с заметкой-списком покупок.
- Если чего-то не хватает, ответь списком недостающего и предложи update_note для заметки со списком покупок.
- Не предлагай delete_note без явной просьбы пользователя.
- Всегда проси подтверждение перед изменением заметки.
""".strip()


REPLY_CLASSIFIER_PROMPT = """
Ты классификатор короткого ответа пользователя на вопрос подтверждения.

Верни только JSON без markdown:
{
  "intent": "confirm | reject | conditional | unclear",
  "confidence": число от 0.0 до 1.0,
  "needs_clarification": true или false
}

Контекст содержит действие, которое бот предложил выполнить.

Правила:
- confirm: пользователь явно согласился выполнить предложенное действие.
- reject: пользователь явно отказался или отменил действие.
- conditional: пользователь согласен только с условием, ограничением или переносом на потом.
- unclear: ответ двусмысленный, саркастичный, слишком короткий без ясного смысла или не относится к вопросу.
- Для опасных действий будь строгим: если нет явного согласия, ставь unclear и needs_clarification=true.
- Учитывай отрицания: "да нет", "не надо", "не сейчас" не являются обычным confirm.
""".strip()


def sha256_hex(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_users():
    raw = os.getenv("BOT_USERS_JSON")
    if not raw:
        raise RuntimeError(
            "BOT_USERS_JSON не задан. Пример настройки есть в README.md."
        )

    try:
        users = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("BOT_USERS_JSON содержит некорректный JSON") from exc

    if not isinstance(users, dict):
        raise RuntimeError("BOT_USERS_JSON должен быть JSON-объектом")

    for user_id, record in users.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"Некорректная запись пользователя {user_id!r}")

        token_sha256 = record.get("token_sha256")

        if not isinstance(token_sha256, str) or len(token_sha256) != 64:
            raise RuntimeError(f"Некорректный token_sha256 для пользователя {user_id!r}")

    return users


def authenticate():
    users = load_users()

    user_id = input("login: ").strip()
    token = getpass.getpass("token: ")

    record = users.get(user_id)
    token_hash = sha256_hex(token)

    if not record or not hmac.compare_digest(token_hash, record["token_sha256"]):
        raise SystemExit("Аутентификация не пройдена")

    return User(user_id)


def message_text(message):
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def extract_json(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("JSON-объект не найден")


def intent_from_payload(payload):
    kind = payload.get("kind", "chat")
    if kind not in {"add_note", "list_notes", "delete_note", "update_note", "chat"}:
        kind = "chat"

    note_text = payload.get("note_text")
    if note_text is not None:
        note_text = str(note_text).strip() or None

    note_id = payload.get("note_id")
    if note_id is not None:
        try:
            note_id = int(note_id)
        except (TypeError, ValueError):
            note_id = None

    return Intent(kind, note_text=note_text, note_id=note_id)


def make_openrouter_llm(model):
    settings = get_settings()
    return ChatOpenAI(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0,
    )


async def classify_intent(user_text, classifier_llm=None):
    if classifier_llm is None:
        settings = get_settings()
        classifier_llm = make_openrouter_llm(settings.openrouter_classifier_model)

    response = await classifier_llm.ainvoke(
        [
            ("system", CLASSIFIER_PROMPT),
            ("human", user_text),
        ]
    )

    try:
        payload = extract_json(message_text(response))
    except Exception:
        logger.exception("Intent classifier returned invalid JSON")
        return Intent("chat")

    return intent_from_payload(payload)


def normalize_reply_text(text):
    text = text.strip().lower()
    for char in ".,!?:;\"'`":
        text = text.replace(char, " ")
    return " ".join(text.split())


def classify_reply_by_rules(user_text):
    text = normalize_reply_text(user_text)

    if not text:
        return {
            "intent": "unclear",
            "confidence": 0.0,
            "needs_clarification": True,
        }

    obvious_confirm = {
        "да",
        "ок",
        "okay",
        "yes",
        "подтверждаю",
        "согласен",
        "сделай",
        "добавь",
        "измени",
        "выполни",
        "да удалить",
        "да удали",
        "подтверждаю удаление",
    }
    obvious_reject = {
        "нет",
        "no",
        "не надо",
        "не нужно",
        "отмена",
        "отмени",
        "стоп",
        "stop",
    }

    if text in obvious_confirm:
        return {
            "intent": "confirm",
            "confidence": 0.99,
            "needs_clarification": False,
        }

    if text in obvious_reject:
        return {
            "intent": "reject",
            "confidence": 0.99,
            "needs_clarification": False,
        }

    return None


def normalize_reply_payload(payload):
    intent = payload.get("intent", "unclear")
    if intent not in {"confirm", "reject", "conditional", "unclear"}:
        intent = "unclear"

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))
    needs_clarification = bool(payload.get("needs_clarification", intent != "confirm"))

    return {
        "intent": intent,
        "confidence": confidence,
        "needs_clarification": needs_clarification,
    }


async def classify_reply(user_text, question_context, classifier_llm=None):
    rule_result = classify_reply_by_rules(user_text)
    if rule_result is not None:
        return rule_result

    if classifier_llm is None:
        settings = get_settings()
        classifier_llm = make_openrouter_llm(settings.openrouter_classifier_model)

    response = await classifier_llm.ainvoke(
        [
            ("system", REPLY_CLASSIFIER_PROMPT),
            (
                "human",
                "Контекст вопроса подтверждения:\n"
                f"{question_context}\n\n"
                f"Ответ пользователя:\n{user_text}",
            ),
        ]
    )

    try:
        payload = extract_json(message_text(response))
    except Exception:
        logger.exception("Reply classifier returned invalid JSON")
        return {
            "intent": "unclear",
            "confidence": 0.0,
            "needs_clarification": True,
        }

    return normalize_reply_payload(payload)


def safe_env_for_mcp(user):
    env = {
        "MCP_USER_ID": user.user_id,
        "NOTES_DB": os.getenv("NOTES_DB", str(Path("data/notes.sqlite").resolve())),
        "MAX_NOTE_LENGTH": os.getenv("MAX_NOTE_LENGTH", "4000"),
    }

    for key in (
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "SYSTEMROOT",
        "SystemRoot",
        "TEMP",
        "TMP",
    ):
        value = os.getenv(key)
        if value:
            env[key] = value

    return env


def patch_mcp_stdio_for_notebook():
    if sys.platform != "win32" or "ipykernel" not in sys.modules:
        return

    import langchain_mcp_adapters.sessions as mcp_sessions

    original_stdio_client = mcp_sessions.stdio_client
    if getattr(original_stdio_client, "_notes_bot_devnull_patch", False):
        return

    async def stdio_client_with_devnull(server):
        async with original_stdio_client(
            server,
            errlog=subprocess.DEVNULL,
        ) as streams:
            yield streams

    patched_stdio_client = asynccontextmanager(stdio_client_with_devnull)
    setattr(patched_stdio_client, "_notes_bot_devnull_patch", True)
    mcp_sessions.stdio_client = patched_stdio_client


def content_to_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)

    return str(value)


def try_parse_json_text(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    if isinstance(value, list):
        parsed_items = []
        for item in value:
            if not isinstance(item, dict) or "text" not in item:
                parsed_items = []
                break

            try:
                parsed_items.append(json.loads(item["text"]))
            except json.JSONDecodeError:
                parsed_items = []
                break

        if parsed_items:
            if len(parsed_items) == 1:
                return parsed_items[0]
            return parsed_items

        text = content_to_text(value)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value

    return value


class NotesGateway:
    def __init__(self, user):
        patch_mcp_stdio_for_notebook()

        server_path = Path(__file__).with_name("mcp_server.py").resolve()

        self.user = user
        self.client = MultiServerMCPClient(
            {
                "notes": {
                    "command": sys.executable,
                    "args": [str(server_path)],
                    "transport": "stdio",
                    "env": safe_env_for_mcp(user),
                }
            }
        )
        self.tools = {}

    async def initialize(self):
        tools = await self.client.get_tools()
        self.tools = {}

        for tool in tools:
            self.tools[tool.name] = tool

        required = {"add_note", "list_notes", "delete_note", "update_note"}
        missing = required - self.tools.keys()
        if missing:
            raise RuntimeError(f"В MCP-сервере заметок отсутствуют инструменты: {sorted(missing)}")

    async def get_notes(self):
        result = await self.tools["list_notes"].ainvoke({})
        parsed = try_parse_json_text(result)

        if isinstance(parsed, dict) and "id" in parsed:
            return [parsed]

        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            return parsed

        return []

    def format_notes(self, notes):
        if not notes:
            return "Заметок нет."

        lines = []
        for note in notes:
            note_id = note.get("id", "?")
            text = note.get("text", "")
            created_at = note.get("created_at", "")
            lines.append(f"{note_id}. {text} ({created_at})")

        return "\n".join(lines)

    async def add_note(self, text):
        if not text:
            return "Что записать? Пример: 'добавь заметку: купить 2 пакета барбариса'."

        result = await self.tools["add_note"].ainvoke({"text": text})
        parsed = try_parse_json_text(result)

        if isinstance(parsed, dict) and "id" in parsed:
            return f"Заметка сохранена. id={parsed['id']}"

        return f"Заметка сохранена.\n{content_to_text(result)}"

    async def list_notes(self):
        notes = await self.get_notes()
        return self.format_notes(notes)

    async def delete_note(self, note_id):
        if note_id is None:
            return "Укажи id заметки. Пример: `удали заметку 3`."

        result = await self.tools["delete_note"].ainvoke({"id": note_id})
        parsed = try_parse_json_text(result)

        if isinstance(parsed, dict) and parsed.get("deleted"):
            return f"Заметка {note_id} удалена."

        return content_to_text(result)

    async def update_note_record(self, note_id, text):
        result = await self.tools["update_note"].ainvoke(
            {
                "id": note_id,
                "text": text,
            }
        )
        parsed = try_parse_json_text(result)
        if isinstance(parsed, dict):
            return parsed
        return None

    async def update_note(self, note_id, text):
        if note_id is None:
            return "Укажи id заметки. Пример: `измени заметку 3: новый текст`."

        if not text:
            return "Укажи новый текст заметки."

        note = await self.update_note_record(note_id, text)
        if note:
            return f"Заметка {note_id} обновлена.\n{note['id']}. {note['text']}"

        return "Не удалось обновить заметку."


class ConsoleBot:
    def __init__(self, user, notes_access_mode=None):
        self.user = user
        self.notes = NotesGateway(user)
        self.memory = []
        self.pending_action = None

        settings = get_settings()
        self.llm = ChatOpenAI(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=0,
        )
        self.classifier_llm = ChatOpenAI(
            model=settings.openrouter_classifier_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=0,
        )

        mode = notes_access_mode or os.getenv("NOTES_ACCESS_MODE", "none")
        self.set_notes_access_mode(mode)

    def set_notes_access_mode(self, mode):
        mode = (mode or "none").strip().lower()
        if mode not in NOTES_ACCESS_MODES:
            raise ValueError(f"Неизвестный режим доступа к заметкам: {mode}")
        self.notes_access_mode = mode

    def remember(self, user_text, answer):
        self.memory.append(("human", user_text))
        self.memory.append(("ai", answer))

    def recent_memory(self):
        return self.memory[-12:]

    async def initialize(self):
        await self.notes.initialize()

    def notes_context(self, notes):
        if not notes:
            return "Заметок нет."

        lines = []
        for note in notes:
            lines.append(f"id={note.get('id')}: {note.get('text')}")
        return "\n".join(lines)

    def pending_action_context(self):
        action = self.pending_action or {}
        return (
            f"Тип действия: {action.get('type')}\n"
            f"Описание: {self.describe_action(action)}\n"
            f"id заметки: {action.get('id')}\n"
            f"Новый текст: {action.get('text')}\n"
            f"Причина: {action.get('reason')}"
        )

    def can_execute_reply(self, reply):
        action_type = (self.pending_action or {}).get("type")
        threshold = 0.95 if action_type == "delete_note" else 0.75

        return (
            reply["intent"] == "confirm"
            and reply["confidence"] >= threshold
            and not reply["needs_clarification"]
        )

    def clarification_for_reply(self, reply):
        action_type = (self.pending_action or {}).get("type")

        if reply["intent"] == "conditional":
            return (
                "Я вижу условие или ограничение, поэтому не буду менять заметки автоматически. "
                "Напиши явно `да, выполнить` или `нет, отмена`."
            )

        if action_type == "delete_note":
            return (
                "Удаление заметки требует явного подтверждения. "
                "Напиши `да, удалить` или `нет, отмена`."
            )

        return (
            "Я не уверен, что это подтверждение. "
            "Напиши `да, выполнить` или `нет, отмена`."
        )

    async def handle_pending_action_reply(self, user_text):
        reply = await classify_reply(
            user_text,
            self.pending_action_context(),
            self.classifier_llm,
        )

        if self.can_execute_reply(reply):
            return await self.execute_pending_action()

        if reply["intent"] == "reject" and reply["confidence"] >= 0.75:
            self.pending_action = None
            return "Ок, изменение заметок отменено."

        return self.clarification_for_reply(reply)

    async def handle(self, user_text):
        mode_answer = self.try_handle_mode_command(user_text)
        if mode_answer:
            return mode_answer

        if self.notes_access_mode == "edit" and self.pending_action:
            return await self.handle_pending_action_reply(user_text)

        intent = await classify_intent(user_text, self.classifier_llm)
        logger.info(
            "user=%s route=%s mode=%s",
            self.user.user_id,
            intent.kind,
            self.notes_access_mode,
        )
        # Ветвление
        if intent.kind == "add_note":
            return await self.notes.add_note(intent.note_text)

        if intent.kind == "list_notes":
            return await self.notes.list_notes()

        if intent.kind == "delete_note":
            return await self.notes.delete_note(intent.note_id)

        if intent.kind == "update_note":
            return await self.notes.update_note(intent.note_id, intent.note_text)

        return await self.handle_general_chat(user_text)

    def try_handle_mode_command(self, user_text):
        text = user_text.strip().lower()
        if not text.startswith("/mode"):
            return None

        parts = text.split()
        if len(parts) == 1:
            return f"Текущий режим заметок: {self.notes_access_mode}"

        self.set_notes_access_mode(parts[1])
        self.pending_action = None
        return f"Режим заметок переключён на: {self.notes_access_mode}"

    async def handle_general_chat(self, user_text):
        if self.notes_access_mode == "none":
            answer = await self.call_plain_llm(user_text)
            self.remember(user_text, answer)
            return answer

        notes = await self.notes.get_notes()

        if self.notes_access_mode == "read":
            answer = await self.call_read_notes_llm(user_text, notes)
            self.remember(user_text, answer)
            return answer

        answer = await self.call_edit_notes_llm(user_text, notes)
        self.remember(user_text, answer)
        return answer

    async def call_plain_llm(self, user_text):
        messages = [
            ("system", SYSTEM_PROMPT),
            *self.recent_memory(),
            ("human", user_text),
        ]
        response = await self.llm.ainvoke(messages)
        return message_text(response)

    async def call_read_notes_llm(self, user_text, notes):
        messages = [
            ("system", SYSTEM_PROMPT),
            ("system", NOTES_READ_PROMPT),
            ("system", "Заметки пользователя:\n" + self.notes_context(notes)),
            *self.recent_memory(),
            ("human", user_text),
        ]
        response = await self.llm.ainvoke(messages)
        return message_text(response)

    async def call_edit_notes_llm(self, user_text, notes):
        messages = [
            ("system", SYSTEM_PROMPT),
            ("system", NOTES_EDIT_PROMPT),
            ("system", "Заметки пользователя:\n" + self.notes_context(notes)),
            *self.recent_memory(),
            ("human", user_text),
        ]
        response = await self.llm.ainvoke(messages)

        try:
            payload = extract_json(message_text(response))
        except Exception:
            logger.exception("Edit-mode LLM returned invalid JSON")
            return message_text(response)

        answer = str(payload.get("answer") or "").strip()
        action = payload.get("proposed_action")

        if isinstance(action, dict) and action.get("type"):
            self.pending_action = action
            reason = action.get("reason") or "LLM предложила изменить заметки."
            action_text = self.describe_action(action)
            return (
                f"{answer}\n\n"
                f"Предложенное действие: {action_text}\n"
                f"Причина: {reason}\n"
                "Напиши `да`, если нужно выполнить это изменение."
            ).strip()

        self.pending_action = None
        return answer or "Не удалось получить ответ."

    def describe_action(self, action):
        action_type = action.get("type")
        note_id = action.get("id")

        if action_type == "add_note":
            return "добавить новую заметку"
        if action_type == "update_note":
            return f"обновить заметку id={note_id}"
        if action_type == "delete_note":
            return f"удалить заметку id={note_id}"

        return str(action)

    async def execute_pending_action(self):
        action = self.pending_action
        self.pending_action = None

        action_type = action.get("type")
        note_id = action.get("id")
        text = action.get("text")

        if action_type == "add_note":
            result = await self.notes.add_note(text)
            notes_after = await self.notes.list_notes()
            return f"{result}\n\nСодержимое заметок после изменения:\n{notes_after}"

        if action_type == "update_note":
            note = await self.notes.update_note_record(note_id, text)
            if not note:
                return "Не удалось обновить заметку."
            return (
                "Готово. Заметка после редактирования:\n"
                f"{note['id']}. {note['text']}"
            )

        if action_type == "delete_note":
            result = await self.notes.delete_note(note_id)
            notes_after = await self.notes.list_notes()
            return f"{result}\n\nСодержимое заметок после изменения:\n{notes_after}"

        return "Неизвестное действие, ничего не изменено."


async def main():
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        get_settings()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    user = authenticate()
    bot = ConsoleBot(user)
    await bot.initialize()

    print(f"Authenticated as {user.user_id} ({user.role})")
    print("Commands: `exit`, `quit`, `выход`, `/mode none|read|edit`")
    print(f"Notes mode: {bot.notes_access_mode}")
    print()

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit", "выход"}:
            break

        try:
            answer = await bot.handle(user_text)
        except Exception:
            logger.exception("Request failed")
            answer = "Произошла ошибка при обработке запроса."

        print(f"bot> {answer}")


if __name__ == "__main__":
    asyncio.run(main())
