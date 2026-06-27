import asyncio
import json

import pytest

import bot
import mcp_server


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeClassifierLLM:
    async def ainvoke(self, messages):
        user_text = messages[-1][1]

        if "добавь" in user_text:
            payload = {
                "kind": "add_note",
                "note_text": "купить 2 пакета барбариса",
                "note_id": None,
            }
        elif "покажи" in user_text:
            payload = {
                "kind": "list_notes",
                "note_text": None,
                "note_id": None,
            }
        elif "удали" in user_text:
            payload = {
                "kind": "delete_note",
                "note_text": None,
                "note_id": 12,
            }
        else:
            payload = {
                "kind": "chat",
                "note_text": None,
                "note_id": None,
            }

        return FakeResponse(json.dumps(payload, ensure_ascii=False))


def test_user_is_always_writer():
    user = bot.User("user1")
    assert user.role == "writer"


def test_llm_classify_intent_notes_and_chat():
    async def run():
        classifier = FakeClassifierLLM()

        add_intent = await bot.classify_intent(
            "добавь заметку: купить 2 пакета барбариса",
            classifier,
        )
        list_intent = await bot.classify_intent("покажи мои заметки", classifier)
        delete_intent = await bot.classify_intent("удали заметку 12", classifier)
        chat_intent = await bot.classify_intent(
            "В каких блюдах используют барбарис?",
            classifier,
        )

        return add_intent, list_intent, delete_intent, chat_intent

    add_intent, list_intent, delete_intent, chat_intent = asyncio.run(run())

    assert add_intent.kind == "add_note"
    assert add_intent.note_text == "купить 2 пакета барбариса"

    assert list_intent.kind == "list_notes"

    assert delete_intent.kind == "delete_note"
    assert delete_intent.note_id == 12

    assert chat_intent.kind == "chat"


def test_classify_reply_rules_for_obvious_answers():
    confirm = asyncio.run(bot.classify_reply("да", "обновить заметку"))
    reject = asyncio.run(bot.classify_reply("не надо", "обновить заметку"))
    delete_confirm = asyncio.run(bot.classify_reply("да, удалить", "удалить заметку"))

    assert confirm["intent"] == "confirm"
    assert confirm["confidence"] == 0.99
    assert confirm["needs_clarification"] is False

    assert reject["intent"] == "reject"
    assert reject["confidence"] == 0.99
    assert reject["needs_clarification"] is False

    assert delete_confirm["intent"] == "confirm"
    assert delete_confirm["confidence"] == 0.99
    assert delete_confirm["needs_clarification"] is False


def test_classify_reply_uses_llm_for_non_exact_confirmation():
    class FakeReplyLLM:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, messages):
            self.calls.append(messages)
            payload = {
                "intent": "confirm",
                "confidence": 0.81,
                "needs_clarification": False,
            }
            return FakeResponse(json.dumps(payload))

    classifier = FakeReplyLLM()
    result = asyncio.run(
        bot.classify_reply(
            "да, сделай",
            "обновить заметку со списком покупок",
            classifier,
        )
    )

    assert result == {
        "intent": "confirm",
        "confidence": 0.81,
        "needs_clarification": False,
    }
    assert len(classifier.calls) == 1


def test_classify_reply_uses_llm_for_gray_zone():
    class FakeReplyLLM:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, messages):
            self.calls.append(messages)
            payload = {
                "intent": "conditional",
                "confidence": 0.82,
                "needs_clarification": True,
            }
            return FakeResponse(json.dumps(payload))

    classifier = FakeReplyLLM()
    result = asyncio.run(
        bot.classify_reply(
            "ну ок, но не сейчас",
            "обновить заметку со списком покупок",
            classifier,
        )
    )

    assert result == {
        "intent": "conditional",
        "confidence": 0.82,
        "needs_clarification": True,
    }
    assert len(classifier.calls) == 1


def test_console_bot_keeps_general_memory(monkeypatch):
    class FakeSettings:
        openrouter_api_key = "test-key"
        openrouter_model = "general-model"
        openrouter_classifier_model = "classifier-model"
        openrouter_base_url = "https://example.test"

    class FakeLLM:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.calls = []

        async def ainvoke(self, messages):
            self.calls.append(messages)

            if self.model == "classifier-model":
                payload = {
                    "kind": "chat",
                    "note_text": None,
                    "note_id": None,
                }
                return FakeResponse(json.dumps(payload))

            return FakeResponse(f"answer {len(self.calls)}")

    monkeypatch.setattr(bot, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(bot, "ChatOpenAI", FakeLLM)

    console_bot = bot.ConsoleBot(bot.User("demo"))

    async def run_chat():
        first_answer = await console_bot.handle("привет")
        second_answer = await console_bot.handle("как дела?")
        return first_answer, second_answer

    first_answer, second_answer = asyncio.run(run_chat())

    assert first_answer == "answer 1"
    assert second_answer == "answer 2"
    assert console_bot.memory == [
        ("human", "привет"),
        ("ai", "answer 1"),
        ("human", "как дела?"),
        ("ai", "answer 2"),
    ]
    assert ("human", "привет") in console_bot.llm.calls[1]
    assert ("ai", "answer 1") in console_bot.llm.calls[1]


def test_edit_mode_executes_confirmed_update(monkeypatch):
    class FakeSettings:
        openrouter_api_key = "test-key"
        openrouter_model = "general-model"
        openrouter_classifier_model = "classifier-model"
        openrouter_base_url = "https://example.test"

    class FakeLLM:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

        async def ainvoke(self, messages):
            if self.model == "classifier-model":
                payload = {
                    "kind": "chat",
                    "note_text": None,
                    "note_id": None,
                }
                return FakeResponse(json.dumps(payload))

            payload = {
                "answer": "Не хватает сахара.",
                "proposed_action": {
                    "type": "update_note",
                    "id": 1,
                    "text": "список покупок: молоко, яйца, сахар",
                    "reason": "Сахар нужен для бостонского пирога.",
                },
            }
            return FakeResponse(json.dumps(payload, ensure_ascii=False))

    class FakeNotesGateway:
        def __init__(self, user):
            self.user = user
            self.updated = None

        async def get_notes(self):
            return [
                {
                    "id": 1,
                    "owner_id": "demo",
                    "text": "список покупок: молоко, яйца",
                    "created_at": "now",
                }
            ]

        async def update_note_record(self, note_id, text):
            self.updated = (note_id, text)
            return {
                "id": note_id,
                "owner_id": "demo",
                "text": text,
                "created_at": "now",
            }

    monkeypatch.setattr(bot, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(bot, "ChatOpenAI", FakeLLM)
    monkeypatch.setattr(bot, "NotesGateway", FakeNotesGateway)

    console_bot = bot.ConsoleBot(
        bot.User("demo"),
        notes_access_mode="edit",
    )

    async def run_chat():
        first_answer = await console_bot.handle("что не хватает?")
        second_answer = await console_bot.handle("да")
        return first_answer, second_answer

    first_answer, second_answer = asyncio.run(run_chat())

    assert "Не хватает сахара" in first_answer
    assert console_bot.pending_action is None
    assert "Заметка после редактирования" in second_answer
    assert "сахар" in second_answer
    assert console_bot.notes.updated == (
        1,
        "список покупок: молоко, яйца, сахар",
    )


def test_pending_action_uses_action_specific_confidence_thresholds():
    console_bot = object.__new__(bot.ConsoleBot)

    console_bot.pending_action = {"type": "update_note"}
    assert (
        console_bot.can_execute_reply(
            {
                "intent": "confirm",
                "confidence": 0.749,
                "needs_clarification": False,
            }
        )
        is False
    )
    assert (
        console_bot.can_execute_reply(
            {
                "intent": "confirm",
                "confidence": 0.75,
                "needs_clarification": False,
            }
        )
        is True
    )

    console_bot.pending_action = {"type": "delete_note"}
    assert (
        console_bot.can_execute_reply(
            {
                "intent": "confirm",
                "confidence": 0.949,
                "needs_clarification": False,
            }
        )
        is False
    )
    assert (
        console_bot.can_execute_reply(
            {
                "intent": "confirm",
                "confidence": 0.95,
                "needs_clarification": False,
            }
        )
        is True
    )


def test_delete_action_requires_explicit_high_confidence(monkeypatch):
    class FakeSettings:
        openrouter_api_key = "test-key"
        openrouter_model = "general-model"
        openrouter_classifier_model = "classifier-model"
        openrouter_base_url = "https://example.test"

    class FakeLLM:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

        async def ainvoke(self, messages):
            payload = {
                "intent": "confirm",
                "confidence": 0.7,
                "needs_clarification": False,
            }
            return FakeResponse(json.dumps(payload))

    class FakeNotesGateway:
        def __init__(self, user):
            self.user = user
            self.deleted = False

        async def delete_note(self, note_id):
            self.deleted = True
            return f"Заметка {note_id} удалена."

    monkeypatch.setattr(bot, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(bot, "ChatOpenAI", FakeLLM)
    monkeypatch.setattr(bot, "NotesGateway", FakeNotesGateway)

    console_bot = bot.ConsoleBot(bot.User("demo"), notes_access_mode="edit")
    console_bot.pending_action = {
        "type": "delete_note",
        "id": 7,
        "text": None,
        "reason": "тест удаления",
    }

    answer = asyncio.run(console_bot.handle("ну ок"))

    assert "Удаление заметки требует явного подтверждения" in answer
    assert console_bot.notes.deleted is False
    assert console_bot.pending_action is not None


def test_mcp_requires_user_id(monkeypatch):
    monkeypatch.delenv("MCP_USER_ID", raising=False)

    with pytest.raises(Exception):
        mcp_server.current_identity()


def test_mcp_notes_add_list_update_delete(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_USER_ID", "user1")
    monkeypatch.setattr(mcp_server, "DB_PATH", tmp_path / "notes.sqlite")

    note = mcp_server.add_note("купить 2 пакета барбариса")
    assert note["id"] == 1
    assert note["owner_id"] == "user1"
    assert note["text"] == "купить 2 пакета барбариса"

    updated = mcp_server.update_note(note["id"], "купить 2 пакета барбариса и яйца")
    assert updated["id"] == note["id"]
    assert updated["text"] == "купить 2 пакета барбариса и яйца"

    notes = mcp_server.list_notes()
    assert len(notes) == 1
    assert notes[0]["text"] == "купить 2 пакета барбариса и яйца"

    result = mcp_server.delete_note(note["id"])
    assert result == {"deleted": True, "id": 1}
    assert mcp_server.list_notes() == []


def test_writer_users_do_not_see_each_other_notes(monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_server, "DB_PATH", tmp_path / "notes.sqlite")

    monkeypatch.setenv("MCP_USER_ID", "user1")
    mcp_server.add_note("заметка user1")

    monkeypatch.setenv("MCP_USER_ID", "user2")
    mcp_server.add_note("заметка user2")

    user2_notes = mcp_server.list_notes()
    assert len(user2_notes) == 1
    assert user2_notes[0]["owner_id"] == "user2"
    assert user2_notes[0]["text"] == "заметка user2"


def test_notes_tools_are_registered():
    tool_names = []
    for tool in mcp_server.mcp._tool_manager.list_tools():
        tool_names.append(tool.name)

    assert "add_note" in tool_names
    assert "list_notes" in tool_names
    assert "delete_note" in tool_names
    assert "update_note" in tool_names
