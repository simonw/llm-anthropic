from copy import deepcopy

import llm
import pytest
from llm.parts import ToolCallPart, ToolResultPart


@pytest.fixture(params=[llm.get_model, llm.get_async_model])
def model(request):
    return request.param("claude-opus-5")


@pytest.mark.parametrize("update_count", [2, 3])
@pytest.mark.parametrize("cache", [False, True])
def test_multiple_system_updates_before_next_assistant(model, update_count, cache):
    messages = [
        llm.system("Initial instruction"),
        llm.user("First question"),
        llm.assistant("First answer"),
    ]
    for index in range(update_count):
        messages.extend(
            [llm.system(f"Instruction {index}"), llm.user(f"Question {index}")]
        )
    prompt = model.prompt(messages=messages, cache=cache).prompt
    original = deepcopy(prompt.messages)

    kwargs = model.build_kwargs(prompt, None)

    assert kwargs["system"] == "Initial instruction"
    assert [m["role"] for m in kwargs["messages"]] == [
        "user",
        "assistant",
        "user",
        "system",
    ]
    assert kwargs["messages"][2]["content"] == [
        {"type": "text", "text": f"Question {index}"} for index in range(update_count)
    ]
    expected_updates = [
        {"type": "text", "text": f"Instruction {index}"}
        for index in range(update_count)
    ]
    if cache:
        expected_updates[-1]["cache_control"] = {"type": "ephemeral"}
    assert kwargs["messages"][3]["content"] == expected_updates
    assert prompt.messages == original
    assert model.build_kwargs(prompt, None) == kwargs


def test_system_updates_do_not_split_parallel_tool_results(model):
    messages = [
        llm.user("Run both checks"),
        llm.assistant(
            ToolCallPart(name="check", arguments={"index": 1}, tool_call_id="c1"),
            ToolCallPart(name="check", arguments={"index": 2}, tool_call_id="c2"),
        ),
        llm.system("Keep the answer brief"),
        llm.tool_message(ToolResultPart(name="check", tool_call_id="c1", output="one")),
        llm.system("Use French"),
        llm.tool_message(ToolResultPart(name="check", tool_call_id="c2", output="two")),
    ]
    prompt = model.prompt(messages=messages).prompt
    original = deepcopy(prompt.messages)

    kwargs = model.build_kwargs(prompt, None)

    assert [m["role"] for m in kwargs["messages"]] == [
        "user",
        "assistant",
        "user",
        "system",
    ]
    assert kwargs["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "c1", "content": "one"},
        {"type": "tool_result", "tool_use_id": "c2", "content": "two"},
    ]
    assert kwargs["messages"][3]["content"] == [
        {"type": "text", "text": "Keep the answer brief"},
        {"type": "text", "text": "Use French"},
    ]
    assert prompt.messages == original


def test_empty_assistant_does_not_flush_pending_system_updates(model):
    prompt = model.prompt(
        messages=[
            llm.user("First question"),
            llm.system("First update"),
            llm.assistant(""),
            llm.user("Second question"),
            llm.system("Second update"),
            llm.user("Third question"),
            llm.assistant("Answer"),
        ]
    ).prompt

    kwargs = model.build_kwargs(prompt, None)

    assert kwargs["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First question"},
                {"type": "text", "text": "Second question"},
                {"type": "text", "text": "Third question"},
            ],
        },
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "First update"},
                {"type": "text", "text": "Second update"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Answer"}]},
    ]


def test_system_normalization_preserves_completed_turn_prefix(model):
    history = [
        llm.system("Initial instruction"),
        llm.user("First question"),
        llm.system("First update"),
        llm.assistant("First answer"),
    ]
    prefix = model.build_kwargs(model.prompt(messages=history).prompt, None)
    prompt = model.prompt(
        messages=[*history, llm.system("Next update"), llm.user("Next question")]
    ).prompt

    kwargs = model.build_kwargs(prompt, None)

    assert kwargs["system"] == prefix["system"]
    assert kwargs["messages"][:3] == prefix["messages"]
    assert kwargs["messages"][3:] == [
        {"role": "user", "content": [{"type": "text", "text": "Next question"}]},
        {"role": "system", "content": [{"type": "text", "text": "Next update"}]},
    ]
