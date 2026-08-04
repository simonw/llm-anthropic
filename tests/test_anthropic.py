import json
import llm
import os
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\xa6\x00\x00\x01\x1a"
    b"\x02\x03\x00\x00\x00\xe6\x99\xc4^\x00\x00\x00\tPLTE\xff\xff\xff"
    b"\x00\xff\x00\xfe\x01\x00\x12t\x01J\x00\x00\x00GIDATx\xda\xed\xd81\x11"
    b"\x000\x08\xc0\xc0.]\xea\xaf&Q\x89\x04V\xe0>\xf3+\xc8\x91Z\xf4\xa2\x08EQ\x14E"
    b"Q\x14EQ\x14EQ\xd4B\x91$I3\xbb\xbf\x08EQ\x14EQ\x14EQ\x14E\xd1\xa5"
    b"\xd4\x17\x91\xc6\x95\x05\x15\x0f\x9f\xc5\t\x9f\xa4\x00\x00\x00\x00IEND\xaeB`"
    b"\x82"
)

ANTHROPIC_API_KEY = os.environ.get("PYTEST_ANTHROPIC_API_KEY", None) or "sk-..."
FIXED_TEST_VERSION = "0.32a0"


def fixed_version_tool():
    def fixed_version() -> str:
        return FIXED_TEST_VERSION

    return llm.Tool.function(
        fixed_version,
        name="fixed_version",
        description="Return a fixed test version string",
    )


@pytest.mark.vcr
def test_prompt():
    model = llm.get_model("claude-sonnet-4.5")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt("Two names for a pet pelican, be brief")
    assert str(response) == snapshot("- Captain\n- Scoop")
    response_dict = dict(response.response_json)
    response_dict.pop("id")  # differs between requests
    assert response_dict == snapshot(
        {
            "container": None,
            "content": [
                {
                    "citations": None,
                    "parsed_output": None,
                    "text": "- Captain\n- Scoop",
                    "type": "text",
                }
            ],
            "model": "claude-sonnet-4-5-20250929",
            "role": "assistant",
            "stop_details": None,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
        }
    )
    assert response.input_tokens == snapshot(17)
    assert response.output_tokens == snapshot(10)
    assert response.token_details is None


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_prompt():
    model = llm.get_async_model("claude-sonnet-4.5")
    model.key = model.key or ANTHROPIC_API_KEY  # don't override existing key
    conversation = model.conversation()
    response = await conversation.prompt("Two names for a pet pelican, be brief")
    assert await response.text() == snapshot("- Captain\n- Scoop")
    response_dict = dict(response.response_json)
    response_dict.pop("id")  # differs between requests
    assert response_dict == snapshot(
        {
            "container": None,
            "content": [
                {
                    "citations": None,
                    "parsed_output": None,
                    "text": "- Captain\n- Scoop",
                    "type": "text",
                }
            ],
            "model": "claude-sonnet-4-5-20250929",
            "role": "assistant",
            "stop_details": None,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
        }
    )
    assert response.input_tokens == snapshot(17)
    assert response.output_tokens == snapshot(10)
    assert response.token_details is None
    response2 = await conversation.prompt("in french")
    assert await response2.text() == snapshot("- Capitaine\n- Bec (beak)")


@pytest.mark.vcr
def test_image_prompt():
    model = llm.get_model("claude-sonnet-4.5")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt(
        "Describe image in three words",
        attachments=[llm.Attachment(content=TINY_PNG)],
    )
    assert str(response) == snapshot("Red square, green square.")
    response_dict = response.response_json
    response_dict.pop("id")  # differs between requests
    assert response_dict == snapshot(
        {
            "container": None,
            "content": [
                {
                    "citations": None,
                    "parsed_output": None,
                    "text": "Red square, green square.",
                    "type": "text",
                }
            ],
            "model": "claude-sonnet-4-5-20250929",
            "role": "assistant",
            "stop_details": None,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
        }
    )
    assert response.input_tokens == snapshot(83)
    assert response.output_tokens == snapshot(9)
    assert response.token_details is None


@pytest.mark.vcr
def test_image_with_no_prompt():
    model = llm.get_model("claude-sonnet-4.5")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt(
        prompt=None,
        attachments=[llm.Attachment(content=TINY_PNG)],
    )
    assert str(response) == snapshot(
        "I need to describe what I see in this image.\n\n"
        "The image shows two solid colored rectangles arranged vertically on a white background:\n\n"
        "1. **Top rectangle**: A bright red rectangle positioned in the upper portion of the image\n"
        "2. **Bottom rectangle**: A bright green (lime green) rectangle positioned in the lower portion of the image\n\n"
        "Both rectangles appear to be roughly the same size and shape (horizontal rectangles/landscape orientation), "
        "and they are separated by white space between them."
    )


@pytest.mark.vcr
def test_url_prompt():
    model = llm.get_model("claude-sonnet-4.5")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt(
        prompt="describe image",
        attachments=[
            llm.Attachment(
                url="https://static.simonwillison.net/static/2024/pelican.jpg"
            )
        ],
    )
    assert str(response) == snapshot(
        "This image shows a **brown pelican** perched on rocky terrain at what appears "
        "to be a marina or harbor. The pelican is captured in profile, displaying its "
        "distinctive features:\n\n"
        "- **Long, prominent bill** with the characteristic pelican pouch\n"
        "- **White head and neck** with darker gray-brown plumage on its body and wings\n"
        "- **Sturdy build** with detailed feather texture visible in the wings\n\n"
        "The background shows several **boats docked in a marina**, slightly out of "
        "focus, creating a typical coastal or waterfront setting. The lighting suggests "
        "this photo was taken during daytime, with bright natural light that creates "
        "a slight halo effect around the bird's head.\n\n"
        "The pelican appears calm and at rest, which is common behavior for these "
        "seabirds in harbor areas where they often wait for fishing opportunities or "
        "scraps from nearby boats. The rocky perch and marina setting are typical "
        "habitats where pelicans congregate along coastlines."
    )


class Dog(BaseModel):
    name: str
    age: int
    bio: str


@pytest.mark.vcr
def test_schema_prompt():
    model = llm.get_model("claude-sonnet-4.5")

    response = model.prompt("Invent a good dog", schema=Dog, key=ANTHROPIC_API_KEY)
    dog = json.loads(response.text())
    assert dog == snapshot(
        {
            "name": "Biscuit",
            "age": 4,
            "bio": (
                "Biscuit is a golden retriever with a gentle soul and boundless "
                "enthusiasm. He greets every person with a wagging tail and has an uncanny "
                "ability to sense when someone needs comfort. His favorite activities "
                "include playing fetch at the beach, napping in sunny spots, and stealing "
                "socks to add to his secret collection under the bed."
            ),
        }
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_schema_prompt_async():
    model = llm.get_async_model("claude-sonnet-4.5")
    response = await model.prompt(
        "Invent a terrific dog", schema=Dog, key=ANTHROPIC_API_KEY
    )
    dog_json = await response.text()
    dog = json.loads(dog_json)
    assert dog == snapshot(
        {
            "name": "Luna",
            "age": 4,
            "bio": (
                "Luna is a brilliant Golden Retriever with a heart of gold who serves as "
                "a certified therapy dog at children's hospitals. She has an uncanny "
                "ability to sense when someone needs comfort and gently rests her head on "
                "their lap. Luna loves swimming in lakes, playing fetch with her favorite "
                "tennis ball, and has learned over 50 commands including helping her owner "
                "retrieve items from around the house."
            ),
        }
    )


@pytest.mark.vcr
def test_prompt_with_prefill_and_stop_sequences():
    model = llm.get_model("claude-haiku-4.5")
    response = model.prompt(
        "Very short function describing a pelican",
        prefill="```python",
        stop_sequences=["```"],
        hide_prefill=True,
        key=ANTHROPIC_API_KEY,
    )
    text = response.text()
    assert text == snapshot(
        "\ndef pelican():\n"
        '    return "A large waterbird with a long bill and a throat pouch for catching fish."\n'
    )


@pytest.mark.vcr
def test_thinking_prompt():
    model = llm.get_model("claude-sonnet-4.5")
    conversation = model.conversation()
    response = conversation.prompt(
        "Two names for a pet pelican, be brief", thinking=True, key=ANTHROPIC_API_KEY
    )
    assert response.text() == snapshot("- Captain\n- Scoop")
    response_dict = dict(response.response_json)
    response_dict.pop("id")  # differs between requests
    # Check structure without exact thinking signature
    assert response_dict["model"] == snapshot("claude-sonnet-4-5-20250929")
    assert response_dict["stop_reason"] == snapshot("end_turn")
    content_types = [block["type"] for block in response_dict["content"]]
    assert "thinking" in content_types
    assert "text" in content_types
    assert response.input_tokens == snapshot(46)
    assert response.output_tokens == snapshot(84)
    assert response.token_details is None


@pytest.mark.vcr
def test_tools():
    model = llm.get_model("claude-haiku-4.5")
    names = ["Charles", "Sammy"]
    chain_response = model.chain(
        "Two names for a pet pelican",
        tools=[
            llm.Tool.function(lambda: names.pop(0), name="pelican_name_generator"),
        ],
        key=ANTHROPIC_API_KEY,
    )
    text = chain_response.text()
    assert text == snapshot(
        " Here are two great names for your pet pelican:\n\n"
        "1. **Charles** - A sophisticated and dignified name, perfect for a pelican with personality!\n"
        "2. **Sammy** - A friendly and playful name that gives off warm, approachable vibes.\n\n"
        "Either of these would make an excellent name for your feathered friend! \U0001f985"
    )
    tool_calls = chain_response._responses[0].tool_calls()
    assert len(tool_calls) == 2
    assert all(call.name == "pelican_name_generator" for call in tool_calls)
    assert [
        result.output for result in chain_response._responses[1].prompt.tool_results
    ] == snapshot(["Charles", "Sammy"])


@pytest.mark.vcr
def test_fixed_version_tool_chain_regression():
    model = llm.get_model("claude-haiku-4.5")
    fixed_version = fixed_version_tool()

    chain_response = model.chain(
        "Use the fixed_version tool. Then tell me the version and make one short joke about it.",
        tools=[fixed_version],
        key=ANTHROPIC_API_KEY,
    )
    text = chain_response.text()
    assert FIXED_TEST_VERSION in text
    assert len(chain_response._responses) == 2
    second_response = chain_response._responses[1]
    assert second_response.prompt.tool_results[0].output == FIXED_TEST_VERSION
    second_request_messages = model.build_messages(
        second_response.prompt, second_response.conversation
    )
    assert [message["role"] for message in second_request_messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert second_request_messages[1]["content"][-1]["type"] == "tool_use"
    assert [block["type"] for block in second_request_messages[2]["content"]] == [
        "tool_result"
    ]


@pytest.mark.vcr
def test_fixed_version_tool_chain_with_thinking_display_regression():
    model = llm.get_model("claude-haiku-4.5")
    from llm.parts import ReasoningPart

    fixed_version = fixed_version_tool()

    chain_response = model.chain(
        "Use the fixed_version tool. Then tell me the version and make one short joke about it. Think about it first.",
        tools=[fixed_version],
        key=ANTHROPIC_API_KEY,
        options={"thinking_display": True},
    )
    text = chain_response.text()
    assert FIXED_TEST_VERSION in text
    assert len(chain_response._responses) == 2

    first_response = chain_response._responses[0]
    reasoning_parts = [
        part
        for message in first_response.messages()
        for part in message.parts
        if isinstance(part, ReasoningPart)
    ]
    assert reasoning_parts[0].provider_metadata["anthropic"]["signature"]

    second_response = chain_response._responses[1]
    second_request_messages = model.build_messages(
        second_response.prompt, second_response.conversation
    )
    assert second_request_messages[1]["content"][0]["type"] == "thinking"
    assert second_request_messages[1]["content"][0]["signature"]
    assert second_request_messages[1]["content"][-1]["type"] == "tool_use"
    assert [block["type"] for block in second_request_messages[2]["content"]] == [
        "tool_result"
    ]


@pytest.mark.vcr
def test_web_search():
    model = llm.get_model("claude-opus-4.1")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt(
        "What is the current weather in San Francisco?", web_search=True
    )
    response_text = str(response)
    assert len(response_text) > 0
    assert any(
        word in response_text.lower()
        for word in ["weather", "temperature", "san francisco", "degree", "forecast"]
    )
    response_dict = dict(response.response_json)
    assert "content" in response_dict
    assert len(response_dict["content"]) > 0


def test_fast_mode_kwargs():
    model = llm.get_model("claude-opus-4.8")
    prompt = llm.Prompt("Hi", model, options=model.Options(fast=True))
    kwargs = model.build_kwargs(prompt, None)
    assert kwargs["speed"] == "fast"
    assert "fast-mode-2026-02-01" in kwargs["betas"]


def test_fast_mode_off_by_default():
    model = llm.get_model("claude-opus-4.8")
    prompt = llm.Prompt("Hi", model, options=model.Options())
    kwargs = model.build_kwargs(prompt, None)
    assert "speed" not in kwargs
    assert "betas" not in kwargs


def test_opus_5_registered():
    model = llm.get_model("claude-opus-5")
    assert model.model_id == "anthropic/claude-opus-5"
    assert model.claude_model_id == "claude-opus-5"
    assert "application/pdf" in model.attachment_types
    assert model.supports_thinking
    assert model.supports_thinking_effort
    assert model.supports_adaptive_thinking
    assert model.supports_web_search
    assert model.use_structured_outputs
    assert model.default_max_tokens == 128000
    async_model = llm.get_async_model("claude-opus-5")
    assert async_model.model_id == "anthropic/claude-opus-5"
    assert async_model.claude_model_id == "claude-opus-5"


def test_opus_5_kwargs():
    model = llm.get_model("claude-opus-5")
    prompt = llm.Prompt("Hi", model, options=model.Options(thinking_effort="max"))
    kwargs = model.build_kwargs(prompt, None)
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["max_tokens"] == 128000
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"]["effort"] == "max"
    assert "betas" not in kwargs


@pytest.mark.vcr
def test_opus_46_prompt():
    model = llm.get_model("claude-opus-4.6")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt("Two names for a pet pelican, be brief")
    text = response.text()
    assert len(text) > 0
    response_dict = dict(response.response_json)
    assert response_dict["model"] == snapshot("claude-opus-4-6")
    assert response.input_tokens > 0
    assert response.output_tokens > 0


@pytest.mark.vcr
def test_sonnet_46_prompt():
    model = llm.get_model("claude-sonnet-4.6")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt("Two names for a pet pelican, be brief")
    text = response.text()
    assert len(text) > 0
    response_dict = dict(response.response_json)
    assert response_dict["model"] == snapshot("claude-sonnet-4-6")
    assert response.input_tokens > 0
    assert response.output_tokens > 0


@pytest.mark.vcr
def test_opus_46_adaptive_thinking():
    model = llm.get_model("claude-opus-4.6")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt("Two names for a pet pelican, be brief", thinking=True)
    text = response.text()
    assert len(text) > 0
    response_dict = dict(response.response_json)
    # Should have thinking content in the response
    content_types = [block["type"] for block in response_dict["content"]]
    assert "thinking" in content_types
    assert "text" in content_types


@pytest.mark.vcr
def test_sonnet_46_effort_without_thinking():
    model = llm.get_model("claude-sonnet-4.6")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt(
        "Two names for a pet pelican, be brief", thinking_effort="low"
    )
    text = response.text()
    assert len(text) > 0


def test_46_prefill_rejected():
    model = llm.get_model("claude-opus-4.6")
    model.key = "test-key"
    with pytest.raises(
        ValueError, match="Prefilling assistant messages is not supported"
    ):
        model.prompt("Hello", prefill="{").text()


def test_max_effort_passed_through():
    # Client-side validation of effort levels was removed - the API is
    # left to reject models that don't support a given level
    model = llm.get_model("claude-sonnet-4.6")
    prompt = llm.Prompt("Hello", model, options=model.Options(thinking_effort="max"))
    kwargs = model.build_kwargs(prompt, None)
    assert kwargs["output_config"]["effort"] == "max"


@pytest.mark.vcr
def test_opus_46_schema():
    model = llm.get_model("claude-opus-4.6")
    response = model.prompt("Invent a good dog", schema=Dog, key=ANTHROPIC_API_KEY)
    dog = json.loads(response.text())
    assert "name" in dog
    assert "age" in dog
    assert "bio" in dog


# Phase 3: StreamEvent tests
from llm.parts import StreamEvent, TextPart, ReasoningPart, ToolCallPart, ToolResultPart


@pytest.mark.vcr
def test_stream_events_text():
    """stream_events() yields text StreamEvents."""
    model = llm.get_model("claude-haiku-4.5")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt("Say just hello")
    events = list(response.stream_events())
    text_events = [e for e in events if e.type == "text"]
    assert len(text_events) > 0
    text = "".join(e.chunk for e in text_events)
    assert "hello" in text.lower() or "Hello" in text


@pytest.mark.vcr
def test_stream_events_thinking():
    """stream_events() yields reasoning StreamEvents for thinking."""
    model = llm.get_model("claude-haiku-4.5")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt("Two names for a pet pelican, be brief", thinking=True)
    events = list(response.stream_events())
    reasoning_events = [e for e in events if e.type == "reasoning"]
    text_events = [e for e in events if e.type == "text"]
    assert len(reasoning_events) > 0, "Should have reasoning events"
    assert len(text_events) > 0, "Should have text events"
    # Reasoning should be in earlier part_index than text
    assert reasoning_events[0].part_index < text_events[0].part_index


@pytest.mark.vcr
def test_parts_thinking():
    """response.parts includes ReasoningPart and TextPart for thinking responses."""
    model = llm.get_model("claude-haiku-4.5")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt("Two names for a pet pelican, be brief", thinking=True)
    response.text()
    parts = [p for m in response.messages() for p in m.parts]
    reasoning_parts = [p for p in parts if isinstance(p, ReasoningPart)]
    text_parts = [p for p in parts if isinstance(p, TextPart)]
    assert len(reasoning_parts) >= 1, "Should have reasoning part"
    assert len(text_parts) >= 1, "Should have text part"
    assert reasoning_parts[0].provider_metadata["anthropic"]["signature"]
    assert reasoning_parts[0].text, "Reasoning text should not be empty"
    assert text_parts[0].text, "Text should not be empty"


@pytest.mark.vcr
def test_stream_events_tool_calls():
    """stream_events() yields tool call StreamEvents."""
    model = llm.get_model("claude-haiku-4.5")
    model.key = model.key or ANTHROPIC_API_KEY
    names = ["Charles"]
    response = model.prompt(
        "Generate one name for a pet pelican",
        tools=[
            llm.Tool.function(lambda: names.pop(0), name="pelican_name_generator"),
        ],
        key=ANTHROPIC_API_KEY,
    )
    events = list(response.stream_events())
    name_events = [e for e in events if e.type == "tool_call_name"]
    args_events = [e for e in events if e.type == "tool_call_args"]
    assert len(name_events) >= 1, "Should have tool_call_name event"
    assert name_events[0].chunk == snapshot("pelican_name_generator")
    assert name_events[0].tool_call_id is not None


@pytest.mark.vcr
def test_web_search_tool_result_ordering():
    """web_search_tool_result parts appear BEFORE the text that uses them."""
    model = llm.get_model("claude-opus-4.1")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt(
        "What is the current weather in San Francisco?", web_search=True
    )
    events = list(response.stream_events())

    # Find indices of first tool_result and first text event
    tool_result_indices = [i for i, e in enumerate(events) if e.type == "tool_result"]
    text_indices = [
        i
        for i, e in enumerate(events)
        if e.type == "text" and e.chunk.strip()  # non-empty text
    ]
    assert len(tool_result_indices) >= 1, "Should have tool_result events"
    assert len(text_indices) >= 1, "Should have text events"

    # The tool_result should come before the main text content
    first_tool_result = tool_result_indices[0]
    first_substantive_text = text_indices[0]
    assert first_tool_result < first_substantive_text, (
        f"tool_result at index {first_tool_result} should come before "
        f"text at index {first_substantive_text}"
    )

    # Also verify via parts
    parts = [p for m in response.messages() for p in m.parts]
    part_types = [type(p).__name__ for p in parts]
    # ToolResultPart should appear before the main TextParts
    if "ToolResultPart" in part_types and "TextPart" in part_types:
        first_result_idx = part_types.index("ToolResultPart")
        first_text_idx = part_types.index("TextPart")
        assert first_result_idx < first_text_idx


# --- messages= parameter --------------------------------------------------
#
# Unit tests that exercise build_messages directly on messages= input.
# Pure structural — no API calls, so no cassettes.


def _build_messages_for(prompt_kwargs):
    """Invoke build_messages on a one-shot Prompt without hitting the API."""
    model = llm.get_model("claude-sonnet-4.5")
    options = prompt_kwargs.pop("options", model.Options())
    p = llm.Prompt(None, model=model, options=options, **prompt_kwargs)
    return model.build_messages(p, None)


def test_build_messages_simple_user_text():
    from llm import user

    msgs = _build_messages_for({"messages": [user("hi")]})
    assert msgs == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_build_messages_skips_system_role():
    from llm import system, user

    msgs = _build_messages_for({"messages": [system("be nice"), user("hi")]})
    # System does not appear in the messages list; it goes to kwargs["system"].
    assert msgs == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_build_messages_merges_tool_then_user():
    """A tool-role message followed by a user message must collapse into one
    Anthropic user turn (tool_result + text in the same content array)."""
    from llm import user, tool_message
    from llm.parts import ToolResultPart

    msgs = _build_messages_for(
        {
            "messages": [
                tool_message(
                    ToolResultPart(name="search", output="sunny", tool_call_id="call_1")
                ),
                user("thanks"),
            ]
        }
    )
    assert msgs == [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "sunny",
                },
                {"type": "text", "text": "thanks"},
            ],
        }
    ]


def test_build_messages_assistant_tool_call_and_text():
    from llm import assistant, user
    from llm.parts import TextPart, ToolCallPart

    msgs = _build_messages_for(
        {
            "messages": [
                user("what time?"),
                assistant(
                    TextPart(text="Let me check"),
                    ToolCallPart(name="clock", arguments={}, tool_call_id="c1"),
                ),
            ]
        }
    )
    assert msgs == [
        {"role": "user", "content": [{"type": "text", "text": "what time?"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check"},
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "clock",
                    "input": {},
                },
            ],
        },
    ]


def test_build_messages_reasoning_round_trips_signature():
    """Thinking blocks from a prior assistant message must preserve the
    Anthropic signature via provider_metadata — otherwise continuation
    requests involving signed thinking get rejected by the API."""
    from llm import assistant, user
    from llm.parts import ReasoningPart, TextPart

    msgs = _build_messages_for(
        {
            "messages": [
                user("q"),
                assistant(
                    ReasoningPart(
                        text="thinking...",
                        provider_metadata={"anthropic": {"signature": "sig-abc"}},
                    ),
                    TextPart(text="answer"),
                ),
            ]
        }
    )
    assert msgs[1]["content"][0] == {
        "type": "thinking",
        "thinking": "thinking...",
        "signature": "sig-abc",
    }


def test_load_conversation_preserves_logged_tool_chain_for_anthropic(tmp_path):
    """Regression for llm -c after a logged tool call chain.

    LLM 0.32a0 rehydrates the final tool-result response as if its
    prompt.messages started with only the current tool_result turn. That
    makes Anthropic reject the next continuation because the request starts
    with an orphan tool_result instead of the preceding assistant tool_use.
    """
    import datetime
    import sqlite_utils
    from llm.cli import load_conversation
    from llm.migrations import migrate
    from llm.parts import ToolCallPart, ToolResultPart

    model = llm.get_model("claude-haiku-4.5")

    def tick() -> str:
        return "tock"

    tool = llm.Tool.function(tick, name="tick")
    conversation = model.conversation()

    def mark_done(response):
        response._done = True
        response._start = 0.0
        response._end = 0.0
        response._start_utcnow = datetime.datetime.now(datetime.timezone.utc)

    first = llm.Response(
        llm.Prompt("q1", model=model, tools=[tool], options=model.Options()),
        model,
        stream=False,
        conversation=conversation,
    )
    first.add_tool_call(llm.ToolCall(name="tick", arguments={}, tool_call_id="c1"))
    mark_done(first)

    tool_result = llm.ToolResult(name="tick", output="tock", tool_call_id="c1")
    second_chain = [
        llm.user("q1"),
        llm.Message(
            role="assistant",
            parts=[ToolCallPart(name="tick", arguments={}, tool_call_id="c1")],
        ),
        llm.Message(
            role="tool",
            parts=[ToolResultPart(name="tick", output="tock", tool_call_id="c1")],
        ),
    ]
    second = llm.Response(
        llm.Prompt(
            "",
            model=model,
            tools=[tool],
            tool_results=[tool_result],
            messages=second_chain,
            options=model.Options(),
        ),
        model,
        stream=False,
        conversation=conversation,
    )
    second._chunks = ["final answer"]
    second._stream_events = [llm.parts.StreamEvent(type="text", chunk="final answer")]
    mark_done(second)

    db = sqlite_utils.Database(str(tmp_path / "logs.db"))
    migrate(db)
    first.log_to_db(db)
    conversation.responses.append(first)
    second.log_to_db(db)
    conversation.responses.append(second)

    loaded = load_conversation(None, database=str(tmp_path / "logs.db"))
    follow_up = loaded.prompt("q2")
    anthropic_messages = model.build_messages(follow_up.prompt, loaded)

    assert anthropic_messages[0]["content"] == [{"type": "text", "text": "q1"}]
    assert anthropic_messages[1]["content"] == [
        {"type": "tool_use", "id": "c1", "name": "tick", "input": {}}
    ]
    assert anthropic_messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "c1", "content": "tock"}
    ]
    assert anthropic_messages[3]["content"] == [
        {"type": "text", "text": "final answer"}
    ]
    assert anthropic_messages[4]["content"] == [{"type": "text", "text": "q2"}]


def test_extract_system_from_messages():
    from llm import system, user

    model = llm.get_model("claude-sonnet-4.5")
    p = llm.Prompt(None, model=model, messages=[system("be helpful"), user("hi")])
    assert model._extract_system(p) == "be helpful"


def test_extract_system_prefers_prompt_system_over_messages():
    """When both paths are populated (synthesized case), prompt.system wins
    since it already composes system= + system_fragments."""
    from llm import user

    model = llm.get_model("claude-sonnet-4.5")
    p = llm.Prompt(None, model=model, system="legacy sys", messages=[user("hi")])
    assert model._extract_system(p) == "legacy sys"


# --- WebFetch server-side tool --------------------------------------------


def test_web_fetch_supported_server_side_tools():
    from llm_anthropic import WebFetch

    model = llm.get_model("claude-sonnet-4.6")
    assert WebFetch in model.supported_server_side_tools
    # Models without web search support don't claim WebFetch either
    old_model = llm.get_model("claude-3-opus")
    assert WebFetch not in old_model.supported_server_side_tools


def test_web_fetch_kwargs():
    from llm_anthropic import WebFetch

    model = llm.get_model("claude-sonnet-4.6")
    prompt = llm.Prompt(
        "Fetch https://www.example.com/",
        model,
        options=model.Options(),
        tools=[
            WebFetch(
                max_uses=3,
                allowed_domains=["example.com"],
                citations=True,
                max_content_tokens=5000,
            )
        ],
    )
    kwargs = model.build_kwargs(prompt, None)
    assert kwargs["tools"] == [
        {
            "type": "web_fetch_20260318",
            "name": "web_fetch",
            "max_uses": 3,
            "allowed_domains": ["example.com"],
            "citations": {"enabled": True},
            "max_content_tokens": 5000,
        }
    ]


def test_web_fetch_kwargs_basic_version_on_older_model():
    from llm_anthropic import WebFetch

    model = llm.get_model("claude-opus-4.1")
    prompt = llm.Prompt(
        "Fetch https://www.example.com/",
        model,
        options=model.Options(),
        tools=[WebFetch()],
    )
    kwargs = model.build_kwargs(prompt, None)
    assert kwargs["tools"] == [{"type": "web_fetch_20250910", "name": "web_fetch"}]


def test_web_fetch_use_cache_requires_newer_model():
    from llm_anthropic import WebFetch

    model = llm.get_model("claude-opus-4.1")
    prompt = llm.Prompt(
        "Fetch https://www.example.com/",
        model,
        options=model.Options(),
        tools=[WebFetch(use_cache=False)],
    )
    with pytest.raises(ValueError):
        model.build_kwargs(prompt, None)
    # Fine on a 4.6 model
    model46 = llm.get_model("claude-sonnet-4.6")
    prompt46 = llm.Prompt(
        "Fetch https://www.example.com/",
        model46,
        options=model46.Options(),
        tools=[WebFetch(use_cache=False)],
    )
    kwargs = model46.build_kwargs(prompt46, None)
    assert kwargs["tools"][0]["use_cache"] is False


def test_web_fetch_unsupported_model_raises():
    from llm_anthropic import WebFetch

    model = llm.get_model("claude-3-opus")
    prompt = llm.Prompt(
        "Fetch https://www.example.com/",
        model,
        options=model.Options(),
        tools=[WebFetch()],
    )
    with pytest.raises(ValueError, match="does not support server-side tool"):
        model.build_kwargs(prompt, None)


def test_web_fetch_alongside_client_tools_kwargs():
    from llm_anthropic import WebFetch

    model = llm.get_model("claude-sonnet-4.6")
    prompt = llm.Prompt(
        "Fetch and report version",
        model,
        options=model.Options(),
        tools=[WebFetch(max_uses=1), fixed_version_tool()],
    )
    kwargs = model.build_kwargs(prompt, None)
    types_and_names = [(tool.get("type"), tool.get("name")) for tool in kwargs["tools"]]
    assert ("web_fetch_20260318", "web_fetch") in types_and_names
    assert (None, "fixed_version") in types_and_names


@pytest.mark.vcr
def test_web_fetch():
    from llm_anthropic import WebFetch

    model = llm.get_model("claude-sonnet-4.6")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt(
        "Fetch https://www.example.com/ and quote the first sentence of the page",
        tools=[WebFetch(max_uses=1)],
    )
    text = str(response)
    assert "example" in text.lower()
    # Server-executed tool call and result should be captured as parts.
    # Dynamic filtering means the model may also make code_execution
    # server tool calls around the fetch itself.
    parts = [p for m in response.messages() for p in m.parts]
    tool_calls = [p for p in parts if isinstance(p, llm.parts.ToolCallPart)]
    web_fetch_calls = [p for p in tool_calls if p.name == "web_fetch"]
    assert web_fetch_calls
    assert web_fetch_calls[0].server_executed
    fetch_results = []
    for part in parts:
        if not isinstance(part, llm.parts.ToolResultPart) or not part.output:
            continue
        try:
            data = json.loads(part.output)
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("type") == "web_fetch_result":
            fetch_results.append(data)
    assert fetch_results
    assert fetch_results[0]["url"] == "https://www.example.com/"


def test_build_messages_replays_server_tool_blocks():
    """Server-executed tool calls/results replay as server_tool_use and
    provider result blocks inside the assistant turn - not as client
    tool_use/tool_result blocks, which the API rejects."""
    from llm.parts import Message, TextPart, ToolCallPart, ToolResultPart
    from llm import user

    model = llm.get_model("claude-sonnet-4.6")
    result_payload = {
        "type": "web_fetch_result",
        "url": "https://www.example.com/",
        "content": {"type": "document"},
    }
    msgs = [
        user("Fetch https://www.example.com/ and tell me its title"),
        Message(
            role="assistant",
            parts=[
                ToolCallPart(
                    name="web_fetch",
                    arguments={"url": "https://www.example.com/"},
                    tool_call_id="srvtoolu_123",
                    server_executed=True,
                ),
                ToolResultPart(
                    name="web_fetch",
                    output=json.dumps(result_payload),
                    tool_call_id="srvtoolu_123",
                    server_executed=True,
                ),
                TextPart(text="The title is Example Domain"),
            ],
        ),
        user("What domain was that page on?"),
    ]
    prompt = llm.Prompt(None, model, options=model.Options(), messages=msgs)
    messages = model.build_messages(prompt, None)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assistant_blocks = messages[1]["content"]
    assert [b["type"] for b in assistant_blocks] == [
        "server_tool_use",
        "web_fetch_tool_result",
        "text",
    ]
    assert assistant_blocks[0]["name"] == "web_fetch"
    assert assistant_blocks[0]["id"] == "srvtoolu_123"
    assert assistant_blocks[1]["tool_use_id"] == "srvtoolu_123"
    assert assistant_blocks[1]["content"] == result_payload


def test_sonnet_and_haiku_4_5_support_web_search():
    from llm_anthropic import WebFetch

    for model_id in ("claude-sonnet-4.5", "claude-haiku-4.5"):
        model = llm.get_model(model_id)
        assert model.supports_web_search
        assert WebFetch in model.supported_server_side_tools


def test_sonnet_5_registered():
    model = llm.get_model("claude-sonnet-5")
    assert model.model_id == "anthropic/claude-sonnet-5"
    assert model.claude_model_id == "claude-sonnet-5"
    assert "application/pdf" in model.attachment_types
    assert model.supports_thinking
    assert model.supports_thinking_effort
    assert model.supports_adaptive_thinking
    assert model.supports_web_search
    assert model.use_structured_outputs
    assert model.default_max_tokens == 128000
    async_model = llm.get_async_model("claude-sonnet-5")
    assert async_model.supports_web_search
    assert async_model.default_max_tokens == 128000
