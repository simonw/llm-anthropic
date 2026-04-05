import json
import llm
import os
import pytest
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


@pytest.mark.vcr
def test_prompt():
    model = llm.get_model("claude-sonnet-4.5")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt("Two names for a pet pelican, be brief")
    assert str(response) == "- Captain\n- Scoop"
    response_dict = dict(response.response_json)
    response_dict.pop("id")  # differs between requests
    assert response_dict == {
        "container": None,
        "content": [{"citations": None, "parsed_output": None, "text": "- Captain\n- Scoop", "type": "text"}],
        "model": "claude-sonnet-4-5-20250929",
        "role": "assistant",
        "stop_details": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
    }
    assert response.input_tokens == 17
    assert response.output_tokens == 10
    assert response.token_details is None


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_prompt():
    model = llm.get_async_model("claude-sonnet-4.5")
    model.key = model.key or ANTHROPIC_API_KEY  # don't override existing key
    conversation = model.conversation()
    response = await conversation.prompt("Two names for a pet pelican, be brief")
    assert await response.text() == "- Captain\n- Scoop"
    response_dict = dict(response.response_json)
    response_dict.pop("id")  # differs between requests
    assert response_dict == {
        "container": None,
        "content": [{"citations": None, "parsed_output": None, "text": "- Captain\n- Scoop", "type": "text"}],
        "model": "claude-sonnet-4-5-20250929",
        "role": "assistant",
        "stop_details": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
    }
    assert response.input_tokens == 17
    assert response.output_tokens == 10
    assert response.token_details is None
    response2 = await conversation.prompt("in french")
    assert await response2.text() == "- Capitaine\n- Bec (beak)"


EXPECTED_IMAGE_TEXT = "Red square, green square."


@pytest.mark.vcr
def test_image_prompt():
    model = llm.get_model("claude-sonnet-4.5")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt(
        "Describe image in three words",
        attachments=[llm.Attachment(content=TINY_PNG)],
    )
    assert str(response) == EXPECTED_IMAGE_TEXT
    response_dict = response.response_json
    response_dict.pop("id")  # differs between requests
    assert response_dict == {
        "container": None,
        "content": [{"citations": None, "parsed_output": None, "text": EXPECTED_IMAGE_TEXT, "type": "text"}],
        "model": "claude-sonnet-4-5-20250929",
        "role": "assistant",
        "stop_details": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
    }

    assert response.input_tokens == 83
    assert response.output_tokens == 9
    assert response.token_details is None


@pytest.mark.vcr
def test_image_with_no_prompt():
    model = llm.get_model("claude-sonnet-4.5")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt(
        prompt=None,
        attachments=[llm.Attachment(content=TINY_PNG)],
    )
    assert str(response) == (
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
    assert str(response) == (
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
    assert dog == {
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


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_schema_prompt_async():
    model = llm.get_async_model("claude-sonnet-4.5")
    response = await model.prompt(
        "Invent a terrific dog", schema=Dog, key=ANTHROPIC_API_KEY
    )
    dog_json = await response.text()
    dog = json.loads(dog_json)
    assert dog == {
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
    assert text == (
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
    assert response.text() == "- Captain\n- Scoop"
    response_dict = dict(response.response_json)
    response_dict.pop("id")  # differs between requests
    assert response_dict == {
        "container": None,
        "content": [
            {
                "signature": "EvoCCkYICxgCKkCY593DZILKPT3+cK6Py3Hcu8WYGSW2vk6yge13JAuAYlQFzUC2c6UlsIWiBMvJjtDc6zfh73EOpWqxYB+pOh5zEgxoquY58aQkL5YJ9YUaDJpBRqMUVxU6SPYLRCIwcxSHXlhIWQw0QhiPV+lrMXgTdk4SuJnWVpHc+uTSxYmxetnHugsdCo6xWShRJaQLKuEBsfQzKHid0WmA8hdgz0mVpI+nAPvnhpHFIZbnGl2HQi7cc87o/HZzCod2tF8mEuqdyNtPHgx+H+Zyr/Pi0kmJF6hOoylmR5nGrXeqR1ttWFzzPF7XpmIr+vw3y/rNCDQVRWxmG/IDyHVLKXtALaFnDpNfvSGv6L3udrgkeWuV2VEzfGPclEIaailiMsxesYC/LJGUcPlqZ9kL18GQ16lr15u6kOUMlyYKA7AoeL6K7U3qCvpSdfGpMfgkasK5ugj0FL3UgFpUrrFbPlpLAdLsNeG2G+XBiOY0TtfokCiI1P7LGAE=",
                "thinking": "The user wants two names for a pet pelican, and wants me to be brief. I'll give two simple, fitting names.\n\nSome options:\n- Pete\n- Percy\n- Captain\n- Scoop\n- Bill\n- Gully\n\nI'll pick two good ones and keep it very short.",
                "type": "thinking",
            },
            {"citations": None, "parsed_output": None, "text": "- Captain\n- Scoop", "type": "text"},
        ],
        "model": "claude-sonnet-4-5-20250929",
        "role": "assistant",
        "stop_details": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
    }

    assert response.input_tokens == 46
    assert response.output_tokens == 84
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
    assert text == (
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
    ] == ["Charles", "Sammy"]


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


@pytest.mark.vcr
def test_opus_46_prompt():
    model = llm.get_model("claude-opus-4.6")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt("Two names for a pet pelican, be brief")
    text = response.text()
    assert len(text) > 0
    response_dict = dict(response.response_json)
    assert response_dict["model"] == "claude-opus-4-6"
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
    assert response_dict["model"] == "claude-sonnet-4-6"
    assert response.input_tokens > 0
    assert response.output_tokens > 0


@pytest.mark.vcr
def test_opus_46_adaptive_thinking():
    model = llm.get_model("claude-opus-4.6")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt(
        "Two names for a pet pelican, be brief", thinking=True
    )
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
    with pytest.raises(ValueError, match="Prefilling assistant messages is not supported"):
        model.prompt("Hello", prefill="{").text()


def test_46_max_effort_opus_only():
    model = llm.get_model("claude-sonnet-4.6")
    model.key = "test-key"
    with pytest.raises(ValueError, match="thinking_effort='max' is only supported"):
        model.prompt("Hello", thinking_effort="max").text()


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
    parts = response.parts
    reasoning_parts = [p for p in parts if isinstance(p, ReasoningPart)]
    text_parts = [p for p in parts if isinstance(p, TextPart)]
    assert len(reasoning_parts) >= 1, "Should have reasoning part"
    assert len(text_parts) >= 1, "Should have text part"
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
    assert name_events[0].chunk == "pelican_name_generator"
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
    tool_result_indices = [
        i for i, e in enumerate(events) if e.type == "tool_result"
    ]
    text_indices = [
        i for i, e in enumerate(events)
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
    parts = response.parts
    part_types = [type(p).__name__ for p in parts]
    # ToolResultPart should appear before the main TextParts
    if "ToolResultPart" in part_types and "TextPart" in part_types:
        first_result_idx = part_types.index("ToolResultPart")
        first_text_idx = part_types.index("TextPart")
        assert first_result_idx < first_text_idx
