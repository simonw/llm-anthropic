import llm
import os
import pytest

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
    model = llm.get_model("claude-3-opus")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt("Two names for a pet pelican, be brief")
    assert str(response) == "1. Pelly\n2. Beaky"
    response_dict = dict(response.response_json)
    response_dict.pop("id")  # differs between requests
    assert response_dict == {
        "content": [{"citations": None, "text": "1. Pelly\n2. Beaky", "type": "text"}],
        "model": "claude-3-opus-20240229",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
    }
    assert response.input_tokens == 17
    assert response.output_tokens == 15
    assert response.token_details is None


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_prompt():
    model = llm.get_async_model("claude-3-opus")
    model.key = model.key or ANTHROPIC_API_KEY  # don't override existing key
    conversation = model.conversation()
    response = await conversation.prompt("Two names for a pet pelican, be brief")
    assert await response.text() == "1. Pelly\n2. Beaky"
    response_dict = dict(response.response_json)
    response_dict.pop("id")  # differs between requests
    assert response_dict == {
        "content": [{"citations": None, "text": "1. Pelly\n2. Beaky", "type": "text"}],
        "model": "claude-3-opus-20240229",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
    }
    assert response.input_tokens == 17
    assert response.output_tokens == 15
    assert response.token_details is None
    # Try a reply
    response2 = await conversation.prompt("in french")
    assert await response2.text() == "1. Pélican\n2. Bec"


EXPECTED_IMAGE_TEXT = (
    "This image shows two simple rectangular blocks of solid colors stacked "
    "vertically. The top rectangle is a bright, vibrant red color, while the "
    "bottom rectangle is a bright, neon green color. The rectangles appear to "
    "be of similar width but may be slightly different in height. The colors "
    "are very saturated and create a striking contrast against each other."
)


@pytest.mark.vcr
def test_image_prompt():
    model = llm.get_model("claude-3.5-sonnet")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt(
        "Describe image in three words",
        attachments=[llm.Attachment(content=TINY_PNG)],
    )
    assert str(response) == EXPECTED_IMAGE_TEXT
    response_dict = response.response_json
    response_dict.pop("id")  # differs between requests
    assert response_dict == {
        "content": [{"citations": None, "text": EXPECTED_IMAGE_TEXT, "type": "text"}],
        "model": "claude-3-5-sonnet-20241022",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
    }

    assert response.input_tokens == 76
    assert response.output_tokens == 75
    assert response.token_details is None


@pytest.mark.vcr
def test_prompt_with_prefill_and_stop_sequences():
    model = llm.get_model("claude-3.5-haiku")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt(
        "Very short function describing a pelican",
        prefill="```python",
        stop_sequences=["```"],
        hide_prefill=True,
    )
    text = response.text()
    assert text.startswith(
        "\ndef describe_pelican():\n"
        '    """\n'
        "    A function describing the characteristics of a pelican.\n"
        "    \n"
        "    Returns:\n"
        "        A dictionary with various details about pelicans.\n"
        '    """\n'
        "    pelican_details = {\n"
        '        "species": "Pelecanus",\n'
        '        "habitat": "Coastal areas, lakes, and rivers",\n'
    )
    assert text.endswith(
        'print("Distinctive Features:", '
        'pelican_info["physical_characteristics"]["distinctive_features"])\n'
    )


@pytest.mark.vcr
def test_thinking_prompt():
    model = llm.get_model("claude-3.7-sonnet")
    model.key = model.key or ANTHROPIC_API_KEY
    conversation = model.conversation()
    response = conversation.prompt(
        "Two names for a pet pelican, be brief", thinking=True
    )
    assert response.text() == "Scoop and Beaky"
    response_dict = dict(response.response_json)
    response_dict.pop("id")  # differs between requests
    assert response_dict == {
        "content": [
            {
                "signature": "EuYBCkQYAiJAZFFuoMD/kqVzMZ887Sa1rJBpa5UU5W+YVHe0PV1dh0T1ZHHOQcSUTMB9iPC6hhduyszf501Ao1McU4sUwlL2UhIM0nyNDklwN6dy0bkUGgx7Ny7JpGHlGJ3+mR8iMHcNBzvnVJwp6XmCs9jieB8BWgth2vmVOuSU+mUYw2bT4pkzVkVsxFnA1lh2T1kjRSpQltDXxi/Pyq3WdD/W4gnV9HIJ4Cb5olNXUrMvKUyoim0MfvyOU7wuyAi7J74CVw0Te6DW8GQf3/1jVYxeMEEBszuSU5IuyxB0BKWW5TfALM0=",
                "thinking": "The person is asking for two names for a pet pelican, and they want me to be brief in my response. I'll provide two concise, creative names that would suit a pelican:\n\n1. Something that relates to their large beak/pouch\n2. Something that relates to water/fishing\n\nI'll keep my response very short as requested.",
                "type": "thinking",
            },
            {"citations": None, "text": "Scoop and Beaky", "type": "text"},
        ],
        "model": "claude-3-7-sonnet-20250219",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
    }

    assert response.input_tokens == 45
    assert response.output_tokens == 94
    assert response.token_details is None
