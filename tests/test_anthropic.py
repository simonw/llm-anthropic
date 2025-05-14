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
def test_image_with_no_prompt():
    model = llm.get_model("claude-3.5-sonnet")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt(
        prompt=None,
        attachments=[llm.Attachment(content=TINY_PNG)],
    )
    assert str(response) == (
        "This image shows two simple rectangular blocks of solid colors stacked "
        "vertically. The top rectangle is colored in a bright red, while the "
        "bottom rectangle is colored in a vibrant lime green. They appear to "
        "be of similar width but the green rectangle seems slightly taller "
        "than the red one. The colors are very saturated and pure, creating "
        "a strong visual contrast between the two blocks."
    )


@pytest.mark.vcr
def test_url_prompt():
    model = llm.get_model("claude-3.5-sonnet")
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
        "This image shows a Brown Pelican perched on rocky ground near what appears "
        "to be a marina or harbor, with boats visible in the background. The pelican "
        "is captured in a profile view, showing off its distinctive long beak and "
        "throat pouch. The bird's feathers appear to be a grayish-brown color, and "
        "there's a nice rim lighting effect around its head and neck created by what "
        "seems to be backlighting from the sun. The pelican's posture is upright and "
        "alert, which is typical for these coastal birds. The setting suggests this "
        "is likely taken at a coastal location where pelicans commonly gather to rest "
        "and fish."
    )


class Dog(BaseModel):
    name: str
    age: int
    bio: str


@pytest.mark.vcr
def test_schema_prompt():
    model = llm.get_model("claude-3.7-sonnet")

    response = model.prompt("Invent a good dog", schema=Dog, key=ANTHROPIC_API_KEY)
    dog = json.loads(response.text())
    assert dog == {
        "name": "Buddy",
        "age": 3,
        "bio": "Buddy is a loyal and energetic Golden Retriever who loves long walks.",
    }


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_schema_prompt_async():
    model = llm.get_async_model("claude-3.7-sonnet")
    response = await model.prompt(
        "Invent a terrific dog", schema=Dog, key=ANTHROPIC_API_KEY
    )
    dog_json = await response.text()
    dog = json.loads(dog_json)
    assert dog == {
        "name": "Maximus Thunder",
        "age": 3,
        "bio": (
            "Maximus Thunder is an extraordinary golden retriever with a natural "
            "talent for search and rescue operations. His keen sense of smell "
            "can detect people trapped under debris from over a mile away. When "
            "he's not saving lives, Maximus enjoys surfing at the beach and has "
            "won three local dog surfing competitions. He's also incredibly "
            "gentle with children and regularly visits hospitals as a therapy "
            "dog. His favorite treat is peanut butter, and he has a unique howl "
            'that sounds remarkably like he\'s saying "hello."'
        ),
    }


@pytest.mark.vcr
def test_prompt_with_prefill_and_stop_sequences():
    model = llm.get_model("claude-3.5-haiku")
    response = model.prompt(
        "Very short function describing a pelican",
        prefill="```python",
        stop_sequences=["```"],
        hide_prefill=True,
        key=ANTHROPIC_API_KEY,
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
    conversation = model.conversation()
    response = conversation.prompt(
        "Two names for a pet pelican, be brief", thinking=True, key=ANTHROPIC_API_KEY
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


@pytest.mark.vcr
def test_tools():
    model = llm.get_model("claude-3.5-haiku")
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
        "I'll help you generate two names for a pet pelican using the pelican name generator tool. "
        "Here are two fun names for your pet pelican:\n"
        "1. Charles - A distinguished, classic name that gives your pelican a bit of sophistication.\n"
        "2. Sammy - A friendly, playful name that suggests a cheerful and approachable personality.\n\n"
        "Would you like me to generate more names or do you like these? Each pelican name can have "
        "its own unique charm, so feel free to ask for more suggestions!"
    )
    tool_calls = chain_response._responses[0].tool_calls()
    assert len(tool_calls) == 2
    assert all(call.name == "pelican_name_generator" for call in tool_calls)
    assert [
        result.output for result in chain_response._responses[1].prompt.tool_results
    ] == ["Charles", "Sammy"]
