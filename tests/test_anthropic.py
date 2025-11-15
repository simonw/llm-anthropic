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
        "content": [{"citations": None, "text": "- Captain\n- Scoop", "type": "text"}],
        "model": "claude-sonnet-4-5-20250929",
        "role": "assistant",
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
        "content": [{"citations": None, "text": "- Captain\n- Scoop", "type": "text"}],
        "model": "claude-sonnet-4-5-20250929",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
    }
    assert response.input_tokens == 17
    assert response.output_tokens == 10
    assert response.token_details is None
    response2 = await conversation.prompt("in french")
    assert await response2.text() == '- Capitaine\n- Bec (meaning "beak")'


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
        "content": [{"citations": None, "text": EXPECTED_IMAGE_TEXT, "type": "text"}],
        "model": "claude-sonnet-4-5-20250929",
        "role": "assistant",
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
            "Biscuit is a loyal golden retriever with a gentle temperament and "
            "boundless enthusiasm for life. He loves swimming in lakes, playing fetch "
            "until sunset, and has an uncanny ability to sense when someone needs "
            "comfort. Known in his neighborhood for his friendly demeanor, Biscuit "
            "volunteers at the local children's hospital bringing joy to young "
            "patients."
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
        "age": 3,
        "bio": (
            "Biscuit is a golden retriever with a heart of pure sunshine. This "
            "enthusiastic pup has mastered over 50 tricks, volunteers at the local "
            "children's hospital bringing smiles to young patients, and has an uncanny "
            "ability to sense when someone needs comfort. With her flowing golden "
            "coat, perpetually wagging tail, and gentle brown eyes, Biscuit embodies "
            "loyalty and joy. She loves swimming in lakes, playing fetch until sunset, "
            "and curling up for story time with her family."
        ),
        "name": "Biscuit",
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
    assert text == (
        "\ndef describe_pelican():\n"
        '    """Returns a brief description of a pelican."""\n'
        '    return "Large seabird with a massive bill, capable of scooping fish from water."\n'
    )


@pytest.mark.vcr
def test_thinking_prompt():
    model = llm.get_model("claude-sonnet-4.5")
    conversation = model.conversation()
    response = conversation.prompt(
        "Two names for a pet pelican, be brief", thinking=True, key=ANTHROPIC_API_KEY
    )
    assert response.text() == "Bill\nScoop"
    response_dict = dict(response.response_json)
    response_dict.pop("id")  # differs between requests
    assert response_dict == {
        "content": [
            {
                "signature": "ErUBCkYICRgCIkCUCsMUICiFm+sgvS255wUaTjAJEX2tc5h+Mir6Kq6OozIF+9a3ygFFnCLjPYf2Jl18eMVqkYVs0Vq9rRJpl6N/EgySPIWO4SVcxV0VqecaDM6REdwo/8lOJenaQCIwzQfkXeoR1nwYqsvrQsf4/NwhTuKfWDtM8a0XHfoH7EFwizaRuTrwV21Ny1nWKbu2Kh2qjLQOYn34/0pMKErgEexGTvXvn5PMMSAqOVqz8BgC",
                "thinking": "I need to provide two brief names for a pet pelican. Pelicans are large water birds with distinctive pouched bills, so I could create names that relate to:\n- Their appearance (bill, pouch)\n- Water/ocean themes\n- Their fishing abilities\n- Classic pet names with a twist\n\nI'll provide two short, creative names without additional commentary, as the request asks me to be brief.",
                "type": "thinking",
            },
            {"citations": None, "text": "Bill\nScoop", "type": "text"},
        ],
        "model": "claude-3-7-sonnet-20250219",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
    }

    assert response.input_tokens == 46
    assert response.output_tokens == 103
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
        "I'll help you generate two potential names for a pet pelican by using the pelican "
        "name generator tool. Great! The tool has generated two fun names for a pet pelican:\n"
        "1. Charles - A dignified name that could suit a sophisticated pelican\n"
        "2. Sammy - A friendly and playful name that gives your pelican a cute personality\n\n"
        "Would you like me to generate more names or do you like these options?"
    )
    tool_calls = chain_response._responses[0].tool_calls()
    assert len(tool_calls) == 2
    assert all(call.name == "pelican_name_generator" for call in tool_calls)
    assert [
        result.output for result in chain_response._responses[1].prompt.tool_results
    ] == ["Charles", "Sammy"]


@pytest.mark.vcr
def test_web_search():
    model = llm.get_model("claude-3.5-sonnet")
    model.key = model.key or ANTHROPIC_API_KEY
    response = model.prompt(
        "What is the current weather in San Francisco?",
        web_search=True
    )
    response_text = str(response)
    assert len(response_text) > 0
    assert any(word in response_text.lower() for word in ["weather", "temperature", "san francisco", "degree", "forecast"])
    response_dict = dict(response.response_json)
    assert "content" in response_dict
    assert len(response_dict["content"]) > 0