# llm-anthropic

[![PyPI](https://img.shields.io/pypi/v/llm-anthropic.svg)](https://pypi.org/project/llm-anthropic/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-anthropic?include_prereleases&label=changelog)](https://github.com/simonw/llm-anthropic/releases)
[![Tests](https://github.com/simonw/llm-anthropic/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/llm-anthropic/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-anthropic/blob/main/LICENSE)

LLM access to models by Anthropic, including the Claude series

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-anthropic
```

<details><summary>Instructions for users who need to upgrade from <code>llm-claude-3</code></summary>

<br>

If you previously used `llm-claude-3` you can upgrade like this:

```bash
llm install -U llm-claude-3
llm keys set anthropic --value "$(llm keys get claude)"
```
The first line will remove the previous `llm-claude-3` version and install this one, because the latest `llm-claude-3` depends on `llm-anthropic`.

The second line sets the `anthropic` key to whatever value you previously used for the `claude` key.

</details>

## Usage

First, set [an API key](https://console.anthropic.com/settings/keys) for Anthropic:
```bash
llm keys set anthropic
# Paste key here
```

You can also set the key in the environment variable `ANTHROPIC_API_KEY`

Run `llm models` to list the models, and `llm models --options` to include a list of their options.

Run prompts like this:
```bash
llm -m claude-4-opus 'Fun facts about walruses'
llm -m claude-4-sonnet 'Fun facts about pelicans'
llm -m claude-3.5-haiku 'Fun facts about armadillos'
```
Image attachments are supported too:
```bash
llm -m claude-3.5-sonnet 'describe this image' -a https://static.simonwillison.net/static/2024/pelicans.jpg
llm -m claude-3-haiku 'extract text' -a page.png
```
The Claude 3.5 and 3.7 and 4 models can handle PDF files:
```bash
llm -m claude-3.5-sonnet 'extract text' -a page.pdf
```
Anthropic's models support [schemas](https://llm.datasette.io/en/stable/schemas.html). Here's how to use Claude 3.7 Sonnet to invent a dog:

```bash
llm -m claude-3.7-sonnet --schema 'name,age int,bio: one sentence' 'invent a surprising dog'
```
Example output:
```json
{
  "name": "Whiskers the Mathematical Mastiff",
  "age": 7,
  "bio": "Whiskers is a mastiff who can solve complex calculus problems by barking in binary code and has won three international mathematics competitions against human competitors."
}
```

## Extended reasoning with Claude 3.7 Sonnet

Claude 3.7 introduced [extended thinking](https://www.anthropic.com/news/visible-extended-thinking) mode, where Claude can expend extra effort thinking through the prompt before producing a response.

Use the `-o thinking 1` option to enable this feature:

```bash
llm -m claude-3.7-sonnet -o thinking 1 'Write a convincing speech to congress about the need to protect the California Brown Pelican'
```
The chain of thought is not currently visible while using LLM, but it is logged to the database and can be viewed using this command:
```bash
llm logs -c --json
```
Or in combination with `jq`:
```bash
llm logs --json -c | jq '.[0].response_json.content[0].thinking' -r
```
By default up to 1024 tokens can be used for thinking. You can increase this budget with the `thinking_budget` option:
```bash
llm -m claude-3.7-sonnet -o thinking_budget 32000 'Write a long speech about pelicans in French'
```

## Model options

The following options can be passed using `-o name value` on the CLI or as `keyword=value` arguments to the Python `model.prompt()` method:

<!-- [[[cog
import cog, llm
_type_lookup = {
    "number": "float",
    "integer": "int",
    "string": "str",
    "object": "dict",
}

model = llm.get_model("claude-3.7-sonnet")
output = []
for name, field in model.Options.schema()["properties"].items():
    any_of = field.get("anyOf")
    if any_of is None:
        any_of = [{"type": field["type"]}]
    types = ", ".join(
        [
            _type_lookup.get(item["type"], item["type"])
            for item in any_of
            if item["type"] != "null"
        ]
    )
    bits = ["- **", name, "**: `", types, "`\n"]
    description = field.get("description", "")
    if description:
        bits.append('\n    ' + description + '\n\n')
    output.append("".join(bits))
cog.out("".join(output))
]]] -->
- **max_tokens**: `int`

    The maximum number of tokens to generate before stopping

- **temperature**: `float`

    Amount of randomness injected into the response. Defaults to 1.0. Ranges from 0.0 to 1.0. Use temperature closer to 0.0 for analytical / multiple choice, and closer to 1.0 for creative and generative tasks. Note that even with temperature of 0.0, the results will not be fully deterministic.

- **top_p**: `float`

    Use nucleus sampling. In nucleus sampling, we compute the cumulative distribution over all the options for each subsequent token in decreasing probability order and cut it off once it reaches a particular probability specified by top_p. You should either alter temperature or top_p, but not both. Recommended for advanced use cases only. You usually only need to use temperature.

- **top_k**: `int`

    Only sample from the top K options for each subsequent token. Used to remove 'long tail' low probability responses. Recommended for advanced use cases only. You usually only need to use temperature.

- **user_id**: `str`

    An external identifier for the user who is associated with the request

- **prefill**: `str`

    A prefill to use for the response

- **hide_prefill**: `boolean`

    Do not repeat the prefill value at the start of the response

- **stop_sequences**: `array, str`

    Custom text sequences that will cause the model to stop generating - pass either a list of strings or a single string

- **cache**: `boolean`

    Use Anthropic prompt cache for any attachments or fragments

- **thinking**: `boolean`

    Enable thinking mode

- **thinking_budget**: `int`

    Number of tokens to budget for thinking

<!-- [[[end]]] -->

The `prefill` option can be used to set the first part of the response. To increase the chance of returning JSON, set that to `{`:

```bash
llm -m claude-3.5-sonnet 'Fun data about pelicans' \
  -o prefill '{'
```
If you do not want the prefill token to be echoed in the response, set `hide_prefill` to `true`:

```bash
llm -m claude-3.5-haiku 'Short python function describing a pelican' \
  -o prefill '```python' \
  -o hide_prefill true \
  -o stop_sequences '```'
```
This example sets `` ``` `` as the stop sequence, so the response will be a Python function without the wrapping Markdown code block.

To pass a single stop sequence, send a string:
```bash
llm -m claude-3.5-sonnet 'Fun facts about pelicans' \
  -o stop-sequences "beak"
```
For multiple stop sequences, pass a JSON array:

```bash
llm -m claude-3.5-sonnet 'Fun facts about pelicans' \
  -o stop-sequences '["beak", "feathers"]'
```

When using the Python API, pass a string or an array of strings:

```python
response = llm.query(
    model="claude-3.5-sonnet",
    query="Fun facts about pelicans",
    stop_sequences=["beak", "feathers"],
)
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-anthropic
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
pytest
```

This project uses [pytest-recording](https://github.com/kiwicom/pytest-recording) to record Anthropic API responses for the tests.

If you add a new test that calls the API you can capture the API response like this:
```bash
PYTEST_ANTHROPIC_API_KEY="$(llm keys get anthropic)" pytest --record-mode once
```
You will need to have stored a valid Anthropic API key using this command first:
```bash
llm keys set anthropic
# Paste key here
```
I use the following sequence:
```bash
# First delete the relevant cassette if it exists already:
rm tests/cassettes/test_anthropic/test_thinking_prompt.yaml
# Run this failing test to recreate the cassette
PYTEST_ANTHROPIC_API_KEY="$(llm keys get claude)" pytest -k test_thinking_prompt --record-mode once
# Now run the test again with --pdb to figure out how to update it
pytest -k test_thinking_prompt --pdb
# Edit test
```