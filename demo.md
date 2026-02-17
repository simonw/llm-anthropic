# Claude 4.6 Models in llm-anthropic

*2026-02-17T21:32:54Z by Showboat 0.6.0*
<!-- showboat-id: aa3e535d-4134-4254-89fe-c1d1149d9ef3 -->

This document demonstrates the newly added Claude Opus 4.6 and Sonnet 4.6 models in the llm-anthropic plugin, covering adaptive thinking, the GA effort parameter, structured outputs, web search, and breaking changes like prefill removal.

## Listing the new models

Both models are registered with dot-notation aliases for convenience.

```bash
uv run llm models | grep "4\.6"
```

```output
Anthropic Messages: anthropic/claude-opus-4-6 (aliases: claude-opus-4.6)
Anthropic Messages: anthropic/claude-sonnet-4-6 (aliases: claude-sonnet-4.6)
```

## Basic prompting with Sonnet 4.6

Sonnet 4.6 is the fastest of the two new models. Here it responds to a simple prompt.

```bash
uv run llm -m claude-sonnet-4.6 "Two names for a pet pelican, be brief"
```

```output
**Pete** or **Scoop**
```

## Structured outputs (schema)

4.6 models use the new `output_config.format` API for structured outputs — no beta header required. The `--schema` flag generates JSON conforming to the given schema.

```bash
uv run llm -m claude-sonnet-4.6 --schema 'name,age int,bio: one sentence' 'invent a dog'
```

```output
{"name":"Biscuit McWoofington","age":4,"bio":"Biscuit is a scruffy golden retriever mix who spends his days chasing squirrels, stealing socks, and melting hearts with his lopsided grin."}
```

## Adaptive thinking

4.6 models use adaptive thinking (`thinking: {type: "adaptive"}`) instead of the old manual budget mode. Claude dynamically decides when and how much to think. Enabled with `-o thinking 1`.

```bash
uv run llm -m claude-sonnet-4.6 -o thinking 1 "What is the square root of 97 to 3 decimal places?"
```

```output
## √97 = **9.849**

**Working it out:**

- 9.8² = 96.04
- 9.9² = 98.01 → so √97 is between 9.8 and 9.9

Narrowing down:
- 9.84² = 96.8256
- 9.85² = 97.0225 → so √97 is between 9.84 and 9.85

Further refinement:
- 9.848² ≈ 96.9834
- 9.849² ≈ 97.0028 → so √97 is between 9.848 and 9.849

Since √97 ≈ 9.84886..., rounded to 3 decimal places = **9.849**
```

We can confirm the model used thinking by inspecting the response logs:

```bash
uv run llm logs --json -c | python3 -c "import sys,json; d=json.load(sys.stdin); types=[b['type'] for b in d[0]['response_json']['content']]; print('Content block types:', types)"
```

```output
Content block types: ['thinking', 'text']
```

## Effort parameter (GA, no beta header)

On 4.6 models, the effort parameter is generally available and works independently of thinking. This means you can control token spend without enabling extended thinking. The effort parameter accepts `low`, `medium`, `high`, or `max` (Opus 4.6 only).

```bash
uv run llm -m claude-sonnet-4.6 -o thinking_effort low "What is the capital of France?"
```

```output
The capital of France is **Paris**.
```

## Breaking change: prefill removal

Prefilling assistant messages (setting the start of the response) is not supported on 4.6 models. The plugin catches this early with a clear error rather than letting the API return a 400.

```bash
uv run llm -m claude-sonnet-4.6 -o prefill '{' 'Return JSON' 2>&1 || true
```

```output
Error: Prefilling assistant messages is not supported by claude-sonnet-4-6. Use structured outputs or system prompt instructions instead.
```

## Max effort is Opus 4.6 only

The new `max` effort level provides the absolute highest capability but is restricted to Opus 4.6. Attempting it on Sonnet 4.6 raises an error.

```bash
uv run llm -m claude-sonnet-4.6 -o thinking_effort max 'Hello' 2>&1 || true
```

```output
Error: thinking_effort='max' is only supported by claude-opus-4-6
```

## Opus 4.6

Opus 4.6 is the most capable model, with 128K max output tokens and support for `max` effort.

```bash
uv run llm -m claude-opus-4.6 "One fun fact about pelicans, one sentence"
```

```output
Pelicans have a pouch beneath their bill that can hold up to 3 gallons of water, which they use to scoop up fish before draining the water and swallowing their catch.
```

## Web search

Both 4.6 models support web search, letting Claude access real-time information.

```bash
uv run llm -m claude-sonnet-4.6 -o web_search 1 "What day of the week is it today?"
```

```output
Today is **Monday, February 17, 2026**.
```
