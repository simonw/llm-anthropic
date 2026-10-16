import llm
import pytest
from llm_anthropic import ClaudeOptions
from pydantic import ValidationError


@pytest.mark.parametrize(
    "sequence",
    ["END", "123", "0", "true", "false", "null", '"END"', '{"end": true}'],
)
@pytest.mark.parametrize("async_", [False, True])
def test_single_stop_sequence_is_preserved(sequence, async_):
    model = (llm.get_async_model if async_ else llm.get_model)("claude-haiku-4.5")
    prompt = llm.Prompt("Hi", model, options=model.Options(stop_sequences=sequence))
    assert model.build_kwargs(prompt, None)["stop_sequences"] == [sequence]


@pytest.mark.parametrize("sequences", [["END", "STOP"], '["END", "STOP"]', []])
def test_stop_sequence_lists(sequences):
    expected = ["END", "STOP"] if sequences else []
    assert ClaudeOptions(stop_sequences=sequences).stop_sequences == expected


@pytest.mark.parametrize("sequences", [[1], '["END", 1]', 123, {"end": True}])
def test_invalid_stop_sequence_lists_and_types(sequences):
    with pytest.raises(ValidationError):
        ClaudeOptions(stop_sequences=sequences)


def test_explicit_none_stop_sequences():
    model = llm.get_model("claude-haiku-4.5")
    prompt = llm.Prompt("Hi", model, options=model.Options(stop_sequences=None))
    assert "stop_sequences" not in model.build_kwargs(prompt, None)
