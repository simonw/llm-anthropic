set positional-arguments

test *args:
    uv run --isolated --with-editable '.[test]' pytest "$@"

llm *args:
    uv run --isolated --with-editable '.[test]' llm "$@"
