set positional-arguments

test *args:
    uv run pytest "$@"

llm *args:
    uv run llm "$@"
