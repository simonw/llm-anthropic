# Default task (run with just 'just')
test *args:
    uv run --isolated --with-editable '.[test]' pytest {{args}}

llm *args:
    uv run --isolated --with-editable '.[test]' llm {{args}}
