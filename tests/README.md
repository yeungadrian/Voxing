# Tests

## Running tests

```sh
uv run pytest
```

## Snapshots

Tests use [pytest-textual-snapshot](https://github.com/Textualize/pytest-textual-snapshot) for snapshot testing.

Update snapshots:

```sh
uv run pytest --snapshot-update
```
