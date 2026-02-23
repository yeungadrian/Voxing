# Voice Assitant Textual Terminal Application

## Guidelines
- Prefer self documenting code with fewer comments
- Never update / create READMEs following changes
- Use modern python 3.13 practices with strong type hints
- - Avoid using Any where possible
- Always use uv to run anything python related
- Code format / style check / static type check via `uv run pre-commit run -a`
- All private methods use single line docstrings
- Prefer simplicity and allow loud crashes
- Always use `uv add` to add dependencies instead of manually editing pyproject.toml
