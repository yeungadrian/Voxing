# Voice Assitant Textual Terminal Application

## Guidelines
- Prefer self documenting code with fewer comments
- Never update / create READMEs following changes
- Use modern python 3.13 practices
- Always use uv to run anything python related
- code style check via `uvx ruff check --fix`
- Type hint everything and Static type check via `uvx ty check`
- Avoid using Any where possible
- All private methods use single line docstrings
- Prefer simplicity and allow loud crashes
- Always use `uv add` to add dependencies instead of manually editing pyproject.toml
