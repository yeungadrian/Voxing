import pytest

from voxing.tui.messages import STATUS_MARKUP, Status, status_markup


def test_each_status_produces_markup() -> None:
    for status in STATUS_MARKUP:
        result = status_markup(status)
        assert isinstance(result, str)
        assert len(result) > 0


def test_idle_contains_commands_hint() -> None:
    assert "commands" in status_markup(Status.IDLE)


def test_error_with_message() -> None:
    result = status_markup(Status.ERROR, "something broke")
    assert "something broke" in result
    assert "$error" in result


def test_error_without_message_raises() -> None:
    with pytest.raises(KeyError):
        status_markup(Status.ERROR, None)
