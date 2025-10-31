import os
from pathlib import Path

import pytest

from AgentSystem.utils import env_loader


@pytest.fixture(autouse=True)
def restore_environment():
    original_env = os.environ.copy()
    yield
    # Restore environment to avoid leaking state between tests
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_env_file(tmp_path: Path) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment line should be ignored",
                "BASIC=value",
                "INLINE=needs_comment # trailing comment should be stripped",
                "WITH_COLON: colon value",
                "QUOTED_DOUBLE=\"quoted # still value\"",
                "QUOTED_SINGLE='single # still value'",
                "WITH_ESCAPE=hash\\#should stay",
                "WITH_INTERPOLATION=${BASIC}_suffix",
                "DOUBLE_INTERPOLATION=\"${BASIC}_${INLINE}\"",
                "SINGLE_INTERPOLATION='${BASIC}_${INLINE}'",
                "ESCAPED_INTERPOLATION=\\${INLINE}",
                "ESCAPED_DOLLAR=\"Cost is \\$5\"",
                "export EXPORTED=from_export",
                "PRESERVED=from_file",
                "OVERWRITE=first",
                "OVERWRITE=second",
            ]
        )
    )
    return env_file


def test_manual_loader_parses_various_patterns(temp_env_file: Path) -> None:
    keys_to_check = {
        "BASIC": "value",
        "INLINE": "needs_comment",
        "WITH_COLON": "colon value",
        "QUOTED_DOUBLE": "quoted # still value",
        "QUOTED_SINGLE": "single # still value",
        "WITH_ESCAPE": "hash#should stay",
        "WITH_INTERPOLATION": "value_suffix",
        "DOUBLE_INTERPOLATION": "value_needs_comment",
        "SINGLE_INTERPOLATION": "${BASIC}_${INLINE}",
        "ESCAPED_INTERPOLATION": "${INLINE}",
        "ESCAPED_DOLLAR": "Cost is $5",
        "EXPORTED": "from_export",
        "OVERWRITE": "second",
    }

    os.environ["PRESERVED"] = "already set"

    original_load_dotenv = env_loader.load_dotenv
    env_loader.load_dotenv = None
    try:
        loader = env_loader.EnvLoader(env_file=str(temp_env_file))
    finally:
        env_loader.load_dotenv = original_load_dotenv

    for key, expected in keys_to_check.items():
        assert os.environ.get(key) == expected
        assert loader.get(key) == expected

    # Ensure pre-existing environment variables are not clobbered
    assert os.environ.get("PRESERVED") == "already set"

    # Accessing via loader should return preserved value
    assert loader.get("PRESERVED") == "already set"
