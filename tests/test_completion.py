import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from infiniloader import cli
from infiniloader.completion import bash_completion


def complete(words):
    script = bash_completion(cli.build_parser())
    script += "\nCOMP_WORDS=(" + " ".join(shlex.quote(word) for word in words) + ")\n"
    script += f"COMP_CWORD={len(words) - 1}\n_iloader_complete\nprintf '%s\\n' \"${{COMPREPLY[@]}}\"\n"
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    return set(result.stdout.splitlines()) - {""}


@pytest.mark.skipif(os.name != "posix" or not shutil.which("bash"), reason="Bash completion integration")
@pytest.mark.parametrize("words,expected", [
    (["iloader", "sh"], {"shard"}),
    (["iloader", "_"], set()),
    (["iloader", "shard", ""], {"plan", "export", "verify"}),
    (["iloader", "shard", "plan", "--mode", "r"], {"replicated"}),
    (["iloader", "run", "unit", "--", "--"], set()),
    (["iloader", "run", "--cache-path", "/tmp/cache", "--"], set()),
    (["iloader", "run", "--cache-path=/tmp/cache", "--"], set()),
    (["iloader", "completion", "b"], {"bash"}),
])
def test_actual_bash_completion_follows_command_context(words, expected):
    assert complete(words) == expected


@pytest.mark.skipif(os.name != "posix" or not shutil.which("bash"), reason="Bash completion integration")
def test_nested_options_match_the_parser():
    options = complete(["iloader", "shard", "plan", "--"])
    assert {"--layout", "--output", "--mode", "--help"} <= options
    assert "--rank" not in options
    assert "--idle-ttl" in complete(["iloader", "start", "--"])


@pytest.mark.skipif(os.name != "posix" or not shutil.which("bash"), reason="Bash completion integration")
def test_path_values_take_precedence_and_preserve_spaces(tmp_path):
    directory = tmp_path / "cache directory"
    directory.mkdir()
    prefix = str(directory)[:-3]
    assert str(directory) in complete(["iloader", "run", "--cache-path", prefix])
    assert str(directory) in complete(["iloader", "status", "--socket", prefix])


def test_parser_handlers_and_remainder_contract_are_preserved():
    parser = cli.build_parser()
    args = parser.parse_args(["start", "/model", "--name", "unit"])
    assert args.func is cli.cmd_start
    args = parser.parse_args(["run", "--base-dir", "/state", "unit", "--", "engine", "--help"])
    assert args.func is cli.cmd_run
    assert args.name_arg == "unit" and args.program == ["engine", "--help"]
    args = parser.parse_args(["run", "--cache-path", "/cache", "engine", "--help"])
    assert args.name_arg == "engine" and args.program == ["--help"]
    args = parser.parse_args(["shard", "plan", "/model", "--layout", "layout.json", "--output", "plan.json"])
    assert args.source == Path("/model") and args.shard_action == "plan"
