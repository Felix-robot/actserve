import json
import sys

from actserve.profile_cli import main, parse_gpu_sample


def test_parse_gpu_sample() -> None:
    sample = parse_gpu_sample("72, 1234, 81920, 211.5")
    assert sample.utilization_percent == 72
    assert sample.memory_used_mib == 1234
    assert sample.memory_total_mib == 81920
    assert sample.power_watts == 211.5


def test_profile_command_writes_privacy_safe_report(tmp_path) -> None:
    output = tmp_path / "profile.json"
    result = main(
        [
            "--output",
            str(output),
            "--",
            sys.executable,
            "-c",
            "print('public workload')",
        ]
    )
    assert result == 0
    report = json.loads(output.read_text())
    assert report["schema"] == "actserve.command_profile.v1"
    assert report["command"][0].startswith("python")
    assert report["command"][1] == "<arguments redacted>"
    assert report["profile"]["metrics"][0]["name"] in {
        "process.exit_code",
        "process.wall",
    }


def test_profile_command_propagates_exit_code(tmp_path) -> None:
    output = tmp_path / "failed.json"
    result = main(
        ["--output", str(output), "--", sys.executable, "-c", "raise SystemExit(7)"]
    )
    assert result == 7
    assert json.loads(output.read_text())["returncode"] == 7
