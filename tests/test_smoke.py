import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]


def test_imports():
    import echo_chamber  # noqa: F401
    import echo_chamber.alert  # noqa: F401
    import echo_chamber.capture  # noqa: F401
    import echo_chamber.consensus  # noqa: F401
    import echo_chamber.dataset  # noqa: F401
    import echo_chamber.dsp  # noqa: F401
    import echo_chamber.fusion_server  # noqa: F401
    import echo_chamber.inference  # noqa: F401
    import echo_chamber.model_np  # noqa: F401
    import echo_chamber.signature_library  # noqa: F401
    import echo_chamber.signature_reference  # noqa: F401
    import echo_chamber.synth  # noqa: F401


def test_cli_help():
    result = subprocess.run(
        [sys.executable, str(APP_DIR / "run_echo_chamber.py"), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "Echo Chamber" in result.stdout


def test_train_classifier_help():
    result = subprocess.run(
        [sys.executable, str(APP_DIR / "train_classifier.py"), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
