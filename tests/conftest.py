import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from echo_chamber import N_BANDS  # noqa: E402
from echo_chamber.dataset import build_dataset  # noqa: E402
from echo_chamber.model_np import SpectralClassifier  # noqa: E402


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory) -> str:
    """Train a small, fast classifier for tests (not the full train_classifier.py run)."""
    out_dir = tmp_path_factory.mktemp("echo_chamber_model")
    onnx_path = out_dir / "test_classifier.onnx"

    X, y = build_dataset(n_per_class=80, seed=1)
    clf = SpectralClassifier(input_dim=N_BANDS, seed=1)
    clf.fit(X, y, epochs=60, verbose=False)
    clf.to_onnx(str(onnx_path))
    return str(onnx_path)


@pytest.fixture()
def repo_model_path() -> str | None:
    """The real trained model from train_classifier.py, if it has been run."""
    path = APP_DIR / "models" / "echo_chamber_classifier.onnx"
    return str(path) if path.exists() else None
