"""Synthesize the covert-channel dataset, train the spectral classifier, export ONNX.

No public dataset exists for these attacks -- this generates labeled training
data purely from the published frequency/modulation specs (SilverPush,
MOSQUITO, DolphinAttack) plus a confusable appliance-whine negative, per the
proposal's data-plan section. Run this once before `run_echo_chamber.py`.

Usage
-----
    python train_classifier.py
    python train_classifier.py --n-per-class 800 --epochs 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from echo_chamber import CLASSES, N_BANDS  # noqa: E402
from echo_chamber.dataset import build_dataset, train_val_split  # noqa: E402
from echo_chamber.model_np import SpectralClassifier  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-per-class", type=int, default=800)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--out-dir", default="models")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1 -- Synthesize labeled ultrasonic dataset (from spec, no recordings)")
    print("=" * 60)
    print(f"  Classes       : {CLASSES}")
    print(f"  Per class     : {args.n_per_class}")
    X, y = build_dataset(n_per_class=args.n_per_class)
    X_train, y_train, X_val, y_val = train_val_split(X, y)
    print(f"  Train / val   : {len(y_train)} / {len(y_val)}   feature dim: {X.shape[1]}")
    print()

    print("=" * 60)
    print("STEP 2 -- Train spectral classifier (numpy MLP, Adam)")
    print("=" * 60)
    clf = SpectralClassifier(input_dim=N_BANDS)
    clf.fit(X_train, y_train, epochs=args.epochs, val=(X_val, y_val))
    val_acc = (clf.predict(X_val) == y_val).mean()
    print(f"\n  Final val accuracy: {val_acc:.3f}")
    if val_acc < 0.75:
        print("  [warn] Validation accuracy below 75% -- classes may need more separation.")
    else:
        print("  Note: this is single-frame (~21ms) accuracy on synthetic data.")
        print("  The live pipeline's consensus window aggregates multiple frames over")
        print("  500ms before flagging, which is materially more reliable than this")
        print("  per-frame number -- see ARCHITECTURE.md.")
    print()

    npz_path = out_dir / "echo_chamber_classifier.npz"
    onnx_path = out_dir / "echo_chamber_classifier.onnx"
    clf.save_npz(str(npz_path))

    print("=" * 60)
    print("STEP 3 -- Export ONNX (Gemm/Relu/Gemm/Relu/Gemm/Softmax)")
    print("=" * 60)
    clf.to_onnx(str(onnx_path))
    print(f"  Saved: {onnx_path}  ({onnx_path.stat().st_size} bytes)")
    print()

    print("=" * 60)
    print("STEP 4 -- Validate ONNX export against the numpy reference")
    print("=" * 60)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    sample = X_val[:8]
    # Model is exported with a static batch=1 (matches single-frame real-time
    # inference), so validate one frame at a time rather than as a true batch.
    onnx_probs = np.concatenate(
        [sess.run(None, {input_name: row[None, :]})[0] for row in sample], axis=0
    )
    np_probs = clf.predict_proba(sample)
    max_abs_diff = float(np.max(np.abs(onnx_probs - np_probs)))
    print(f"  Max |onnx - numpy| over {len(sample)} samples: {max_abs_diff:.6f}")
    assert max_abs_diff < 1e-4, "ONNX export diverges from the numpy reference"
    print("  ONNX export matches the trained model. OK")
    print()
    print(f"Model ready for mcp__quad__convert_model: {onnx_path.resolve()}")


if __name__ == "__main__":
    main()
