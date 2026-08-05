"""Tiny numpy MLP spectral classifier + hand-authored ONNX export.

Kept dependency-free (numpy only) for training so the sample doesn't need a
full torch/sklearn stack just to produce a ~5K-parameter classifier; the
ONNX export uses the lightweight `onnx` package (protobuf authoring only,
no runtime) so the resulting graph is a completely standard
Gemm/Relu/Gemm/Relu/Gemm/Softmax chain that qairt-converter and
onnxruntime both understand natively.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

INPUT_NAME = "spectral_features"
OUTPUT_NAME = "class_probs"


@dataclass
class SpectralClassifier:
    input_dim: int
    hidden1: int = 32
    hidden2: int = 16
    n_classes: int = 5
    seed: int = 7
    weights: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.weights:
            return
        rng = np.random.default_rng(self.seed)

        def he(fan_in: int, fan_out: int) -> np.ndarray:
            return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)).astype(np.float32)

        self.weights = {
            "W1": he(self.input_dim, self.hidden1), "b1": np.zeros(self.hidden1, dtype=np.float32),
            "W2": he(self.hidden1, self.hidden2), "b2": np.zeros(self.hidden2, dtype=np.float32),
            "W3": he(self.hidden2, self.n_classes), "b3": np.zeros(self.n_classes, dtype=np.float32),
        }

    # -- forward pass -----------------------------------------------------

    def _forward(self, X: np.ndarray, cache: dict | None = None) -> np.ndarray:
        W = self.weights
        z1 = X @ W["W1"] + W["b1"]
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ W["W2"] + W["b2"]
        a2 = np.maximum(z2, 0.0)
        logits = a2 @ W["W3"] + W["b3"]
        if cache is not None:
            cache.update(X=X, z1=z1, a1=a1, z2=z2, a2=a2, logits=logits)
        return logits

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(self._forward(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # -- training (manual backprop + Adam) --------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 300,
        batch_size: int = 64,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        val: tuple[np.ndarray, np.ndarray] | None = None,
        verbose: bool = True,
    ) -> list[float]:
        n = len(y)
        y_onehot = np.eye(self.n_classes, dtype=np.float32)[y]
        rng = np.random.default_rng(123)

        m = {k: np.zeros_like(v) for k, v in self.weights.items()}
        v = {k: np.zeros_like(w) for k, w in self.weights.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        history: list[float] = []
        best_val_acc = -1.0
        best_weights = {k: w.copy() for k, w in self.weights.items()}

        for epoch in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                Xb, Yb = X[idx], y_onehot[idx]
                Xb = Xb + rng.normal(0.0, 0.05, size=Xb.shape).astype(np.float32)  # feature-noise regularizer
                cache: dict = {}
                logits = self._forward(Xb, cache)
                probs = self._softmax(logits)
                bsz = len(idx)

                d_logits = (probs - Yb) / bsz
                W = self.weights
                grads = {
                    "W3": cache["a2"].T @ d_logits,
                    "b3": d_logits.sum(axis=0),
                }
                d_a2 = d_logits @ W["W3"].T
                d_z2 = d_a2 * (cache["z2"] > 0)
                grads["W2"] = cache["a1"].T @ d_z2
                grads["b2"] = d_z2.sum(axis=0)
                d_a1 = d_z2 @ W["W2"].T
                d_z1 = d_a1 * (cache["z1"] > 0)
                grads["W1"] = cache["X"].T @ d_z1
                grads["b1"] = d_z1.sum(axis=0)

                t += 1
                for k in self.weights:
                    m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                    v[k] = beta2 * v[k] + (1 - beta2) * (grads[k] ** 2)
                    m_hat = m[k] / (1 - beta1 ** t)
                    v_hat = v[k] / (1 - beta2 ** t)
                    epoch_lr = lr * (0.5 ** (epoch // 100))  # step decay
                    self.weights[k] -= epoch_lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * self.weights[k])

            logits = self._forward(X)
            probs = self._softmax(logits)
            loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-9))
            history.append(float(loss))

            if val is not None:
                Xv, yv = val
                val_acc = (self.predict(Xv) == yv).mean()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = {k: w.copy() for k, w in self.weights.items()}

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                acc = (self.predict(X) == y).mean()
                msg = f"  epoch {epoch:>4}  loss={loss:.4f}  train_acc={acc:.3f}"
                if val is not None:
                    msg += f"  val_acc={val_acc:.3f}  best_val_acc={best_val_acc:.3f}"
                print(msg)

        if val is not None and best_val_acc >= 0:
            self.weights = best_weights
            if verbose:
                print(f"  Restored best-val checkpoint (val_acc={best_val_acc:.3f})")
        return history

    # -- ONNX export --------------------------------------------------------

    def to_onnx(self, output_path: str) -> None:
        """Hand-author a Gemm/Relu/Gemm/Relu/Gemm/Softmax ONNX graph from `self.weights`."""
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        W = self.weights

        def init(name: str, arr: np.ndarray) -> onnx.TensorProto:
            # Gemm expects B as (out, in) when transB=1 -- store weights transposed.
            return numpy_helper.from_array(arr.astype(np.float32), name=name)

        initializers = [
            init("W1", W["W1"].T), init("b1", W["b1"]),
            init("W2", W["W2"].T), init("b2", W["b2"]),
            init("W3", W["W3"].T), init("b3", W["b3"]),
        ]

        nodes = [
            helper.make_node("Gemm", ["input", "W1", "b1"], ["z1"], transB=1),
            helper.make_node("Relu", ["z1"], ["a1"]),
            helper.make_node("Gemm", ["a1", "W2", "b2"], ["z2"], transB=1),
            helper.make_node("Relu", ["z2"], ["a2"]),
            helper.make_node("Gemm", ["a2", "W3", "b3"], ["logits"], transB=1),
            helper.make_node("Softmax", ["logits"], [OUTPUT_NAME], axis=1),
        ]

        # Static batch=1: the deployed NPU context binary always processes one
        # analysis frame at a time in the real-time streaming pipeline, and a
        # static shape is what qairt-converter needs to compile a context binary.
        graph = helper.make_graph(
            nodes,
            "echo_chamber_spectral_classifier",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, self.input_dim])],
            [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [1, self.n_classes])],
            initializer=initializers,
        )
        # "input" is the graph-level tensor name; rename to the documented
        # feature-vector name for downstream clarity while keeping node wiring above.
        graph.input[0].name = INPUT_NAME
        for node in graph.node:
            node.input[:] = [INPUT_NAME if i == "input" else i for i in node.input]

        model = helper.make_model(
            graph,
            producer_name="echo_chamber",
            opset_imports=[helper.make_opsetid("", 17)],
        )
        model.ir_version = 8
        onnx.checker.check_model(model)
        onnx.save(model, output_path)

    def save_npz(self, path: str) -> None:
        np.savez(path, **self.weights, input_dim=self.input_dim,
                 hidden1=self.hidden1, hidden2=self.hidden2, n_classes=self.n_classes)

    @classmethod
    def load_npz(cls, path: str) -> "SpectralClassifier":
        data = np.load(path)
        weights = {k: data[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
        return cls(
            input_dim=int(data["input_dim"]), hidden1=int(data["hidden1"]),
            hidden2=int(data["hidden2"]), n_classes=int(data["n_classes"]),
            weights=weights,
        )
