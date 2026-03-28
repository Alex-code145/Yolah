import os
import numpy as np
import torch

try:
    import openvino.runtime as ov
except Exception:
    ov = None

from .nn_model import state_to_tensor
from Jeu.Yolah import Move


def export_model_to_onnx(model: torch.nn.Module, onnx_path: str):
    """Export the PyTorch model to ONNX if the file does not exist.

    The exported model has two outputs: policy (B,4096) and value (B,1).
    """
    if os.path.exists(onnx_path):
        return onnx_path

    model.eval()
    dummy = torch.randn(1, 3, 8, 8)
    try:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            opset_version=14,
            input_names=["input"],
            output_names=["policy", "value"],
            dynamic_axes={"input": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        )
    except Exception as e:
        raise RuntimeError(f"Failed to export model to ONNX: {e}")
    return onnx_path


class OpenVINOEvaluator:
    def __init__(self, onnx_path: str, device: str = "CPU"):
        if ov is None:
            raise RuntimeError("OpenVINO runtime not available")
        core = ov.Core()
        model = core.read_model(onnx_path)
        # compile model for device
        self.compiled = core.compile_model(model, device_name=device)
        # get input and output info
        self.input_name = self.compiled.input(0).get_any_name()
        # outputs
        self.policy_name = self.compiled.output(0).get_any_name()
        self.value_name = self.compiled.output(1).get_any_name()

    def __call__(self, state):
        # state: YolahState
        black, white, empty, black_score, white_score, ply = state.game.get_state()
        current_player = state.current_player()
        t = state_to_tensor(black, white, empty, current_player)
        arr = t.unsqueeze(0).numpy().astype(np.float32)
        # run inference
        # compiled_model can be called with input array directly; returns a list/tuple of outputs
        res = self.compiled([arr])
        # res may be a single object or mapping; handle common cases
        try:
            policy_np = np.asarray(res[self.policy_name]) if isinstance(res, dict) else np.asarray(res[0])
            value_np = np.asarray(res[self.value_name]) if isinstance(res, dict) else np.asarray(res[1])
        except Exception:
            # fallback: if res is a list/tuple
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                policy_np = np.asarray(res[0])
                value_np = np.asarray(res[1])
            else:
                raise RuntimeError("Unexpected OpenVINO inference result format")

        probs = np.exp(policy_np)  # logits -> exp, will normalize below
        probs = probs.reshape(-1)
        val = float(value_np.reshape(-1)[0])

        # map probs to legal moves
        legal = state.legal_moves()
        move_probs = {}
        total = 0.0
        for m in legal:
            idx = m.from_sq.sq * 64 + m.to_sq.sq
            p = float(probs[idx])
            move_probs[str(m)] = p
            total += p

        if total > 0:
            for k in move_probs:
                move_probs[k] /= total
        else:
            unif = 1.0 / len(legal)
            for m in legal:
                move_probs[str(m)] = unif

        return move_probs, val

    def batch_eval(self, states):
        """Batch evaluate a list of states using the compiled OpenVINO model.

        Returns a list of (move_probs, value) tuples.
        """
        # prepare batch array
        arrs = []
        for state in states:
            black, white, empty, black_score, white_score, ply = state.game.get_state()
            current_player = state.current_player()
            t = state_to_tensor(black, white, empty, current_player)
            arrs.append(t.numpy().astype(np.float32))
        batch = np.stack(arrs, axis=0)

        res = self.compiled([batch])
        try:
            policy_np = np.asarray(res[self.policy_name]) if isinstance(res, dict) else np.asarray(res[0])
            value_np = np.asarray(res[self.value_name]) if isinstance(res, dict) else np.asarray(res[1])
        except Exception:
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                policy_np = np.asarray(res[0])
                value_np = np.asarray(res[1])
            else:
                raise RuntimeError("Unexpected OpenVINO inference result format")

        policy_np = policy_np.reshape((policy_np.shape[0], -1))
        value_np = value_np.reshape(-1)

        results = []
        for i, state in enumerate(states):
            probs = np.exp(policy_np[i])
            legal = state.legal_moves()
            move_probs = {}
            total = 0.0
            for m in legal:
                idx = m.from_sq.sq * 64 + m.to_sq.sq
                p = float(probs[idx])
                move_probs[str(m)] = p
                total += p

            if total > 0:
                for k in move_probs:
                    move_probs[k] /= total
            else:
                unif = 1.0 / len(legal)
                for m in legal:
                    move_probs[str(m)] = unif

            results.append((move_probs, float(value_np[i])))

        return results
