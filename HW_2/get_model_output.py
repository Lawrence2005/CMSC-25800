import json
import struct
import subprocess

import numpy as np
from PIL import Image


_MODEL_PROC = None


def start_model_proc(model_path="/local/homework/hw2_files/model_cli"):
    global _MODEL_PROC
    if _MODEL_PROC is None:
        _MODEL_PROC = subprocess.Popen(
            [model_path, "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    return _MODEL_PROC


def stop_model_proc():
    global _MODEL_PROC
    if _MODEL_PROC is not None:
        try:
            _MODEL_PROC.stdin.close()
        except Exception:
            pass
        try:
            _MODEL_PROC.terminate()
            _MODEL_PROC.wait(timeout=2)
        except Exception:
            pass
        _MODEL_PROC = None


def query_model(x_adv: Image.Image):
    if _MODEL_PROC is None:
        raise RuntimeError(
            "Model process not started. Call start_model_proc() first."
        )

    if not isinstance(x_adv, Image.Image):
        raise TypeError("x_adv must be a PIL.Image.Image")

    if x_adv.size != (32, 32):
        raise ValueError(
            f"Wrong image size: got {x_adv.size}, expected (32, 32)"
        )

    x_adv = x_adv.convert("RGB")
    x_adv = np.asarray(x_adv).astype(np.float32) / 255.0
    x_adv = np.transpose(x_adv, (2, 0, 1))[None, ...]

    if not np.isfinite(x_adv).all():
        raise ValueError("Input contains non-finite values")

    payload = x_adv.astype(np.float32, copy=False).tobytes()
    header = struct.pack("<I", len(payload))

    proc = _MODEL_PROC

    proc.stdin.write(header)
    proc.stdin.write(payload)
    proc.stdin.flush()

    line = proc.stdout.readline()
    if not line:
        err = proc.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"model process died. stderr:\n{err}")

    out = json.loads(line.decode("utf-8"))
    if "error" in out:
        raise RuntimeError(out["error"])

    pred = int(out["pred"])
    logits = np.array(out["logits"], dtype=np.float32)
    return pred, logits
