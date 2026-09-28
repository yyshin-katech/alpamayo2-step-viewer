"""Build the notebook's model inputs for sample 0 (CPU only) and dump the token layout."""
from __future__ import annotations
import torch
from alpamayo2_super import helper
from build import build_meta_model
from common import MODEL_ID, OUT, SNAP, write_json

def main():
    model, _ = build_meta_model(SNAP, name_or_path=MODEL_ID)
    data = torch.load(OUT / "data" / "sample0.pt", weights_only=False)
    inputs = helper.prepare_model_inputs(data, model.config, model.tokenizer)
    tok = model.tokenizer
    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape), v.dtype)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                print(" ", k, kk, tuple(vv.shape) if torch.is_tensor(vv) else type(vv))
        else:
            print(k, type(v))
    tk = inputs["tokenized_data"] if "tokenized_data" in inputs else inputs
    ids = tk["input_ids"][0]
    print("L =", ids.numel())
    # run-length encode
    runs = []
    prev = None
    for i, t in enumerate(ids.tolist()):
        if prev is not None and t == prev and t in (151655, 155684):
            runs[-1][2] += 1
        else:
            runs.append([i, t, 1])
        prev = t
    lines = []
    for s, t, n in runs:
        s_txt = tok.convert_ids_to_tokens(t)
        lines.append(f"{s:5d} {t:7d} x{n:<4d} {s_txt!r}")
    (OUT / "data" / "prompt_layout.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:80]))
    print("...")
    print("\n".join(lines[-40:]))

if __name__ == "__main__":
    main()
