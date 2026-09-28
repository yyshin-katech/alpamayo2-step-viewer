"""Smoke test of stream.py on the real snapshot: coverage, bitwise equality, throughput."""
import time
import torch
from safetensors import safe_open

from common import SNAP, MODEL_ID
from build import build_meta_model, move_buffers_to_cuda
from stream import Streamer, LayerPrefetcher

torch.cuda.init()
t0 = time.perf_counter()
model, missing = build_meta_model(SNAP, name_or_path=MODEL_ID)
move_buffers_to_cuda(model)
s = Streamer(model, SNAP)
print(f"build+index {time.perf_counter()-t0:.1f}s")
print("resident", {k: round(u.nbytes / 1e9, 3) for k, u in s.resident.items()})
print("vision", round(s.vision.nbytes / 1e9, 3), "ranges", len(s.vision.ranges))
print("layer0", round(s.layers[0].nbytes / 1e9, 4), "ranges", len(s.layers[0].ranges),
      "max layer", round(max(u.nbytes for u in s.layers) / 1e9, 4))
print("expert units", len(s.expert_units), "total", round(sum(u.nbytes for u in s.expert_units) / 1e9, 3),
      "misc", round(s.expert_units[-1].nbytes / 1e9, 3))

handles = {}
def ref(key):
    loc = s.index.locs[key]
    if loc.path not in handles:
        handles[loc.path] = safe_open(loc.path, "pt", device="cpu")
    return handles[loc.path].get_tensor(key)

def check(unit, sample=None):
    bad = 0
    ents = unit.entries if sample is None else unit.entries[:sample]
    for e in ents:
        cur = e.module._parameters[e.attr] if e.is_param else e.module._buffers[e.attr]
        r = ref(e.key)
        assert cur.device.type == "cuda" and cur.dtype == r.dtype and tuple(cur.shape) == tuple(r.shape), e.key
        if not torch.equal(cur.cpu().view(torch.uint8) if cur.dtype != torch.bool else cur.cpu(),
                           r.view(torch.uint8) if r.dtype != torch.bool else r):
            bad += 1
            print("MISMATCH", e.key)
    return bad, len(ents)

t0 = time.perf_counter(); s.load_resident(); torch.cuda.synchronize()
print(f"resident load {time.perf_counter()-t0:.1f}s alloc={torch.cuda.memory_allocated()/2**30:.2f}GiB")
for k, u in s.resident.items():
    print(" check", k, check(u))

t0 = time.perf_counter(); s.vision.load_sync(s.reader)
print(f"vision load {time.perf_counter()-t0:.1f}s")
print(" check vision", check(s.vision)); s.vision.uninstall()

pf = LayerPrefetcher(s.layers, s.reader, n_slots=2, cache_ids=(0, 2))
for li in range(4):
    t0 = time.perf_counter(); pf.install(li)
    dt = time.perf_counter() - t0
    b = check(s.layers[li])
    s.layers[li].uninstall()
    print(f" layer{li} install {dt:.2f}s (wait {pf.wait_s:.2f} h2d {pf.h2d_s:.2f}) check {b}")
rd = s.reader.bytes_read / s.reader.seconds / 1e9
pf.stop()
print(f"reader avg {rd:.2f} GB/s over {s.reader.bytes_read/1e9:.1f} GB")

t0 = time.perf_counter()
for u in s.expert_units[:2] + s.expert_units[-1:]:
    u.load_sync(s.reader)
    print(" check", u.name, check(u))
    u.uninstall()
print(f"expert sample load {time.perf_counter()-t0:.1f}s")
torch.cuda.empty_cache()
print("alloc after", round(torch.cuda.memory_allocated() / 2**30, 3), "GiB; max", round(torch.cuda.max_memory_allocated() / 2**30, 2))
s.close()
print("OK")
