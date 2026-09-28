"""Layer-streaming weight loader for Alpamayo2-Super on a ~10 GB GPU.

The 71.6 GB checkpoint cannot be resident, so tensors are read straight from the
safetensors shards with O_DIRECT into page-locked (cudaHostRegister) host buffers,
copied into fresh CUDA tensors and swapped into a meta-initialised model right
before the owning module runs, then swapped back to meta afterwards.

Values are bit-identical to ``from_pretrained``: the bytes are the checkpoint
bytes, and the destination dtype is the dtype HF would cast to (the meta
parameter's dtype).
"""

from __future__ import annotations

import json
import mmap
import os
import queue
import struct
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
from torch import nn

_DT = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
}
ALIGN = 4096          # O_DIRECT offset/length/buffer alignment
GAP = 1 << 20         # coalesce tensors separated by at most this many bytes
CHUNK = 64 << 20      # per-request read size


@dataclass(frozen=True)
class TensorLoc:
    path: str
    start: int        # absolute byte offset in the shard file
    nbytes: int
    dtype: torch.dtype
    shape: tuple


class CkptIndex:
    """Key -> location map built from the safetensors headers (no data read)."""

    def __init__(self, snap):
        snap = str(snap)
        idx_path = os.path.join(snap, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as fh:
                files = sorted(set(json.load(fh)["weight_map"].values()))
        else:
            files = ["model.safetensors"]
        self.locs: dict[str, TensorLoc] = {}
        self.file_sizes: dict[str, int] = {}
        for f in files:
            p = os.path.realpath(os.path.join(snap, f))
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            base = 8 + n
            self.file_sizes[p] = os.path.getsize(p)
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                s, e = v["data_offsets"]
                self.locs[k] = TensorLoc(p, base + s, e - s, _DT[v["dtype"]], tuple(v["shape"]))


class PinnedBuffer:
    """Anonymous mmap registered as page-locked memory (fast async H2D, no torch host cache)."""

    def __init__(self, nbytes: int):
        self.nbytes = nbytes
        self._mm = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        self.mv = memoryview(self._mm)
        self.t = torch.frombuffer(self._mm, dtype=torch.uint8)
        err = torch.cuda.cudart().cudaHostRegister(self.t.data_ptr(), nbytes, 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister failed: {err}")

    def close(self) -> None:
        if self._mm is None:
            return
        torch.cuda.cudart().cudaHostUnregister(self.t.data_ptr())
        del self.t
        self.mv.release()
        self._mm.close()
        self._mm = None


def _pread_exact(fd: int, fsz: int, mv: memoryview, boff: int, off: int, n: int) -> None:
    """Read n bytes (aligned request) at file offset off into mv[boff:]; EOF short reads are ok."""
    need = min(n, fsz - off)
    got = 0
    while got < need:
        r = os.preadv(fd, [mv[boff + got: boff + n]], off + got)
        if r <= 0:
            break
        got += r
        if got % ALIGN:          # short read at EOF; a further O_DIRECT read would be unaligned
            break
    if got < need:
        raise IOError(f"short read fd={fd} off={off} n={n} got={got} need={need}")


class Reader:
    """O_DIRECT reader with a small thread pool (NVMe peaks at ~2 outstanding 64 MB requests)."""

    def __init__(self, file_sizes: dict[str, int], nthreads: int = 2):
        self._pool = ThreadPoolExecutor(nthreads, thread_name_prefix="odirect")
        self._fds: dict[str, int] = {}
        self._sizes = file_sizes
        self._lock = threading.Lock()
        self.bytes_read = 0
        self.seconds = 0.0

    def _fd(self, path: str) -> int:
        with self._lock:
            if path not in self._fds:
                self._fds[path] = os.open(path, os.O_RDONLY | os.O_DIRECT)
            return self._fds[path]

    def read(self, unit: "Unit", buf: PinnedBuffer) -> None:
        t0 = time.perf_counter()
        futs = []
        for path, off, n, boff in unit.ranges:
            fd = self._fd(path)
            fsz = self._sizes[path]
            for c in range(0, n, CHUNK):
                futs.append(self._pool.submit(
                    _pread_exact, fd, fsz, buf.mv, boff + c, off + c, min(CHUNK, n - c)))
        for f in futs:
            f.result()
        with self._lock:
            self.bytes_read += unit.nbytes
            self.seconds += time.perf_counter() - t0

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


@dataclass
class Entry:
    key: str
    module: nn.Module
    attr: str
    is_param: bool
    loc: TensorLoc


class Unit:
    """A group of checkpoint tensors that are read and installed together."""

    def __init__(self, name: str, keys, model: nn.Module, index: CkptIndex):
        self.name = name
        self.entries: list[Entry] = []
        for k in keys:
            mod_name, _, attr = k.rpartition(".")
            mod = model.get_submodule(mod_name)
            if attr in mod._parameters:
                is_param = True
            elif attr in mod._buffers:
                is_param = False
            else:
                raise KeyError(k)
            self.entries.append(Entry(k, mod, attr, is_param, index.locs[k]))
        self._layout()
        self._saved = None

    @property
    def installed(self) -> bool:
        return self._saved is not None

    def _layout(self) -> None:
        by_file = defaultdict(list)
        for e in self.entries:
            by_file[e.loc.path].append(e)
        self.ranges: list[tuple[str, int, int, int]] = []   # (path, file_off, nbytes, buf_off)
        self.offsets: dict[str, int] = {}
        boff = 0
        for path in sorted(by_file):
            groups = []
            for e in sorted(by_file[path], key=lambda e: e.loc.start):
                end = e.loc.start + e.loc.nbytes
                if groups and e.loc.start - groups[-1][1] <= GAP:
                    groups[-1][1] = max(groups[-1][1], end)
                    groups[-1][2].append(e)
                else:
                    groups.append([e.loc.start, end, [e]])
            for s, t, es in groups:
                a = s // ALIGN * ALIGN
                b = -(-t // ALIGN) * ALIGN
                self.ranges.append((path, a, b - a, boff))
                for e in es:
                    self.offsets[e.key] = boff + e.loc.start - a
                boff += b - a
        self.nbytes = boff

    def install(self, buf: PinnedBuffer) -> None:
        """Copy tensors from buf into new CUDA tensors and swap them into the modules."""
        assert not self.installed, self.name
        saved = []
        with torch.inference_mode(False):
            for e in self.entries:
                o = self.offsets[e.key]
                raw = buf.t[o: o + e.loc.nbytes]
                if o % torch.empty((), dtype=e.loc.dtype).element_size():
                    raw = raw.clone()
                src = raw.view(e.loc.dtype).view(e.loc.shape)
                old = e.module._parameters[e.attr] if e.is_param else e.module._buffers[e.attr]
                dst = torch.empty(e.loc.shape, dtype=old.dtype, device="cuda")
                dst.copy_(src, non_blocking=True)
                if e.is_param:
                    e.module._parameters[e.attr] = nn.Parameter(dst, requires_grad=old.requires_grad)
                else:
                    e.module._buffers[e.attr] = dst
                saved.append(old)
        # The staging buffer is only reusable once the async copies have landed.
        torch.cuda.current_stream().synchronize()
        self._saved = saved

    def uninstall(self) -> None:
        assert self.installed, self.name
        for e, old in zip(self.entries, self._saved):
            if e.is_param:
                e.module._parameters[e.attr] = old
            else:
                e.module._buffers[e.attr] = old
        self._saved = None

    def load_sync(self, reader: Reader) -> None:
        buf = PinnedBuffer(self.nbytes)
        try:
            reader.read(self, buf)
            self.install(buf)
        finally:
            buf.close()


class LayerPrefetcher:
    """Background thread that streams layer units 0..N-1, 0..N-1, ... into staging slots.

    A fixed set of layers is kept in a pinned RAM cache after the first read, so
    every decode step only re-reads the remaining layers from disk.
    """

    def __init__(self, units: list[Unit], reader: Reader, n_slots: int = 2, cache_ids=()):
        self.units = units
        self.reader = reader
        self.cache_ids = set(cache_ids)
        slot_bytes = max(u.nbytes for i, u in enumerate(units) if i not in self.cache_ids)
        self.slots = [PinnedBuffer(slot_bytes) for _ in range(n_slots)]
        self.cache: dict[int, PinnedBuffer] = {}
        self._free: queue.Queue = queue.Queue()
        for i in range(n_slots):
            self._free.put(i)
        self._ready: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._err: BaseException | None = None
        self.wait_s = 0.0
        self.h2d_s = 0.0
        self._thread = threading.Thread(target=self._run, name="prefetch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            li = 0
            while not self._stop.is_set():
                u = self.units[li]
                if li in self.cache_ids:
                    if li not in self.cache:
                        b = PinnedBuffer(u.nbytes)
                        self.reader.read(u, b)
                        self.cache[li] = b
                    self._ready.put((li, self.cache[li], None))
                else:
                    slot = None
                    while slot is None and not self._stop.is_set():
                        try:
                            slot = self._free.get(timeout=0.2)
                        except queue.Empty:
                            pass
                    if slot is None:
                        break
                    self.reader.read(u, self.slots[slot])
                    self._ready.put((li, self.slots[slot], slot))
                li = (li + 1) % len(self.units)
        except BaseException as ex:  # propagate to the consumer
            self._err = ex
            self._ready.put(None)

    def install(self, li: int) -> None:
        """Block until layer li is staged, install it on the GPU and recycle its slot."""
        t0 = time.perf_counter()
        while True:
            try:
                item = self._ready.get(timeout=1.0)
                break
            except queue.Empty:
                if self._err is not None or not self._thread.is_alive():
                    raise RuntimeError("prefetch thread died") from self._err
        if item is None:
            raise RuntimeError("prefetch thread failed") from self._err
        got, buf, slot = item
        if got != li:
            raise RuntimeError(f"layer order mismatch: staged {got}, requested {li}")
        t1 = time.perf_counter()
        self.units[li].install(buf)
        if slot is not None:
            self._free.put(slot)
        self.wait_s += t1 - t0
        self.h2d_s += time.perf_counter() - t1

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        while not self._ready.empty():
            self._ready.get_nowait()
        for b in self.slots:
            b.close()
        for b in self.cache.values():
            b.close()
        self.slots, self.cache = [], {}


class Streamer:
    """Wires units and hooks onto a meta-built Alpamayo2Super."""

    def __init__(self, model: nn.Module, snap, cache_ids=(0, 5, 11, 16, 21, 27, 32, 37, 43, 48, 53, 59),
                 n_slots: int = 2, log=print):
        self.model = model
        self.log = log
        self.index = CkptIndex(snap)
        self.reader = Reader(self.index.file_sizes)
        keys = sorted(self.index.locs)
        vlm = model.vlm
        lm = vlm.model.language_model
        n_layers = len(lm.layers)

        def pick(prefix):
            return [k for k in keys if k.startswith(prefix)]

        self.resident = {
            "patch_embed": Unit("patch_embed", pick("vlm.model.visual.patch_embed."), model, self.index),
            "embed_tokens": Unit("embed_tokens", pick("vlm.model.language_model.embed_tokens."), model, self.index),
            "lm_head": Unit("lm_head", pick("vlm.lm_head."), model, self.index),
            "norm": Unit("norm", pick("vlm.model.language_model.norm."), model, self.index),
        }
        vis_keys = [k for k in pick("vlm.model.visual.") if not k.startswith("vlm.model.visual.patch_embed.")]
        self.vision = Unit("vision", vis_keys, model, self.index)
        self.layers = [Unit(f"llm{i}", pick(f"vlm.model.language_model.layers.{i}."), model, self.index)
                       for i in range(n_layers)]
        exp_keys = pick("expert.")
        exp_layers = len(model.expert.expert.layers)
        self.expert_units = [Unit(f"exp{i}", [k for k in exp_keys if k.startswith(f"expert.expert.layers.{i}.")],
                                  model, self.index) for i in range(exp_layers)]
        rest = [k for k in exp_keys if not k.startswith("expert.expert.layers.")]
        self.expert_units.append(Unit("expert_misc", rest, model, self.index))

        covered = set()
        for u in [*self.resident.values(), self.vision, *self.layers, *self.expert_units]:
            for e in u.entries:
                assert e.key not in covered, e.key
                covered.add(e.key)
        missing = set(keys) - covered
        if missing:
            raise RuntimeError(f"checkpoint keys not assigned to any unit: {sorted(missing)[:10]}")

        self.cache_ids = tuple(i for i in cache_ids if i < n_layers)
        self.n_slots = n_slots
        self.prefetcher: LayerPrefetcher | None = None
        self.expert_loaded = False
        self.handles = []
        self.timeline: list[tuple[str, float]] = []

    # -- setup -------------------------------------------------------------
    def load_resident(self) -> None:
        for u in self.resident.values():
            u.load_sync(self.reader)

    def attach(self) -> None:
        m = self.model
        lm = m.vlm.model.language_model
        self.prefetcher = LayerPrefetcher(self.layers, self.reader, self.n_slots, self.cache_ids)

        def vis_pre(mod, args, kwargs):
            t0 = time.perf_counter()
            self.vision.load_sync(self.reader)
            self.timeline.append(("vision_load", time.perf_counter() - t0))

        def vis_post(mod, args, kwargs, out):
            self.vision.uninstall()

        self.handles.append(m.vlm.model.visual.register_forward_pre_hook(vis_pre, with_kwargs=True, prepend=True))
        self.handles.append(m.vlm.model.visual.register_forward_hook(vis_post, with_kwargs=True))

        for i, layer in enumerate(lm.layers):
            def pre(mod, args, kwargs, i=i):
                self.prefetcher.install(i)

            self.handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True, prepend=True))

        def exp_pre(mod, args, kwargs):
            if not self.expert_loaded:
                self._enter_expert_phase()

        self.handles.append(m.expert.action_in_proj.register_forward_pre_hook(exp_pre, with_kwargs=True, prepend=True))

    def attach_uninstall_hooks(self) -> None:
        """Registered after the capture hooks so captures still see the installed weights."""
        lm = self.model.vlm.model.language_model
        for i, layer in enumerate(lm.layers):
            def post(mod, args, kwargs, out, i=i):
                self.layers[i].uninstall()

            self.handles.append(layer.register_forward_hook(post, with_kwargs=True))

    def _enter_expert_phase(self) -> None:
        t0 = time.perf_counter()
        self.prefetcher.stop()
        for name in ("embed_tokens", "lm_head"):
            self.resident[name].uninstall()
        torch.cuda.empty_cache()
        for u in self.expert_units:
            u.load_sync(self.reader)
        self.expert_loaded = True
        self.timeline.append(("expert_load", time.perf_counter() - t0))
        self.log(f"[stream] expert phase: loaded {sum(u.nbytes for u in self.expert_units)/1e9:.2f} GB "
                 f"in {time.perf_counter()-t0:.1f}s; alloc={torch.cuda.memory_allocated()/2**30:.2f} GiB")

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        if self.prefetcher is not None and self.prefetcher.slots:
            self.prefetcher.stop()
        self.reader.close()
