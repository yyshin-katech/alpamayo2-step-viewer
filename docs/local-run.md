# Local run guide

This guide takes you from getting the code to making a capture on your own computer and viewing it in the viewer. How to download the model and the dataset, and their licenses, are covered separately in [model-and-data.md](model-and-data.md).

Run every command from the repository folder (`alpamayo2-step-viewer/`).

## 1. Requirements

| Item | Requirement | Notes |
|---|---|---|
| OS | Linux x86_64 | Tested on Ubuntu 22.04. The dependency list is for Linux x86_64. It does not run on macOS or Windows (CUDA, O_DIRECT). WSL2 has not been tested. |
| GPU | NVIDIA, at least 10 GB of memory, bf16 support | Ampere (RTX 30xx) or newer is recommended. Tested on an RTX 3080 10 GB. |
| Driver | An NVIDIA driver that supports CUDA 12.8 | R570 or newer is recommended. The PyTorch wheels include the CUDA runtime, so the CUDA toolkit (nvcc) is not needed. |
| RAM | 32 GB recommended | Part of the weights is kept in page-locked memory, about 14 GB with the default settings (estimate). If you run short, see [§8](#8-troubleshooting). |
| Storage | About 120 GB free | Model 72 GB, at least 30 GB free under `walk/out` (the capture itself is about 9 GB), `.venv` about 7 GB, uv cache about 4 GB |
| File system | The model must be on a local disk that supports O_DIRECT (ext4, xfs, etc.) | An NVMe SSD is strongly recommended. The run time is set almost entirely by the speed of the storage. |
| Network | Downloads: wheels about 4 GB, model 72 GB, sample a few hundred MB | Capture, analysis and the viewer run offline. The viewer loads no external resources. |
| Browser | A recent Chrome, Edge, Firefox, etc. | |
| (Optional) self-test | Node.js 22 or newer, Chrome or Chromium | [§7](#7-optional-headless-self-test) |

## 2. Get the code

```bash
git clone --branch english https://github.com/yyshin-katech/alpamayo2-step-viewer.git
cd alpamayo2-step-viewer
git clone https://github.com/NVlabs/alpamayo2.git
git -C alpamayo2 checkout 5e7975f4a2100ee8ac1a62b79239bbabefddcbef
```

`--branch english` checks out this English version. The default branch `main` has the Korean version.

The original code is pinned to one commit. `walk/` hooks into the module structure of `alpamayo2_super` (layer names, the order of the forward computation) and replays the model-building steps of `from_pretrained`. It was tested at this commit (2026-09-15, "Treat empty string as missing in resolve_checkpoint_name_or_path (#16)"). At other commits the capture may break or the values may differ.

The `alpamayo2/` folder is listed in `.gitignore`, so it is never committed to this repository.

## 3. Python environment

uv creates a virtual environment (`.venv`) inside the repository. The system Python and global packages are left untouched.

If you do not have uv, install it into `tools/` inside the repository. This does not change your global PATH or shell configuration.

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$PWD/tools" sh
```

`env.sh` sets environment variables so that the Python and the cache that uv downloads stay in `tools/` inside the repository. Run it once in every new terminal.

```bash
source env.sh
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.lock.txt
uv pip install --python .venv --no-deps -e ./alpamayo2
```

- `requirements.lock.txt` pins the runtime dependencies of `alpamayo2_super` to exact versions (torch 2.8.0 + CUDA 12.8 wheels, transformers 4.57.1, huggingface-hub 0.36.2, physical-ai-av 0.2.2, and others). flash-attn is left out; this tool uses PyTorch SDPA attention.
- `alpamayo2_super` is installed with `--no-deps`. The original package lists `flash-attn` as a dependency, so a plain install tries to build flash-attn.

Check:

```bash
.venv/bin/python -c "import torch, alpamayo2_super; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You should see `2.8.0+cu128 True <GPU name>`.

## 4. Download the model and data

Follow [model-and-data.md](model-and-data.md).

1. Log in to Hugging Face (`.venv/bin/hf auth login`)
2. Accept the license on the dataset page
3. Download the model (`.venv/bin/hf download nvidia/Alpamayo2-Super --revision 00554695e729a6ff0b6281fd2c81b18d06e33dbe`)

The sample is fetched in 5.1 below.

## 5. Run

Order: `fetch_data.py` → `run.py` → `analyze.py` → `serve.py`. If the output of an earlier step is missing, each script stops with a message that says what to run first.

### 5.1 Fetch the sample: `walk/fetch_data.py` (online, CPU)

```bash
.venv/bin/python walk/fetch_data.py
```

Downloads only notebook sample 0 (clip `030c760c-ae38-49aa-9ad8-f5650a545d26`, t0 = 5.1 s) from PhysicalAI-AV and saves it to `walk/out/data/` (about 300 MB). It holds 6 cameras × 4 frames, the past trajectory and the camera calibration. You must be logged in with an account that has accepted the dataset license.

### 5.2 Capture: `walk/run.py` (GPU, offline)

```bash
.venv/bin/python walk/run.py
```

- Builds the model on the meta device (`walk/build.py`), then reads the weights from disk one layer at a time, loads them onto the GPU, computes, and unloads them again (`walk/stream.py`). This is how a 72 GB checkpoint runs in 10 GB of GPU memory. The loaded values are the exact checkpoint bytes, and the dtypes are the same as with `from_pretrained`.
- The price is that the LLM weights (about 62 GB) are read once for the prefill and once more for every CoT token generated. By default 12 of the 64 layers stay in memory and are not read again. A single run reads several hundred GB, so its run time is set almost entirely by the speed of the storage.
- The inference settings are pinned in `walk/common.py`: seed 42, top-p 0.98, temperature 0.6, 1 trajectory sample, 10 flow steps, bf16, SDPA attention.
- Outputs go to `walk/out/raw/` (per-step tensors and `meta.json`, about 9 GB) and `walk/out/result/` (the figure and JSON of notebook cell 7).
- Before starting, it checks for a CUDA GPU, every shard of the model snapshot, `walk/out/data/sample0.pt`, and the free space (at least 30 GB) on the disk that holds `walk/out`.
- It stops if `walk/out/raw` is not empty. To capture again, pass `--force`; it deletes the existing `raw` and writes a new one.
- It turns on `HF_HUB_OFFLINE=1` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default. If you have already set them in your environment, your values are used.
- Close other programs that use the GPU while it runs.

### 5.3 Analysis: `walk/analyze.py` (GPU, offline)

```bash
HF_HUB_OFFLINE=1 .venv/bin/python walk/analyze.py
```

- Only reads the capture (`walk/out/raw`). It writes summaries that are too heavy to compute in the browser (token statistics across all layers, PCA, logit lens, v-lens, SQNR tables, frame images) to `walk/out/derived/`. From the checkpoint it reads only a few head weights, such as lm_head and the final norm.
- For each lens, it checks that the last layer reproduces bit for bit what the model actually produced (the final norm output, the lm_head logits, the expert velocity). The results are written to `derived/checks.json` and shown in the Checks tab of the viewer.
- To rebuild only some parts, pass the ones you need, separated by commas, to `--only`, choosing from `tokens,vision,llm,lens,expert,quant,images,traj,manifest`. `manifest` is the list that includes the check results, so if you rebuilt other parts, add `manifest` at the end as well. Example: `--only lens,manifest`

### 5.4 Viewer: `viewer/serve.py`

```bash
python3 viewer/serve.py
```

- Open <http://127.0.0.1:8765/viewer/index.html> in a browser.
- It uses only the standard library, so the system `python3` is fine. The browser does not download whole tensor files; it reads only the bytes it needs to show, over HTTP Range.
- To change the port, use `--port 9000` or `WALK_PORT=9000`.
- Controls: **Next step ▶** or the → key moves forward, and ← moves back. The **Detailed** list shows the five parts of each block or layer separately. Click a value in a chart or table to see its stored bits, what its index means and its neighboring values in the inspector on the right. The **A** key opens and closes the analysis drawer (Esc also closes it). The **How to use** panel in the viewer (the **?** button) explains the same things.
- The values are a capture of a single sample (batch 1, 1 trajectory sample). Read them as relative comparisons between layers, steps and methods. Values marked "estimate" (the v-lens and others) are not values the model produced directly.

> [!IMPORTANT]
> The server binds only to `127.0.0.1` and rejects with 421 any request whose Host header is not `127.0.0.1` or `localhost` (this blocks DNS rebinding). **Do not change it to bind to another address, and do not show it to others through port forwarding, tunnels, reverse proxies or remote screen sharing.** The camera frames and values on the screen come from the dataset ([model-and-data.md §6](model-and-data.md#6-what-the-licenses-require)).

## 6. (Optional) Check scripts

| Command | What it does | What it needs |
|---|---|---|
| `.venv/bin/python walk/test_tiny.py [--rebuild]` | Uses a random model shrunk only in width and depth (the prompt construction, 24 images, DeepStack and the shared KV cache stay the same) to compare a reference `from_pretrained` run with a streaming + capture run, and checks that the trajectory, log-probabilities, tokens and text are bit-exact. Outputs go to `walk/out/tiny*`. `--rebuild` rebuilds the small model. | GPU, the config and tokenizer files of the model snapshot, `walk/out/data/sample0.pt` |
| `.venv/bin/python walk/test_stream.py` | On the real snapshot, checks that the streaming loader loads every tensor with the same values as the checkpoint, and measures the read speed. | GPU, model snapshot |
| `.venv/bin/python walk/inspect_prompt.py` | Builds the model input on the CPU and writes the token layout to `walk/out/data/prompt_layout.txt`. | Model snapshot, `walk/out/data/sample0.pt` |

## 7. (Optional) Headless self-test

The viewer has a built-in self-test that renders every step in turn and collects errors. Add `?selftest=1` to the address to run it; the result is left in `window.__selftest`. `viewer/tools/cdp.mjs` runs it in headless Chrome and saves the result as JSON (Node.js 22 or newer, no external packages).

Run it while the server is up. The result JSON can contain values from the capture, so keep it under `walk/out/`.

```bash
node viewer/tools/cdp.mjs test "" walk/out/selftest_base.json                        # every step in normal mode + analysis drawer
node viewer/tools/cdp.mjs test "detail=1&drawer=0" walk/out/selftest_detail.json     # detailed mode
node viewer/tools/cdp.mjs test "variants=1&drawer=0" walk/out/selftest_variants.json # selection variants
node viewer/tools/cdp.mjs test "monkey=1" walk/out/selftest_monkey.json              # random clicks
node viewer/tools/cdp.mjs test "from=53&to=53&recomp=1" walk/out/selftest_recomp.json # full recomputation in the drawer
```

- When it finishes, it prints one `DONE ok=true … errors=0 …` line and, if there are any, the lists of errors and warnings.
- Options:
  - `detail=1`: detailed mode.
  - `from`, `to`: step range.
  - `drawer=0`: skip the analysis drawer checks.
  - `variants=1`: five selection variants.
  - `monkey=1`: clicks buttons, select boxes, sliders and canvases at random. It does not click links.
  - `recomp=1`: runs the full recomputation in the drawer. Works only when the drawer checks are on, and waits up to 30 minutes.
- The drawer checks always use the LLM layer 20 step (step 53 in normal mode) as their background, whatever the range. So to check only the drawer, pass `from=53&to=53`.
- The fourth argument is the time limit (seconds, default 3600), and the fifth is the Chrome debugging port (default 9222). When you run several at once, give each a different port.
- Environment variables:
  - `CHROME`: the Chrome/Chromium executable. The default is `/usr/bin/google-chrome`.
  - `CDP_BASE`: the viewer address. The default is `http://127.0.0.1:8765/viewer/index.html`, so change it if you changed the server port.
  - `CDP_TMP`: where to put the temporary Chrome profile. The default is the system temp folder, and the profile is deleted at the end.
- To stop a long check, use `kill <PID>` (SIGTERM). If you stop it with `kill -9`, the Chrome processes and the temporary profile (`chrome-prof-<port>`) are left behind.
- `shot` mode (`node viewer/tools/cdp.mjs shot "<query-and-hash>" out.png [w] [h] [port]`) saves the screen as a PNG. Screenshots contain dataset frames and values, so do not let them leave your computer.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `model snapshot missing or incomplete` | The model has not been downloaded, or only partly. Run the same `hf download` command again to resume. If you downloaded it elsewhere, set `HF_HUB_CACHE` or `ALPAMAYO2_SUPER_SNAPSHOT` to match ([model-and-data.md §4](model-and-data.md#4-download-the-model)). |
| `OSError: [Errno 22] Invalid argument` (when the streaming loader opens or reads a file) | The model folder is on a file system that does not support O_DIRECT. tmpfs, some FUSE and network drives, and Windows drives such as `/mnt/c` on WSL2 can behave this way. Move it to a local ext4 or xfs disk. |
| `RuntimeError: cudaHostRegister failed`, or the process suddenly dies during the run (OOM killer) | Not enough RAM. Remove layers from the default `cache_ids=(0, 5, 11, …, 59)` of `Streamer.__init__` in `walk/stream.py`. Each layer removed saves about 1 GB of RAM, but about 1 GB more is read for the prefill and for every decode step. `cache_ids=()` turns the cache off. |
| `CUDA out of memory` | Close other programs that use the GPU (check with `nvidia-smi`) and run again. It does not run on GPUs with less than 10 GB of memory. |
| `401`, `403`, `GatedRepoError` (`fetch_data.py`, `hf download`) | You are not logged in, you have not accepted the license on the dataset page, or your token lacks permissions ([model-and-data.md §2–3](model-and-data.md#2-hugging-face-account-and-token)). |
| `fetch_data.py` fails with an offline error | `HF_HUB_OFFLINE=1` is still set in your shell. Run `unset HF_HUB_OFFLINE` and try again. |
| `no sample under …/walk/out/data` | Run `walk/fetch_data.py` first. |
| `only … GB free under …/walk/out` | The disk that holds `walk/out` has less than 30 GB free. |
| `…/walk/out/raw is not empty; pass --force to overwrite` | A capture already exists. Pass `--force` to make a new one. |
| `no capture under …` / `no derived data under …` | Run `run.py` and then `analyze.py` first. |
| `cannot listen on 127.0.0.1:8765` | The port is already in use. Pick another port with `--port`. |
| `421 unknown Host` in the browser | Open `http://127.0.0.1:<port>/viewer/index.html`, or use `localhost`. The server is built so that it does not open under other names or IP addresses. |
| Opening the file directly shows a message asking you to use the server | HTTP Range does not work over `file://`. Open it through `serve.py`. |
| Characters such as x̂ and γ̂ show up as boxes | No installed font has the combining characters. On Ubuntu: `sudo apt install fonts-noto-core fonts-noto-mono fonts-dejavu-core` |
| `cdp.mjs` cannot find Chrome | Point to the executable, as in `CHROME=/usr/bin/chromium node viewer/tools/cdp.mjs …`. |

## 9. Clean up

How to delete the captures, the sample and the caches is described in [model-and-data.md §7](model-and-data.md#7-clean-up). The dataset license requires you to destroy all copies when it ends.
