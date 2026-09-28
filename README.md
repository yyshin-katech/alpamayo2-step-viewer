# alpamayo2-step-viewer

A local tool for following, one step at a time, how NVIDIA **Alpamayo 2 Super** processes a single driving scene, using values you capture by running the model on your own GPU.

- **`walk/`** runs inference once and saves the intermediate tensors of every step as safetensors. It reads the 72 GB checkpoint from disk one layer at a time, so it runs in 10 GB of GPU memory. The weights are loaded as the exact checkpoint bytes, and a small random model is used to check bit for bit that streaming and capture do not change the computation (`walk/test_tiny.py`).
- **`viewer/`** shows the saved tensors in the browser, one step at a time: camera patches → vision encoder → 64-layer LLM prefill → CoT decoding → flow matching in the action expert → predicted trajectory. It has normal and detailed modes, an inspector that shows the stored bits and neighboring values of any value you click, and an analysis drawer (11 tabs) that looks across layers.

> This branch (`english`) is the English version. The Korean version is on the `main` branch.

## What is included and what is not

| In this repository | Not in this repository (download or create it yourself) |
|---|---|
| Capture and analysis scripts (`walk/`) | Model weights `nvidia/Alpamayo2-Super` (about 72 GB) |
| Viewer (`viewer/`: HTML, CSS and JS, a local server, a headless self-test driver) | A sample from the dataset `nvidia/PhysicalAI-Autonomous-Vehicles` |
| Pinned dependency list (`requirements.lock.txt`) | Model code `NVlabs/alpamayo2` (cloned at a pinned commit) |
| Run guide and model/data download guide (`docs/`) | Captures (`walk/out/`), screenshots, logs, result numbers |

> [!WARNING]
> **Look at captures and viewer screens only on your own computer.** Every file under `walk/out/` and everything the viewer shows (screenshots, recordings, screen sharing) comes from the PhysicalAI-AV dataset. The dataset license forbids distributing or hosting the dataset, even in part, and treats results and performance numbers about the dataset as confidential information. Do not commit, publish or share them, and do not expose the viewer server beyond 127.0.0.1. See [docs/model-and-data.md](docs/model-and-data.md) for details.

## Requirements (summary)

- Linux x86_64 (tested on Ubuntu 22.04)
- NVIDIA GPU with at least 10 GB of memory and bf16 support (Ampere or newer recommended), and a driver that supports CUDA 12.8
- 32 GB of RAM recommended, about 120 GB of free disk space (a local disk that supports O_DIRECT; NVMe recommended)
- A Hugging Face account (you must accept the dataset license)
- uv downloads Python 3.12 for you. The system Python and global packages are left untouched.

The full table and the reasons are in [docs/local-run.md](docs/local-run.md#1-requirements).

## Quick start

```bash
# 1. Code (this repository + the original model code at a pinned commit)
git clone --branch english https://github.com/yyshin-katech/alpamayo2-step-viewer.git
cd alpamayo2-step-viewer
git clone https://github.com/NVlabs/alpamayo2.git
git -C alpamayo2 checkout 5e7975f4a2100ee8ac1a62b79239bbabefddcbef

# 2. Python environment (if you do not have uv yet: curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$PWD/tools" sh)
source env.sh
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.lock.txt
uv pip install --python .venv --no-deps -e ./alpamayo2

# 3. Log in → accept the license on the dataset page → download the model (docs/model-and-data.md)
.venv/bin/hf auth login
.venv/bin/hf download nvidia/Alpamayo2-Super --revision 00554695e729a6ff0b6281fd2c81b18d06e33dbe

# 4. Fetch the sample → capture → analyze → view
.venv/bin/python walk/fetch_data.py
.venv/bin/python walk/run.py
HF_HUB_OFFLINE=1 .venv/bin/python walk/analyze.py
python3 viewer/serve.py        # then open http://127.0.0.1:8765/viewer/index.html in a browser
```

`run.py` reads the weights from disk many times over, so its run time is set almost entirely by the speed of your storage. What each step does and what it checks is described in [docs/local-run.md](docs/local-run.md).

## Documentation

- [docs/local-run.md](docs/local-run.md): requirements, installation, running each step, optional checks and self-tests, troubleshooting
- [docs/model-and-data.md](docs/model-and-data.md): Hugging Face account and token, downloading the model and the dataset, what the licenses require, cleanup

## Folder layout

```text
walk/
  common.py          paths, pinned sample and model revision, snapshot check
  fetch_data.py      fetch notebook sample 0 (online)
  build.py           build the model on the meta device (replays the from_pretrained steps)
  stream.py          layer-by-layer weight streaming (O_DIRECT + page-locked memory)
  capture.py         hooks that record intermediates (the model computation is untouched)
  run.py             capture run → walk/out/raw, walk/out/result
  analyze.py         derived analysis → walk/out/derived
  inspect_prompt.py  (optional) show the prompt token layout
  test_tiny.py       (optional) bit-exact check of streaming and capture on a small random model
  test_stream.py     (optional) check the streaming loader on the real snapshot
viewer/
  index.html  css/  js/   the viewer (no external resources)
  serve.py                local server (standard library only, 127.0.0.1 only)
  tools/cdp.mjs           (optional) headless Chrome self-test driver
docs/                     guides
env.sh                    environment variables that keep uv's Python and cache in tools/ inside the repository
requirements.lock.txt     pinned dependencies (Python 3.12, Linux x86_64, torch 2.8.0 + CUDA 12.8)
```

The folders you create yourself, `alpamayo2/` (the original clone), `.venv/`, `tools/` (uv) and `walk/out/` (sample and captures), are listed in `.gitignore`.

## License

- Code and documentation in this repository: Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE))
- The model weights (OpenMDW-1.1) and the dataset (NVIDIA Autonomous Vehicle Dataset License) are not included in this repository and are covered by their own licenses.

This is an unofficial project. It is not affiliated with or endorsed by NVIDIA.
