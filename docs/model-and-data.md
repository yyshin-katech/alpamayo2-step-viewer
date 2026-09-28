# Downloading the model and the dataset

> [!NOTE]
> Checked on 2026-09-28. The license summaries are not legal advice, and the original texts take precedence. Read the originals yourself before you download.

This repository contains neither the model weights nor the dataset. Download both with your own Hugging Face account.

## 1. At a glance

| Item | Repository | License | Size | Access |
|---|---|---|---|---|
| Model weights | [nvidia/Alpamayo2-Super](https://huggingface.co/nvidia/Alpamayo2-Super) | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) | About 72 GB (15 safetensors shards) | When checked on 2026-09-28, it could be downloaded without an agreement step. The original README says it is gated, so if an agreement screen appears, follow it. |
| Dataset | [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) | NVIDIA Autonomous Vehicle Dataset License Agreement | This tool downloads only one sample (a few hundred MB) and the metadata | You must accept the license on the dataset page (gated) |
| Model code | [NVlabs/alpamayo2](https://github.com/NVlabs/alpamayo2) | Apache-2.0 | Small | Public. How to clone it: [local-run.md §2](local-run.md#2-get-the-code) |

## 2. Hugging Face account and token

1. Sign up at <https://huggingface.co>.
2. Create a token under Settings → Access Tokens. A `Read` token is enough. For a fine-grained token, turn on "Read access to contents of all public gated repos you can access".
3. Log in. Paste the token. You can answer `n` to "Add token as git credential?".

   ```bash
   .venv/bin/hf auth login
   ```

4. Check: `.venv/bin/hf auth whoami`

The token is saved to `~/.cache/huggingface/token` (under `HF_HOME` if you changed it). Do not commit or share this file or the token string.

## 3. Accept the dataset license

1. Signed in with the same account, open the [dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).
2. Read the license (NVIDIA Autonomous Vehicle Dataset License Agreement) to the end and accept it. You get access right away. After you download, the original PDF also stays in the cache (`LICENSE.pdf`).
3. Read the obligations in [§6](#6-what-the-licenses-require). Above all, first check that your purpose is within what the license allows (§1).

## 4. Download the model

```bash
.venv/bin/hf download nvidia/Alpamayo2-Super --revision 00554695e729a6ff0b6281fd2c81b18d06e33dbe
```

- `--revision` must match `MODEL_REVISION` in `walk/common.py`. This tool was tested at this revision, and the code looks for this revision's folder.
- Where it goes: `~/.cache/huggingface/hub/models--nvidia--Alpamayo2-Super/snapshots/00554695e729a6ff0b6281fd2c81b18d06e33dbe/`. It holds the 15 shards (`model-000NN-of-00015.safetensors`), `model.safetensors.index.json`, and the config, preprocessor and tokenizer files.
- If the download is interrupted, run the same command again. Files already downloaded are skipped, and the rest resumes.
- To keep it on another disk, use one of the two options below. Either way, it must be a local disk that supports O_DIRECT (ext4, xfs, etc.; NVMe recommended). Avoid putting it inside this repository folder.
  - Change the cache location. Set the same value, such as `export HF_HUB_CACHE=/data/hf-hub`, in both the shell that downloads and the shell that runs. If you change `HF_HOME` instead, the token location moves with it.
  - Download straight into a folder: `.venv/bin/hf download nvidia/Alpamayo2-Super --revision 00554695e729a6ff0b6281fd2c81b18d06e33dbe --local-dir /data/Alpamayo2-Super`. Then set `export ALPAMAYO2_SUPER_SNAPSHOT=/data/Alpamayo2-Super` before you run anything.
- Check:

  ```bash
  .venv/bin/python -c "import sys; sys.path.insert(0, 'walk'); from common import check_snapshot; s = check_snapshot(); print(s, len(list(s.glob('*.safetensors'))), 'shards')"
  ```

  You should see the path and `15 shards`. If any files are missing, it tells you which ones and the command that downloads them.

## 5. Download the dataset sample

```bash
.venv/bin/python walk/fetch_data.py
```

- Downloads only notebook sample 0 (clip `030c760c-ae38-49aa-9ad8-f5650a545d26`, t0 = 5.1 s). It does not download the whole dataset.
- Writes `source.pt`, `sample0.pt` and `sample0_keys.json` (about 300 MB in total) to `walk/out/data/`.
- The metadata and the license PDF stay in the Hugging Face cache (`…/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/`), about 33 MB in total.
- The loader (`physical_ai_av` 0.2.2) reads the latest (main) revision of the dataset. When this tool was made, main was `33f9bf447ed3bcb7d545ce13f4226f824214fafb`. If the dataset is updated, the sample may change.
- If you get `401`, `403` or `GatedRepoError`, check the login in §2 and the agreement in §3. If `HF_HUB_OFFLINE=1` is still set in your shell, nothing can be downloaded, so run `unset HF_HUB_OFFLINE`.

## 6. What the licenses require

### Model weights: OpenMDW-1.1

A permissive license that places no restrictions on the use of outputs. If you redistribute the weights, you must include the license and notices with them. This repository does not distribute the weights. Original text: <https://openmdw.ai/license/1-1/>

### Dataset: NVIDIA Autonomous Vehicle Dataset License Agreement (summary)

| Section | Summary |
|---|---|
| §1 | Rights are granted solely "for your **internal development** of autonomous vehicles and automated driving assisted systems using NVIDIA technology". |
| §3 | Defines the dataset itself, its output, and any results of benchmarking, competitive analysis, regression or performance data relating to the dataset as **Confidential Information**, and forbids disclosing it to third parties other than Authorized Users. |
| §4.1 | Forbids using it to monitor individuals or groups in an unethical manner, or to enable law enforcement (including road traffic laws). |
| §4.4, §4.5 | Forbids identifying or profiling individuals (including by license plate numbers), attempting to de-anonymize the dataset, inferring sensitive attributes such as race, gender, age or health, and biometric processing. |
| §4.6 | Forbids distributing, selling, renting, sublicensing, transferring, embedding or hosting the dataset **in whole or in part**, or otherwise making it available to others, and creating derivative works of the dataset. |
| §4.8 | You must keep track of where your copies are stored. |
| §8 | Expires 12 months after the date of initial delivery or download. On termination, you must stop using the dataset and destroy all copies. You must also delete it upon written notice by NVIDIA (§4.9). |

Read the original and decide for yourself how the captures this tool makes (camera frames, intermediate tensors, trajectories) fall under these sections. At the very least, do not let them leave your computer.

### What this means for this tool

Treat everything below as coming from the dataset and keep it on your own computer only.

- All of `walk/out/`: `data/` (the sample), `raw/` (captured tensors), `derived/` (analysis results, including frame images), `result/` (trajectory figure and JSON), `tiny/` and `tiny_raw/` (for checks), and self-test result JSON
- Viewer screens: screenshots, recordings, screen sharing
- CoT text, metrics such as minADE/minFDE, and performance numbers such as run times

In practice:

- Do not commit, upload or post them, or share them over messengers. `.gitignore` blocks `walk/out/`, images and logs, but `git add -f` can still add them, so be careful.
- Do not expose the viewer server beyond 127.0.0.1. Do not change the bind address, and do not use port forwarding, tunnels or reverse proxies.
- In issues and questions, include only error messages and your environment (OS, GPU, driver, package versions). Do not attach data, captures, screenshots or result numbers.
- Write down the date of your first download (the license expires after 12 months).

## 7. Clean up

```bash
rm -rf walk/out                                                                   # sample, captures, analysis, results
rm -rf ~/.cache/huggingface/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles  # dataset metadata, license PDF
rm -rf ~/.cache/huggingface/hub/models--nvidia--Alpamayo2-Super                   # (optional) the 72 GB model
rm -rf ~/.cache/huggingface/xet                                                   # (optional) transfer cache; caches of other repositories go too
.venv/bin/hf auth logout                                                          # delete the token
```

- Pieces of downloaded files can also remain in the xet transfer cache. When you must destroy every copy of the dataset, delete it as well.
- If you moved things with `HF_HOME`, `HF_HUB_CACHE` or `--local-dir`, delete them at those paths.
- If you copied `walk/out/` somewhere else, delete that copy too (§4.8, §8).
