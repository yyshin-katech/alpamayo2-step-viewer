# 로컬 실행 가이드

코드를 받아 자기 컴퓨터에서 캡처를 만들고 뷰어로 보기까지의 과정입니다. 모델·데이터셋을 받는 방법과 라이선스는 [model-and-data.md](model-and-data.md)에 따로 정리했습니다.

명령은 모두 레포 폴더(`alpamayo2-step-viewer/`)에서 실행합니다.

## 1. 요구 사항

| 항목 | 요구 | 비고 |
|---|---|---|
| OS | Linux x86_64 | Ubuntu 22.04에서 확인했습니다. 의존성 목록이 Linux x86_64용입니다. macOS·Windows에서는 돌지 않습니다(CUDA, O_DIRECT). WSL2는 확인하지 않았습니다. |
| GPU | NVIDIA, 메모리 10 GB 이상, bf16 지원 | Ampere(RTX 30xx) 이상을 권장합니다. RTX 3080 10 GB에서 확인했습니다. |
| 드라이버 | CUDA 12.8을 지원하는 NVIDIA 드라이버 | R570 이상을 권장합니다. PyTorch 휠에 CUDA 런타임이 들어 있어 CUDA 툴킷(nvcc)은 필요 없습니다. |
| RAM | 32 GB 권장 | 가중치 일부를 page-locked 메모리에 올려 둡니다. 기본 설정에서 약 14 GB입니다(추정). 부족하면 [§8](#8-문제-해결)을 보세요. |
| 저장공간 | 빈 공간 약 120 GB | 모델 72 GB, `walk/out` 아래 여유 30 GB 이상(캡처 자체는 약 9 GB), `.venv` 약 7 GB, uv 캐시 약 4 GB |
| 파일시스템 | 모델을 둔 곳이 O_DIRECT를 지원하는 로컬 디스크(ext4, xfs 등) | NVMe SSD를 강하게 권장합니다. 실행 시간을 거의 저장장치 속도가 정합니다. |
| 네트워크 | 받는 양: 휠 약 4 GB, 모델 72 GB, 샘플 수백 MB | 캡처·분석·뷰어는 오프라인으로 돕니다. 뷰어는 외부 리소스를 읽지 않습니다. |
| 브라우저 | 최신 Chrome, Edge, Firefox 등 | |
| (선택) 셀프테스트 | Node.js 22 이상, Chrome 또는 Chromium | [§7](#7-선택-헤드리스-셀프테스트) |

## 2. 코드 받기

```bash
git clone https://github.com/yyshin-katech/alpamayo2-step-viewer.git
cd alpamayo2-step-viewer
git clone https://github.com/NVlabs/alpamayo2.git
git -C alpamayo2 checkout 5e7975f4a2100ee8ac1a62b79239bbabefddcbef
```

원본 코드는 커밋을 고정합니다. `walk/`는 `alpamayo2_super`의 모듈 구조(레이어 이름, 전방 계산 순서)에 훅을 걸고 `from_pretrained`의 모델 구성 단계를 그대로 따라 합니다. 이 커밋(2026-09-15, "Treat empty string as missing in resolve_checkpoint_name_or_path (#16)")에서 확인했습니다. 다른 커밋에서는 캡처가 깨지거나 값이 달라질 수 있습니다.

`alpamayo2/` 폴더는 `.gitignore`에 들어 있어 이 레포에 커밋되지 않습니다.

## 3. Python 환경

uv로 레포 안에 가상환경(`.venv`)을 만듭니다. 시스템 Python과 전역 패키지는 건드리지 않습니다.

uv가 없으면 레포 안 `tools/`에 설치합니다. 전역 PATH나 셸 설정은 바꾸지 않습니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$PWD/tools" sh
```

`env.sh`는 uv가 받는 Python과 캐시를 레포 안 `tools/`에 두도록 환경변수를 정합니다. 새 터미널을 열 때마다 한 번 실행합니다.

```bash
source env.sh
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.lock.txt
uv pip install --python .venv --no-deps -e ./alpamayo2
```

- `requirements.lock.txt`는 `alpamayo2_super`의 런타임 의존성을 버전까지 고정한 목록입니다(torch 2.8.0 + CUDA 12.8 휠, transformers 4.57.1, huggingface-hub 0.36.2, physical-ai-av 0.2.2 등). flash-attn은 뺐습니다. 이 도구는 PyTorch SDPA 어텐션을 씁니다.
- `alpamayo2_super`는 `--no-deps`로 설치합니다. 원본 패키지는 `flash-attn`을 의존성으로 적어 두어서, 그냥 설치하면 flash-attn 빌드를 시도합니다.

확인:

```bash
.venv/bin/python -c "import torch, alpamayo2_super; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`2.8.0+cu128 True <GPU 이름>`이 나오면 됩니다.

## 4. 모델과 데이터 받기

[model-and-data.md](model-and-data.md)를 따라 진행합니다.

1. Hugging Face 로그인 (`.venv/bin/hf auth login`)
2. 데이터셋 페이지에서 라이선스 동의
3. 모델 받기 (`.venv/bin/hf download nvidia/Alpamayo2-Super --revision 00554695e729a6ff0b6281fd2c81b18d06e33dbe`)

샘플은 아래 5.1에서 받습니다.

## 5. 실행

순서: `fetch_data.py` → `run.py` → `analyze.py` → `serve.py`. 앞 단계의 결과가 없으면 각 스크립트가 무엇을 먼저 하라는 메시지를 내고 멈춥니다.

### 5.1 샘플 받기: `walk/fetch_data.py` (온라인, CPU)

```bash
.venv/bin/python walk/fetch_data.py
```

노트북 샘플 0(clip `030c760c-ae38-49aa-9ad8-f5650a545d26`, t0 = 5.1 s) 하나만 PhysicalAI-AV에서 받아 `walk/out/data/`(약 300 MB)에 저장합니다. 카메라 6대 × 4프레임, 과거 궤적, 카메라 보정값이 들어 있습니다. 데이터셋 라이선스에 동의한 계정으로 로그인되어 있어야 합니다.

### 5.2 캡처: `walk/run.py` (GPU, 오프라인)

```bash
.venv/bin/python walk/run.py
```

- 모델을 meta 장치에 만든 뒤(`walk/build.py`), 가중치를 레이어 단위로 디스크에서 읽어 GPU에 올리고, 계산하고, 다시 내립니다(`walk/stream.py`). 그래서 GPU 메모리 10 GB로 72 GB 체크포인트를 돌릴 수 있습니다. 올리는 값은 체크포인트 바이트 그대로이고 dtype도 `from_pretrained`와 같습니다.
- 그 대신 LLM 가중치(약 62 GB)를 프리필에서 한 번 읽고, CoT 토큰을 하나 만들 때마다 한 번씩 다시 읽습니다. 64층 가운데 12층은 기본으로 메모리에 올려 두어 다시 읽지 않습니다. 한 번 실행에 수백 GB를 읽게 되므로 실행 시간은 거의 저장장치 속도가 정합니다.
- 추론 설정은 `walk/common.py`에 고정되어 있습니다: seed 42, top-p 0.98, temperature 0.6, 궤적 샘플 1개, 플로 스텝 10, bf16, 어텐션은 SDPA.
- 결과는 `walk/out/raw/`(단계별 텐서와 `meta.json`, 약 9 GB)와 `walk/out/result/`(노트북 셀 7의 그림·JSON)에 씁니다.
- 시작 전에 CUDA GPU, 모델 스냅샷의 모든 샤드, `walk/out/data/sample0.pt`, `walk/out`이 있는 디스크의 빈 공간(30 GB 이상)을 확인합니다.
- `walk/out/raw`가 비어 있지 않으면 멈춥니다. 다시 캡처하려면 `--force`를 줍니다. 기존 `raw`를 지우고 새로 씁니다.
- `HF_HUB_OFFLINE=1`과 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 기본으로 켭니다. 환경변수로 이미 정해 두었다면 그 값을 씁니다.
- 실행 중에는 GPU를 쓰는 다른 프로그램을 닫아 두세요.

### 5.3 분석: `walk/analyze.py` (GPU, 오프라인)

```bash
HF_HUB_OFFLINE=1 .venv/bin/python walk/analyze.py
```

- 캡처(`walk/out/raw`)는 읽기만 합니다. 브라우저에서 계산하기엔 무거운 요약(층 전체의 토큰 통계, PCA, 로짓 렌즈, v-렌즈, SQNR 표, 프레임 이미지)을 `walk/out/derived/`에 씁니다. 체크포인트에서는 lm_head와 최종 norm 같은 헤드 가중치 몇 개만 읽습니다.
- 렌즈마다 마지막 층이 모델이 실제로 낸 값(최종 norm 출력, lm_head 로짓, 전문가 속도)을 비트 단위로 재현하는지 확인합니다. 결과는 `derived/checks.json`에 적히고 뷰어의 검증 탭에 나옵니다.
- 일부만 다시 만들려면 `--only`에 `tokens,vision,llm,lens,expert,quant,images,traj,manifest` 가운데 필요한 것을 쉼표로 적습니다. `manifest`는 검사 결과를 포함한 목록이라서, 다른 부분을 다시 만들었다면 `manifest`를 맨 뒤에 함께 적습니다. 예: `--only lens,manifest`

### 5.4 뷰어: `viewer/serve.py`

```bash
python3 viewer/serve.py
```

- 브라우저에서 <http://127.0.0.1:8765/viewer/index.html> 을 엽니다.
- 표준 라이브러리만 쓰므로 시스템 `python3`로 돌려도 됩니다. 브라우저는 텐서 파일 전체를 받지 않고, HTTP Range로 보여 줄 부분의 바이트만 읽습니다.
- 포트를 바꾸려면 `--port 9000` 또는 `WALK_PORT=9000`을 씁니다.
- 조작: **다음 단계 ▶** 또는 → 키로 진행하고 ←로 돌아갑니다. **세부** 목록은 블록·레이어마다 다섯 부분을 따로 봅니다. 차트·표의 값을 누르면 오른쪽 인스펙터에 저장된 비트, 인덱스의 의미, 이웃 값이 나옵니다. **A** 키는 분석 서랍을 열고 닫습니다(Esc로도 닫힘). 화면 안의 **사용법**에도 같은 설명이 있습니다.
- 값은 한 샘플(배치 1, 궤적 샘플 1개)의 캡처입니다. 층·단계·방식 사이의 상대 비교로 읽으세요. "추정"이라고 표시한 값(v-렌즈 등)은 모델이 직접 낸 값이 아닙니다.

> [!IMPORTANT]
> 서버는 `127.0.0.1`에만 바인드하고, Host 헤더가 `127.0.0.1`이나 `localhost`가 아닌 요청은 421로 거절합니다(DNS 리바인딩 차단). **다른 주소로 바인드하도록 고치거나 포트 포워딩, 터널, 리버스 프록시, 원격 화면 공유로 남에게 보이지 마세요.** 화면에 나오는 카메라 프레임과 값은 데이터셋에서 나온 것입니다([model-and-data.md §6](model-and-data.md#6-라이선스에서-지킬-것)).

## 6. (선택) 점검 스크립트

| 명령 | 하는 일 | 필요한 것 |
|---|---|---|
| `.venv/bin/python walk/test_tiny.py [--rebuild]` | 너비·깊이만 줄인 무작위 모델(프롬프트 구성, 이미지 24장, 딥스택, 공유 KV 캐시는 그대로)로 `from_pretrained` 기준 실행과 스트리밍+캡처 실행을 비교해, 궤적·로그확률·토큰·텍스트가 비트 단위로 같은지 확인합니다. 결과는 `walk/out/tiny*`에 씁니다. `--rebuild`는 작은 모델을 다시 만듭니다. | GPU, 모델 스냅샷의 설정·토크나이저 파일, `walk/out/data/sample0.pt` |
| `.venv/bin/python walk/test_stream.py` | 실제 스냅샷에서 스트리밍 로더가 텐서를 빠짐없이, 체크포인트와 같은 값으로 올리는지와 읽기 속도를 확인합니다. | GPU, 모델 스냅샷 |
| `.venv/bin/python walk/inspect_prompt.py` | 모델 입력을 CPU에서 만들어 토큰 배치를 `walk/out/data/prompt_layout.txt`에 적습니다. | 모델 스냅샷, `walk/out/data/sample0.pt` |

## 7. (선택) 헤드리스 셀프테스트

뷰어에는 모든 단계를 차례로 그려 보며 오류를 모으는 셀프테스트가 들어 있습니다. 주소에 `?selftest=1`을 붙여 열면 돌고, 결과는 `window.__selftest`에 남습니다. `viewer/tools/cdp.mjs`는 헤드리스 Chrome으로 이를 돌리고 결과를 JSON으로 저장합니다(Node.js 22 이상, 외부 패키지 없음).

서버를 띄운 상태에서 실행합니다. 결과 JSON에는 캡처에서 나온 값이 들어갈 수 있으므로 `walk/out/` 아래에 둡니다.

```bash
node viewer/tools/cdp.mjs test "" walk/out/selftest_base.json                        # 일반 모드 전 단계 + 분석 서랍
node viewer/tools/cdp.mjs test "detail=1&drawer=0" walk/out/selftest_detail.json     # 세부 모드
node viewer/tools/cdp.mjs test "variants=1&drawer=0" walk/out/selftest_variants.json # 선택 변형
node viewer/tools/cdp.mjs test "monkey=1" walk/out/selftest_monkey.json              # 무작위 클릭
node viewer/tools/cdp.mjs test "from=53&to=53&recomp=1" walk/out/selftest_recomp.json # 서랍 전체 재계산
```

- 끝나면 `DONE ok=true … errors=0 …` 한 줄과, 있으면 오류·경고 목록을 출력합니다.
- 옵션:
  - `detail=1`: 세부 모드.
  - `from`·`to`: 단계 범위.
  - `drawer=0`: 분석 서랍 검사 생략.
  - `variants=1`: 선택 변형 다섯 가지.
  - `monkey=1`: 버튼·선택 상자·슬라이더·캔버스를 무작위로 누릅니다. 링크는 누르지 않습니다.
  - `recomp=1`: 서랍의 전체 재계산을 돌립니다. 서랍 검사를 켰을 때만 동작하고 최대 30분 기다립니다.
- 서랍 검사는 범위와 상관없이 LLM 레이어 20 단계(일반 모드 53번)를 배경으로 씁니다. 그래서 서랍만 볼 때는 `from=53&to=53`을 줍니다.
- 네 번째 인자는 제한 시간(초, 기본 3600), 다섯 번째는 Chrome 디버그 포트(기본 9222)입니다. 여러 개를 동시에 돌릴 때는 포트를 서로 다르게 줍니다.
- 환경변수:
  - `CHROME`: Chrome/Chromium 실행 파일. 기본은 `/usr/bin/google-chrome`입니다.
  - `CDP_BASE`: 뷰어 주소. 기본은 `http://127.0.0.1:8765/viewer/index.html`이므로, 서버 포트를 바꿨다면 맞춰 줍니다.
  - `CDP_TMP`: 임시 Chrome 프로필을 둘 곳. 기본은 시스템 임시 폴더이고, 끝나면 지웁니다.
- 오래 걸리는 검사를 끌 때는 `kill <PID>`(SIGTERM)을 씁니다. `kill -9`로 끄면 Chrome 프로세스와 임시 프로필(`chrome-prof-<포트>`)이 남습니다.
- `shot` 모드(`node viewer/tools/cdp.mjs shot "<query-and-hash>" out.png [w] [h] [port]`)는 화면을 PNG로 저장합니다. 스크린샷에는 데이터셋 프레임과 값이 담기므로 자기 컴퓨터 밖으로 내보내지 마세요.

## 8. 문제 해결

| 증상 | 원인과 해결 |
|---|---|
| `model snapshot missing or incomplete` | 모델을 아직 받지 않았거나 덜 받았습니다. 같은 `hf download` 명령을 다시 실행하면 이어 받습니다. 다른 위치에 받았다면 `HF_HUB_CACHE` 또는 `ALPAMAYO2_SUPER_SNAPSHOT`을 맞춥니다([model-and-data.md §4](model-and-data.md#4-모델-받기)). |
| `OSError: [Errno 22] Invalid argument` (스트리밍 로더가 파일을 열거나 읽을 때) | 모델 폴더가 O_DIRECT를 지원하지 않는 파일시스템에 있습니다. tmpfs, 일부 FUSE·네트워크 드라이브, WSL2의 `/mnt/c` 같은 Windows 드라이브가 그럴 수 있습니다. 로컬 ext4·xfs 디스크로 옮기세요. |
| `RuntimeError: cudaHostRegister failed`, 또는 실행 중 프로세스가 갑자기 죽음(OOM killer) | RAM이 부족합니다. `walk/stream.py`의 `Streamer.__init__` 기본값 `cache_ids=(0, 5, 11, …, 59)`에서 층을 줄입니다. 층 하나를 빼면 RAM이 약 1 GB 줄고, 대신 프리필과 디코드 스텝마다 약 1 GB를 더 읽습니다. `cache_ids=()`이면 캐시를 쓰지 않습니다. |
| `CUDA out of memory` | GPU를 쓰는 다른 프로그램을 닫고(`nvidia-smi`로 확인) 다시 실행합니다. 메모리가 10 GB보다 작은 GPU에서는 돌지 않습니다. |
| `401`, `403`, `GatedRepoError` (`fetch_data.py`, `hf download`) | 로그인하지 않았거나, 데이터셋 페이지에서 라이선스에 동의하지 않았거나, 토큰 권한이 부족합니다([model-and-data.md §2–3](model-and-data.md#2-hugging-face-계정과-토큰)). |
| `fetch_data.py`가 오프라인 오류를 냄 | 셸에 `HF_HUB_OFFLINE=1`이 남아 있습니다. `unset HF_HUB_OFFLINE` 뒤에 다시 실행합니다. |
| `no sample under …/walk/out/data` | `walk/fetch_data.py`를 먼저 실행합니다. |
| `only … GB free under …/walk/out` | `walk/out`이 있는 디스크의 빈 공간이 30 GB보다 적습니다. |
| `…/walk/out/raw is not empty; pass --force to overwrite` | 캡처가 이미 있습니다. 다시 만들려면 `--force`를 줍니다. |
| `no capture under …` / `no derived data under …` | `run.py` → `analyze.py` 순서로 먼저 실행합니다. |
| `cannot listen on 127.0.0.1:8765` | 포트를 이미 쓰고 있습니다. `--port`로 다른 포트를 고릅니다. |
| 브라우저에 `421 unknown Host` | `http://127.0.0.1:<포트>/viewer/index.html` 또는 `localhost`로 엽니다. 다른 이름·IP로는 열리지 않게 만들어 두었습니다. |
| 파일로 열었더니 서버로 열라는 안내가 나옴 | `file://`로는 HTTP Range를 쓸 수 없습니다. `serve.py`로 엽니다. |
| x̂, γ̂ 같은 글자가 네모로 보임 | 결합 문자를 가진 글꼴이 없습니다. Ubuntu: `sudo apt install fonts-noto-core fonts-noto-mono fonts-dejavu-core` |
| `cdp.mjs`가 Chrome을 찾지 못함 | `CHROME=/usr/bin/chromium node viewer/tools/cdp.mjs …`처럼 실행 파일을 지정합니다. |

## 9. 지우기

캡처·샘플·캐시를 지우는 방법은 [model-and-data.md §7](model-and-data.md#7-지우기)에 있습니다. 데이터셋 라이선스는 종료 시 사본을 모두 파기하도록 요구합니다.
