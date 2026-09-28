# alpamayo2-step-viewer

NVIDIA **Alpamayo 2 Super**가 주행 장면 하나를 처리하는 과정을, 자기 GPU에서 직접 돌려 캡처한 값으로 한 단계씩 따라가 보는 로컬 도구입니다.

- **`walk/`** 는 모델을 한 번 추론하면서 단계별 중간 텐서를 safetensors로 저장합니다. 72 GB 체크포인트를 레이어 단위로 디스크에서 읽어 올리므로 GPU 메모리 10 GB로 돌아갑니다. 가중치는 체크포인트 바이트를 그대로 올리고, 스트리밍·캡처가 계산을 바꾸지 않는지는 작은 무작위 모델로 비트 단위 비교합니다(`walk/test_tiny.py`).
- **`viewer/`** 는 저장한 텐서를 브라우저에서 한 단계씩 보여 줍니다. 카메라 패치 → 비전 인코더 → LLM 64층 프리필 → CoT 디코딩 → 행동 전문가의 플로 매칭 → 예측 궤적 순서입니다. 일반·세부 모드, 값을 누르면 저장된 비트와 이웃 값을 보여 주는 인스펙터, 층을 가로지르는 분석 서랍(11탭)이 있습니다.

> **English summary.** A local, step-by-step viewer for NVIDIA Alpamayo 2 Super. You download the weights and one PhysicalAI-AV sample yourself, run the model once on your own GPU (10 GB is enough; the weights are streamed layer by layer), capture every intermediate, and browse them at `127.0.0.1`. This repository ships **code only**: no weights, no dataset files, no captures, screenshots, logs or results. The captures come from a dataset whose license forbids sharing it, so keep them on your own machine. The guides in `docs/` are in Korean.

## 들어 있는 것과 없는 것

| 이 레포에 있음 | 이 레포에 없음 (직접 받거나 직접 만듦) |
|---|---|
| 캡처·분석 스크립트 (`walk/`) | 모델 가중치 `nvidia/Alpamayo2-Super` (약 72 GB) |
| 뷰어 (`viewer/`: HTML·CSS·JS, 로컬 서버, 헤드리스 셀프테스트 드라이버) | 데이터셋 `nvidia/PhysicalAI-Autonomous-Vehicles`의 샘플 |
| 버전을 고정한 의존성 목록 (`requirements.lock.txt`) | 모델 코드 `NVlabs/alpamayo2` (고정 커밋으로 clone) |
| 실행 가이드, 모델·데이터 받기 가이드 (`docs/`) | 캡처(`walk/out/`), 스크린샷, 로그, 실행 결과 수치 |

> [!WARNING]
> **캡처와 화면은 자기 컴퓨터에서만 보세요.** `walk/out/` 아래 모든 파일과 뷰어 화면(스크린샷, 녹화, 화면 공유)은 PhysicalAI-AV 데이터셋에서 나온 것입니다. 이 데이터셋의 라이선스는 데이터셋을 일부라도 배포·호스팅하는 것을 금지하고, 데이터셋 관련 결과·성능 수치를 기밀정보로 봅니다. 커밋·게시·공유하지 말고, 뷰어 서버를 127.0.0.1 밖으로 열지 마세요. 자세한 내용은 [docs/model-and-data.md](docs/model-and-data.md)에 있습니다.

## 요구 사항 (요약)

- Linux x86_64 (Ubuntu 22.04에서 확인)
- NVIDIA GPU: 메모리 10 GB 이상, bf16 지원(Ampere 이상 권장). CUDA 12.8을 지원하는 드라이버
- RAM 32 GB 권장, 빈 디스크 약 120 GB (O_DIRECT를 지원하는 로컬 디스크, NVMe 권장)
- Hugging Face 계정 (데이터셋 라이선스 동의 필요)
- Python 3.12는 uv가 받아 줍니다. 시스템 Python·전역 패키지는 건드리지 않습니다.

자세한 표와 이유는 [docs/local-run.md](docs/local-run.md#1-요구-사항)에 있습니다.

## 빠른 시작

```bash
# 1. 코드 (이 레포 + 고정 커밋의 원본 모델 코드)
git clone https://github.com/yyshin-katech/alpamayo2-step-viewer.git
cd alpamayo2-step-viewer
git clone https://github.com/NVlabs/alpamayo2.git
git -C alpamayo2 checkout 5e7975f4a2100ee8ac1a62b79239bbabefddcbef

# 2. Python 환경 (uv가 없으면 먼저: curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$PWD/tools" sh)
source env.sh
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.lock.txt
uv pip install --python .venv --no-deps -e ./alpamayo2

# 3. 로그인 → 데이터셋 페이지에서 라이선스 동의 → 모델 받기 (docs/model-and-data.md)
.venv/bin/hf auth login
.venv/bin/hf download nvidia/Alpamayo2-Super --revision 00554695e729a6ff0b6281fd2c81b18d06e33dbe

# 4. 샘플 받기 → 캡처 → 분석 → 뷰어
.venv/bin/python walk/fetch_data.py
.venv/bin/python walk/run.py
HF_HUB_OFFLINE=1 .venv/bin/python walk/analyze.py
python3 viewer/serve.py        # 브라우저에서 http://127.0.0.1:8765/viewer/index.html
```

`run.py`는 디스크에서 가중치를 여러 번 다시 읽으므로 실행 시간은 거의 저장장치 속도가 정합니다. 단계마다 무엇을 하고 무엇을 확인하는지는 [docs/local-run.md](docs/local-run.md)에 적었습니다.

## 문서

- [docs/local-run.md](docs/local-run.md): 요구 사항, 설치, 단계별 실행, 선택 점검·셀프테스트, 문제 해결
- [docs/model-and-data.md](docs/model-and-data.md): Hugging Face 계정·토큰, 모델·데이터셋 받기, 라이선스에서 지킬 것, 지우기

## 폴더 구성

```text
walk/
  common.py          경로, 샘플·모델 리비전 고정값, 스냅샷 확인
  fetch_data.py      노트북 샘플 0 받기 (온라인)
  build.py           meta 장치에 모델 구성 (from_pretrained 단계 재현)
  stream.py          레이어 단위 가중치 스트리밍 (O_DIRECT + page-locked 메모리)
  capture.py         중간값 기록 훅 (모델 계산은 건드리지 않음)
  run.py             캡처 실행 → walk/out/raw, walk/out/result
  analyze.py         파생 분석 → walk/out/derived
  inspect_prompt.py  (선택) 프롬프트 토큰 배치 보기
  test_tiny.py       (선택) 작은 무작위 모델로 스트리밍·캡처 비트 일치 점검
  test_stream.py     (선택) 실제 스냅샷으로 스트리밍 로더 점검
viewer/
  index.html  css/  js/   뷰어 (외부 리소스 없음)
  serve.py                로컬 서버 (표준 라이브러리만, 127.0.0.1 전용)
  tools/cdp.mjs           (선택) 헤드리스 Chrome 셀프테스트 드라이버
docs/                     가이드
env.sh                    uv의 Python·캐시를 레포 안 tools/에 두는 환경변수
requirements.lock.txt     고정 의존성 (Python 3.12, Linux x86_64, torch 2.8.0 + CUDA 12.8)
```

직접 만드는 폴더 `alpamayo2/`(원본 clone), `.venv/`, `tools/`(uv), `walk/out/`(샘플·캡처)은 `.gitignore`에 들어 있습니다.

## 라이선스

- 이 레포의 코드와 문서: Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE))
- 모델 가중치(OpenMDW-1.1)와 데이터셋(NVIDIA Autonomous Vehicle Dataset License)은 이 레포에 들어 있지 않고, 각자의 라이선스를 따릅니다.

NVIDIA와 관계없는 비공식 프로젝트이며, NVIDIA의 보증을 받지 않았습니다.
