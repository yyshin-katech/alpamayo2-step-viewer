# 모델과 데이터셋 받기

> [!NOTE]
> 2026-09-28에 확인한 내용입니다. 라이선스 요약은 법률 자문이 아니고, 원문이 우선합니다. 받기 전에 원문을 직접 읽으세요.

이 레포에는 모델 가중치도 데이터셋도 들어 있지 않습니다. 둘 다 각자 자기 Hugging Face 계정으로 받습니다.

## 1. 한눈에

| 대상 | 저장소 | 라이선스 | 크기 | 접근 |
|---|---|---|---|---|
| 모델 가중치 | [nvidia/Alpamayo2-Super](https://huggingface.co/nvidia/Alpamayo2-Super) | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) | 약 72 GB (safetensors 샤드 15개) | 2026-09-28 확인 당시 동의 절차 없이 받을 수 있었습니다. 원본 README에는 gated라고 적혀 있으니, 동의 화면이 나오면 따르세요. |
| 데이터셋 | [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) | NVIDIA Autonomous Vehicle Dataset License Agreement | 이 도구는 샘플 하나(수백 MB)와 메타데이터만 받음 | 데이터셋 페이지에서 라이선스 동의 필요(gated) |
| 모델 코드 | [NVlabs/alpamayo2](https://github.com/NVlabs/alpamayo2) | Apache-2.0 | 작음 | 공개. clone 방법은 [local-run.md §2](local-run.md#2-코드-받기) |

## 2. Hugging Face 계정과 토큰

1. <https://huggingface.co> 에 가입합니다.
2. Settings → Access Tokens에서 토큰을 만듭니다. `Read` 토큰이면 됩니다. fine-grained 토큰이라면 "Read access to contents of all public gated repos you can access"를 켭니다.
3. 로그인합니다. 토큰을 붙여 넣고, "Add token as git credential?"에는 `n`으로 답해도 됩니다.

   ```bash
   .venv/bin/hf auth login
   ```

4. 확인합니다: `.venv/bin/hf auth whoami`

토큰은 `~/.cache/huggingface/token`에 저장됩니다(`HF_HOME`을 바꿨다면 그 아래). 이 파일이나 토큰 문자열을 커밋하거나 공유하지 마세요.

## 3. 데이터셋 라이선스 동의

1. 로그인한 계정 그대로 [데이터셋 페이지](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)를 엽니다.
2. 라이선스(NVIDIA Autonomous Vehicle Dataset License Agreement)를 끝까지 읽고 동의합니다. 동의하면 바로 접근할 수 있습니다. 원문 PDF는 받은 뒤 캐시에도 남습니다(`LICENSE.pdf`).
3. [§6](#6-라이선스에서-지킬-것)의 의무를 읽어 둡니다. 특히 사용 목적이 라이선스가 허용하는 범위(§1)에 들어가는지 먼저 확인하세요.

## 4. 모델 받기

```bash
.venv/bin/hf download nvidia/Alpamayo2-Super --revision 00554695e729a6ff0b6281fd2c81b18d06e33dbe
```

- `--revision`은 `walk/common.py`의 `MODEL_REVISION`과 같아야 합니다. 이 도구는 이 리비전에서 확인했고, 코드도 이 리비전 폴더를 찾습니다.
- 받는 곳: `~/.cache/huggingface/hub/models--nvidia--Alpamayo2-Super/snapshots/00554695e729a6ff0b6281fd2c81b18d06e33dbe/`. 샤드 15개(`model-000NN-of-00015.safetensors`)와 `model.safetensors.index.json`, 설정·전처리기·토크나이저 파일이 들어갑니다.
- 끊기면 같은 명령을 다시 실행하면 됩니다. 받은 파일은 건너뛰고 이어 받습니다.
- 다른 디스크에 두려면 아래 둘 중 하나를 씁니다. 어느 쪽이든 O_DIRECT를 지원하는 로컬 디스크(ext4, xfs 등, NVMe 권장)여야 하고, 이 레포 폴더 안은 피합니다.
  - 캐시 위치를 바꿉니다. 받는 셸과 실행하는 셸 모두에서 `export HF_HUB_CACHE=/data/hf-hub`처럼 같은 값을 줍니다. `HF_HOME`을 바꾸면 토큰 위치도 함께 바뀝니다.
  - 폴더에 바로 받습니다: `.venv/bin/hf download nvidia/Alpamayo2-Super --revision 00554695e729a6ff0b6281fd2c81b18d06e33dbe --local-dir /data/Alpamayo2-Super`. 그다음 실행 전에 `export ALPAMAYO2_SUPER_SNAPSHOT=/data/Alpamayo2-Super`를 줍니다.
- 확인:

  ```bash
  .venv/bin/python -c "import sys; sys.path.insert(0, 'walk'); from common import check_snapshot; s = check_snapshot(); print(s, len(list(s.glob('*.safetensors'))), 'shards')"
  ```

  경로와 `15 shards`가 나오면 됩니다. 빠진 파일이 있으면 무엇이 빠졌는지와 받는 명령을 알려 줍니다.

## 5. 데이터셋 샘플 받기

```bash
.venv/bin/python walk/fetch_data.py
```

- 노트북 샘플 0(clip `030c760c-ae38-49aa-9ad8-f5650a545d26`, t0 = 5.1 s) 하나만 받습니다. 데이터셋 전체를 받지 않습니다.
- `walk/out/data/`에 `source.pt`, `sample0.pt`, `sample0_keys.json`(합쳐 약 300 MB)을 씁니다.
- Hugging Face 캐시(`…/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/`)에는 메타데이터와 라이선스 PDF가 남습니다(합쳐 약 33 MB).
- 로더(`physical_ai_av` 0.2.2)는 데이터셋의 최신(main) 리비전을 읽습니다. 이 도구를 만들 때 main은 `33f9bf447ed3bcb7d545ce13f4226f824214fafb`이었습니다. 데이터셋이 갱신되면 샘플이 달라질 수 있습니다.
- `401`, `403`, `GatedRepoError`가 나면 §2의 로그인과 §3의 동의를 확인합니다. 셸에 `HF_HUB_OFFLINE=1`이 남아 있으면 받지 못하니 `unset HF_HUB_OFFLINE`을 합니다.

## 6. 라이선스에서 지킬 것

### 모델 가중치: OpenMDW-1.1

허용적인 라이선스이고, 출력물 사용에 제약을 두지 않습니다. 가중치를 다시 배포할 때는 라이선스와 고지를 함께 둬야 합니다. 이 레포는 가중치를 배포하지 않습니다. 원문: <https://openmdw.ai/license/1-1/>

### 데이터셋: NVIDIA Autonomous Vehicle Dataset License Agreement (요지)

| 조항 | 요지 |
|---|---|
| §1 | 권리는 "NVIDIA 기술을 사용하는 자율주행차·자동 주행보조 시스템의 **내부 개발**" 용도로만 주어집니다. |
| §3 | 데이터셋 자체와 그 출력(output), 데이터셋에 관한 벤치마크·경쟁 분석·회귀·성능 데이터를 **기밀정보**로 정의하고, 권한 있는 사용자 외의 제3자에게 공개하는 것을 금지합니다. |
| §4.1 | 개인·집단을 비윤리적으로 감시하는 데, 또는 법 집행(교통 법규 포함)에 쓰는 것을 금지합니다. |
| §4.4, §4.5 | 개인을 식별·프로파일링하거나(번호판 포함) 비식별 처리를 되돌리려는 시도, 인종·성별·나이·건강 같은 민감한 속성 추론, 생체정보 처리를 금지합니다. |
| §4.6 | 데이터셋을 **일부라도** 배포·판매·대여·재라이선스·양도·임베드·호스팅하거나 남에게 제공하는 것, 그리고 데이터셋의 파생물(derivative works)을 만드는 것을 금지합니다. |
| §4.8 | 사본을 어디에 두었는지 추적해야 합니다. |
| §8 | 처음 받은 날로부터 12개월 뒤 만료됩니다. 종료되면 사용을 멈추고 사본을 모두 파기해야 합니다. NVIDIA가 서면으로 요청할 때도 지워야 합니다(§4.9). |

이 도구가 만드는 캡처(카메라 프레임, 중간 텐서, 궤적)가 위 조항에 어떻게 해당하는지는 원문을 읽고 스스로 판단하세요. 적어도 자기 컴퓨터 밖으로 내보내지는 마세요.

### 이 도구에서 해당하는 것

다음은 모두 데이터셋에서 나온 것으로 보고 자기 컴퓨터 안에만 둡니다.

- `walk/out/` 전체: `data/`(샘플), `raw/`(캡처 텐서), `derived/`(분석 결과, 프레임 이미지 포함), `result/`(궤적 그림·JSON), `tiny/`·`tiny_raw/`(점검용), 셀프테스트 결과 JSON
- 뷰어 화면: 스크린샷, 녹화, 화면 공유
- CoT 문장, minADE·minFDE 같은 지표, 실행 시간 같은 성능 수치

그래서 이렇게 합니다.

- 커밋, 업로드, 게시, 메신저 공유를 하지 않습니다. `.gitignore`가 `walk/out/`과 이미지·로그를 막아 두었지만 `git add -f`로는 들어갈 수 있으니 조심합니다.
- 뷰어 서버를 127.0.0.1 밖으로 열지 않습니다. 바인드 주소를 바꾸거나 포트 포워딩, 터널, 리버스 프록시를 쓰지 않습니다.
- 이슈나 질문에는 오류 메시지와 환경(OS, GPU, 드라이버, 패키지 버전)만 적습니다. 데이터, 캡처, 스크린샷, 결과 수치는 붙이지 않습니다.
- 첫 다운로드 날짜를 적어 둡니다(12개월 만료).

## 7. 지우기

```bash
rm -rf walk/out                                                                   # 샘플·캡처·분석·결과
rm -rf ~/.cache/huggingface/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles  # 데이터셋 메타데이터·라이선스 PDF
rm -rf ~/.cache/huggingface/hub/models--nvidia--Alpamayo2-Super                   # (선택) 모델 72 GB
rm -rf ~/.cache/huggingface/xet                                                   # (선택) 전송 캐시. 다른 저장소의 캐시도 함께 지워짐
.venv/bin/hf auth logout                                                          # 토큰 삭제
```

- xet 전송 캐시에도 받은 파일의 조각이 남을 수 있습니다. 데이터셋 사본을 모두 파기해야 할 때는 이것도 지웁니다.
- `HF_HOME`, `HF_HUB_CACHE`, `--local-dir`로 위치를 바꿨다면 그 경로에서 지웁니다.
- `walk/out/`을 다른 곳에 복사해 두었다면 그 사본도 지웁니다(§4.8, §8).
