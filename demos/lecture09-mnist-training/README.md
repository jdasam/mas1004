# 데모: MNIST Training (손글씨 숫자 분류기가 학습되는 동안 신경망이 바뀌는 모습)

7차시와 같은 데이터(학습 10,000장, 시험 2,000장)와 같은 모델(784 → 64 ReLU → 10 softmax, 파라미터 50,890개)을 브라우저에서 실제로 학습하면서 네 가지를 함께 보여 준다.
MAS1004 9차시(2026-10-06) 수업 중 프로젝터용. 설계: `docs/superpowers/specs/2026-10-06-lecture09-mnist-training-design.md`.

- 은닉 뉴런 64개의 가중치(뉴런마다 784개)를 28 × 28 타일로. 빨강 양수, 파랑 음수, ±0.08에서 색이 가득 찬다.
- 은닉층 지도: 시험 이미지 앞 500장의 은닉값 64개를 PCA 두 방향에 그린다. 점은 정답 숫자 글자.
- 시험 이미지 10장(숫자마다 시험 세트에서 처음 나오는 것)의 1등 추측과 확률 막대 10개. 검은 막대가 정답, 빨간 막대는 틀린 1등.
- 손실과 정확도 곡선: 학습 이미지 앞 2,000장과 시험 2,000장으로 잰다. 가로축은 epoch.

## 여는 법

- `index.html`과 `mnist-data.js`를 같은 폴더에 두고 `index.html`을 브라우저로 연다(`file://`도 된다). 둘 중 하나만 있으면 동작하지 않는다.
- GitHub Pages: `mas1004-2026/tools/publish_demo.sh lecture09-mnist-training` → https://jdasam.github.io/mas1004/demos/lecture09-mnist-training/

## 조작

- Play/Pause(`Space`), +1 step(`→`), +10 steps, Reset(같은 시드로 처음부터, 매번 같은 학습).
- Steps/s: 1, 5(기본), 20, 100, Max(이 컴퓨터에서 초당 약 290걸음).
- Learning rate: 0.01, 0.1(기본), 1. 바꾸면 다음 걸음부터 적용된다. 비교하려면 Reset 뒤에 바꾼다.
- URL 옵션: `?lr=1`, `?speed=20`(또는 `max`), `?play=1`, `?seed=2`.

## 학습

배치 32의 확률적 경사 하강법. 1 epoch = 313걸음(마지막 배치 16장), epoch마다 시드 고정으로 다시 섞는다.
초기값은 W1이 표준편차 0.01(처음 타일이 거의 흰색이도록 작게), W2가 표준편차 1/8, 편향 0.
곡선 점은 2걸음(20걸음까지), 5걸음(100까지), 20걸음(400까지), 그 뒤 50걸음마다 찍는다.

## 실측 (`node test/measure.js 3`, 시드 1)

| 학습률 | 10걸음 | 50걸음 | 100걸음 | 1 epoch | 3 epoch |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 25.3% | 55.7% | 73.6% | 82.5% | 89.8% |
| 0.1 | 57.1% | 81.7% | 88.3% | 91.0% | 93.4% |
| 1 | 16.8% | 40.5% | 49.1% | 56.5% | 81.3% |

시험 정확도(2,000장). 학습률 1에서는 손실이 오르내리고 가중치 타일 여럿이 파랗게 가득 차며 은닉층 지도가 몇 줄로 눌린다.
처음 몇 epoch 동안은 학습과 시험 곡선이 거의 겹친다. 7차시의 99% 대 95% 차이는 30 epoch을 돈 뒤의 것이다.

## 구조와 테스트

- `index.html`: `<script id="mt-core">`(전역 `MT`, DOM 없음: 데이터 분할, 모델, 역전파, SGD, PCA)와 `<script id="mt-ui">`(화면, 학습 루프).
- `mnist-data.js`: `mas1004-2026/tools/make_mnist_demo_data.py`가 `mas1004/data/mnist_small.npz`에서 만든다(픽셀 0~255 정수, gzip, base64, 2.6MB). 브라우저에서 `DecompressionStream`으로 푼다.
- `test/`(게시에서 빠진다):

```bash
node --test test/core.test.js                                  # 역전파와 중앙 차분 대조 등 6개
node test/measure.js 3                                         # 위 실측 표
uv run --no-project --with playwright python test/smoke.py     # 브라우저 캡처, test/shots/
```
