# 데모: Names RNN (이름의 다음 문자를 예측하는 RNN)

미국 이름으로 학습한 문자 단위 RNN 언어 모델을 브라우저에서 ONNX Runtime Web으로 돌린다. 모델은 다음 문자 27개(이름의 끝 표시 `.`과 a~z) 각각에 확률을 주고, 데모는 그중 하나를 뽑아 붙이는 일을 되풀이해 이름을 만든다.
MAS1004 13차시(`mas1004-2026/lecture13_language_models.md`)에서 n-gram 데모(`../lecture13-ngram/`) 다음에 쓴다. 데이터와 모델(`data.js`, `model/`)은 `mas1004-2026/tools/make_lecture13_names_rnn.py`가 만들었고, 이 폴더에서는 다시 만들지 않는다. 화면은 영어, 이 문서는 교수용이다.

## 수업에서 보이려는 것

1. 언어 모델은 가능한 다음 문자 모두에 확률을 준다. 생성은 하나를 뽑아 붙이고 다시 확률을 구하는 일의 반복이다. (Generate)
2. 학습은 실제로 다음에 온 문자의 확률을 높인다. 같은 앞부분 뒤에 여러 문자가 올 수 있으므로 학습된 모델의 확률은 데이터의 빈도에 가까워진다. 12,000걸음 모델은 이름의 첫 문자로 a에 확률 0.139를 주고, 학습 이름 28,829개 중 a로 시작하는 이름의 비율은 0.138(3,973개)이다. (Training의 첫 문자 그림, Generate의 `Compare with training names`)
3. 모델은 규칙을 들은 적이 없다. 학습 걸음이 늘면 무작위 글자열이 이름 같은 글자열로 바뀌고, 만든 이름의 대부분은 데이터에 없는 새 이름이다. (Training의 이름 표, Generate의 이름 목록)
4. 모델은 데이터(미국 이름)를 닮는다. dasaem 같은 한국 이름은 몇 문자에서 확률이 낮다(−log p가 크다). 학생이 자기 이름을 넣어 본다. (Score a name)

## 화면

맨 위 줄: 화면 고르기(`Generate | Score a name | Training`), `Model after N training steps`(0, 30, 100, 300, 1,000, 3,000, 12,000 중 하나, 기본 12,000), 그 모델의 검증 손실(검증 이름 3,204개에서 문자당 평균 −log p). 세 화면 모두 여기서 고른 모델을 쓴다. 확률은 n-gram 데모와 덱처럼 소수로 쓴다(0.01 이상은 소수 셋째 자리, 그보다 작으면 유효숫자 두 자리: 0.421, 0.048, 0.0094). 맨 아래 줄에 이름 데이터의 출처와 RNN 한 걸음에 걸린 시간이 있다.

### Generate

- 맨 위에 지금까지 만든 이름을 `.`부터 토큰으로 쓴다(`. e m m _`). 문자 아래 숫자는 그 문자가 붙을 때의 확률이다(지금 고른 모델과 T로 다시 계산한다). 마지막에 붙은 문자는 빨간색이다. 그 아래 문장: "The model has read . e m m. Below: its probability for each possible next character."
- 버튼: `Sample next`(뽑기), `Most likely next`(확률이 가장 큰 문자), `Finish the name`(`.`이 나올 때까지 뽑기, 문자마다 0.22초씩 보인다), `Undo`, `Start over`(다음 번호의 이름을 새로 시작). 막대를 클릭하면 그 문자를 붙인다. 키보드도 된다: a~z와 `.`은 그 문자를 붙이고, Backspace는 Undo, Enter는 Sample next.
- `.`을 고르면 이름이 끝난다. "Finished: zaylahn. '.' was chosen, so the name ends here." 다음에 그 이름이 학습 이름에 있는지, 검증 이름에만 있는지, 새 이름인지 쓴다. 막대는 회색으로 바뀌고 고른 `.`만 빨간색으로 남으며, 뽑는 버튼은 꺼진다.
- 막대 27개: 다음 문자의 확률. 순서는 `.`, a~z로 고정해 걸음마다 막대 자리가 바뀌지 않는다. `Sort by probability`를 켜면 큰 순서다. 큰 다섯 개(0.005 이상)에 숫자를 쓴다. `Show scores`를 켜면 softmax 전 점수를 문자 아래에 쓴다. 막대에 마우스를 올리면 확률, 점수, score / T가 나온다.
- `Compare with training names`: 지금 앞부분으로 시작하는 학습 이름에서 다음 문자별 비율을 회색 막대로 모델 막대 옆에 그린다. `.`은 이름이 거기서 끝난 비율이다. `.`뿐일 때는 첫 문자 비율이다. 그런 학습 이름이 없으면(예: dasae) 셀 것이 없고 모델은 그래도 모든 문자에 확률을 준다고 쓴다.
- 막대 아래 띠: 27개 확률을 `.`, a~z 순서로 0부터 1까지 이어 붙이고, 이 자리의 난수를 세로선으로 표시한다. `Sample next`는 난수가 떨어진 칸의 문자를 붙인다("The random number for this position, 0.434, falls in 'e', so Sample next adds 'e'."). 누르면 고른 막대가 0.24초 동안 빨간색이 된 뒤 문자가 붙는다.
- Temperature T: 0.2~2.0(기본 1.0), "Probabilities = softmax(scores / T)". 막대, 띠, 뽑기, 오른쪽 목록이 모두 이 T의 확률을 쓴다.
- Seed: 고치면 이름 번호가 1로 돌아가고 만들던 이름이 지워진다. 옆에 지금 이름 번호("name 3")가 있다.
- `Hidden values`: 문자마다 그 문자를 읽은 뒤의 은닉값 h 128개를 16줄 × 8칸으로 그린다(0보다 크면 빨강, 작으면 파랑, ±1에서 가장 진하다). 다음 걸음으로 넘어가는 것은 이 128개뿐이다.
- 오른쪽 `20 names at a time`: `Generate 20 names`(두 번째부터 `Generate 20 more`)를 누를 때마다 이름을 20개씩 더 만든다(1~20번, 21~40번, ...). 이름마다 `in training names`, `in validation names (not trained on)`, `new` 중 하나를 붙인다. 위의 "New names: 13 of 20 (65%)"는 지금까지 만든 이름 전체의 비율이고, 이름이 하나씩 나올 때마다 바뀐다. 목록에는 마지막 20개만 보인다. 모델, T, 시드를 바꾸면 같은 번호의 이름을 다시 만든다. `Clear`로 비운다.

### Score a name

- 이름을 친다. 대문자는 소문자로 바꾸고 a~z가 아닌 문자는 빼며, 뺀 문자를 적는다("Ignored: '-', '!'. The model only knows the letters a to z."). 30자까지. 프리셋: emma, olivia, jennifer, christiana, dasaem, minji, seoyeon, zzyzx.
- 이름이 학습 이름에 있는지, 검증 이름에만 있는지(학습하지 않은 이름), 둘 다 없는지 쓴다.
- 표의 열: Position, The model has seen(`. d a s`), Next character, RNN probability(막대 포함), −log p. 마지막 줄의 다음 문자는 끝 표시 `.`이다. RNN 확률이 가장 낮은 줄은 빨간색이고 "least likely"가 붙는다.
- `Compare with counting (previous 2 letters)`를 켜면 Counting (previous 2 letters), −log p 두 열이 붙는다. 기본은 꺼짐이다(13차시 덱에서 n-gram 부를 빼서 수업에서는 세기를 다루지 않는다).
- Counting 열: 학습 이름 28,829개에서 앞 두 문자 뒤에 이 문자가 온 횟수 / 앞 두 문자가 나온 횟수. 이름 앞은 `..`으로 채운다(첫 문자는 `..` 뒤, 둘째 문자는 `.d` 뒤). 칸 아래에 "1,526 of 28,829 after '..'"처럼 횟수를 쓴다. 0이면 "0 (never seen)"과 ∞.
- 표 아래 왼쪽: RNN이 이 이름에 준 평균 −log p(이 이름의 교차 엔트로피 손실), 이 모델의 검증 이름 평균, 확률이 가장 낮은 문자. 오른쪽(세기를 켰을 때): 세기의 평균 −log p(한 문자라도 0이면 ∞)와 검증 이름에서의 세기 결과(검증 이름의 22,866문자 중 135문자가 같은 앞 두 문자 뒤에 나온 적이 없어 ∞, 나머지 문자의 평균 2.191).
- 이 화면은 T를 쓰지 않는다(T = 1, 모델의 확률 그대로).

### Training

- 왼쪽: 학습 중 손실. 회색 선은 50걸음마다 기록한 배치(이름 64개)의 손실, 채운 점은 체크포인트의 학습 손실(학습 이름 28,829개 전체), 빈 원은 검증 손실이다. 가로축은 `Log`(기본, 0걸음도 그리려고 log10(걸음 + 10)) 또는 `Linear`. 점이나 그 근처, 아래 표의 열을 클릭하면 그 모델로 바뀐다. 고른 모델은 빨간 세로선이다. 아래 표에 걸음마다 학습 손실과 검증 손실이 있다.
- 오른쪽: 이름의 첫 문자. `.` 다음에 모델이 주는 확률(검정)과 학습 이름 중 그 문자로 시작하는 비율(회색). 세로축을 0~0.15로 고정해서 걸음을 바꾸면 검정 막대만 움직인다. 위 문장이 a의 두 값을 쓴다("After the start '.', the model after 12,000 steps gives a the probability 0.139. Of the 28,829 training names, 3,973 start with a: a share of 0.138."). 막대에 마우스를 올리면 두 값과 이름 수가 나온다.
- 아래: 체크포인트 일곱 개 각각에서 만든 이름 1~12번(지금 시드, T = 1). 같은 번호의 이름은 같은 난수를 쓰므로 열 사이의 차이는 모델의 차이에서만 나온다. 고른 걸음의 열만 진하게 쓰고 이름마다 표시(new, training, validation, 30자까지 끝나지 않았으면 no end)를 붙이며, 표 아래에 새 이름 수를 쓴다. 열 머리를 클릭하면 그 모델로 바뀐다.

## 정한 것

사용자에게 묻지 못하고 정한 것과 이유다. 바꾸려면 말해 달라.

- 난수와 뽑기. 시드와 이름 번호로 이름마다 난수 31개를 정하고(3, 4차시 데모와 같은 xmur3 + mulberry32), 이름의 k번째 문자에 k번째 난수를 쓴다. 뽑기는 확률을 `.`, a~z 순서로 0부터 1까지 이어 붙이고 난수가 떨어진 칸을 고른다. 그래서 같은 시드, 모델, T면 앞에서 무엇을 눌렀든 같은 이름이 나오고, 모델이나 T만 바꾸면 같은 난수로 다른 이름이 나와 비교할 수 있다.
  - Generate에서 만드는 n번째 이름(`Start over`마다 번호가 하나 오른다), 오른쪽 목록의 n번째 이름, Training 표의 n번째 줄(T = 1)이 같은 난수를 쓴다. 시드 1에서 `Finish the name`을 처음 누르면 목록 1번과 같은 zaylahn이 나온다.
  - `Undo` 뒤에 `Sample next`를 누르면 같은 문자가 다시 나온다(같은 자리, 같은 난수). 다른 문자를 보려면 막대를 누르거나 시드를 바꾼다. n-gram 데모는 난수를 차례로 쓰므로 Undo 뒤에 다른 토큰이 나올 수 있어, 두 데모의 동작이 다르다.
- 띠에 다음 난수를 미리 보인다. T를 움직이면 칸 너비가 바뀌어 같은 난수가 다른 문자에 떨어지는 것이 보인다(아래 예시의 `. e m m`).
- 이름은 30자에서 멈춘다. 학습 스크립트의 `sample`과 같다. 0걸음 모델에서는 자주 일어난다(시드 1의 12개 중 2개). 목록에는 "no '.' after 30 letters", Training 표에는 "no end"로 쓰고 새 이름으로 센다. 직접 만드는 이름도 30자에서 멈춘다.
- 이름의 표시는 학습 스크립트의 통계(in_train, in_val_only, new)와 같은 규칙이다. names.txt에는 같은 이름이 여러 줄 있어(32,033줄, 서로 다른 이름 29,494개) 검증 이름 3,204줄 중 433줄은 학습 이름에도 있다. 이런 이름은 학습 이름으로 표시하고, Score a name에서는 "It is also in the validation names: names.txt lists some names twice."를 덧붙인다. zzyzx도 학습 이름이다. 첫 문자로 `.`이 나온 빈 이름은 따로 센다.
- 새 이름 비율은 누를 때마다 20개씩 더 만들어 지금까지 만든 이름 전체로 낸다(최대 1,000개, `list=1000`으로 바로 열 수 있다). 같은 시드의 같은 번호는 늘 같은 이름이라, 같은 20개를 다시 보이는 대신 다음 20개를 만들게 했다.
- 세기 열은 학습 이름 28,829개로 센다(검증 이름과 견주려고). n-gram 데모는 32,033개 전체로 세므로 같은 이름이라도 횟수와 확률이 조금 다르다(minjun의 첫 m이 이 데모에서는 2,257 / 28,829, n-gram 데모에서는 2,538 / 32,033). 앞 두 문자가 학습 이름에 한 번도 없으면(분모 0) 확률 0, −log p ∞로 쓰고 "'xq' is never followed by anything"처럼 적는다. 프리셋 이름에는 이런 경우가 없다.
- 확률은 n-gram 데모, 덱과 같이 소수로 쓴다. 0.01 이상은 소수 셋째 자리(0.421), 그보다 작으면 유효숫자 두 자리(0.0094)이고, n-gram 데모의 `fmtP`와 같은 함수다. 그래프 눈금도 소수(0, 0.05, 0.10)다. 생성한 이름 중 새 이름의 비율("New names: 13 of 20 (65%)")은 확률이 아니라 개수의 비율이라 백분율로 두었다(n-gram 데모의 "24% of them had only one possible next token"과 같다).
- 표의 첫 열 이름은 Step 대신 Position이다. 맨 위 줄의 training steps와 헷갈리지 않게 했다.
- 프리셋에 christiana와 seoyeon을 더했다. christiana는 검증 이름에만 있어 학습하지 않은 이름인데 평균 −log p가 1.00으로 검증 평균 2.016보다 낮다. seoyeon은 한국 이름이고 seo 뒤의 y가 0.00099, 세기는 ∞다.
- Training의 이름 표는 고른 걸음 하나가 아니라 일곱 걸음을 나란히 보이고 고른 열만 진하게 쓴다.
- Generate에 `Compare with training names`(처음에는 꺼짐)를 더했다. 첫 문자뿐 아니라 어느 앞부분에서나 모델 확률과 데이터 빈도를 견줄 수 있다. 덱의 joh 장(학습 이름 42개 중 n 29개, a 13개)을 데모에서 그대로 보일 수 있다.
- 강조색은 n-gram 데모, 과목 테마와 같은 진한 빨강(#b3001b) 하나다(마지막 문자, 고른 막대, 띠에서 난수가 떨어진 칸, 확률이 가장 낮은 줄, ∞, new, 고른 걸음). 모델 확률은 검정, 학습 이름 비율은 회색이다. 은닉값은 빨강과 파랑으로 그린다(11차시 데모의 CAM과 같은 색). 아이콘과 회색 카드는 쓰지 않았다.
- 로그 축은 0걸음을 그리려고 log10(걸음 + 10)을 쓴다. 그래서 0과 10 사이가 실제 로그 눈금보다 좁다.

## 수업에서 보일 예시 (교수용)

아래 값은 모두 데모 화면에 나오는 값이다. `test/smoke.py`가 같은 값을 `test/examples.json`에 두고 돌릴 때마다 대조한다. 데모를 고쳐 값이 바뀌면 테스트가 실패하므로, 의도한 변경이면 `--update`로 다시 쓰고 이 절도 고친다. 주소는 게시한 뒤의 주소이고, 로컬에서는 `index.html` 뒤에 같은 `?...`를 붙인다.

덱의 「Names During Training」 표, 「Sampling and Temperature」의 이름 두 줄, 「New Names」의 숫자와 이름은 아래 값(데모, 시드 1)을 그대로 쓴다. 학습 스크립트가 PyTorch 난수로 뽑은 표본(`assets/lecture13/names-rnn.json`)은 데모 화면에 나오지 않아 덱에 쓰지 않는다. 확률은 같은 모델이라 덱과 데모가 같다. 데모는 소수 셋째 자리까지 쓴다: 첫 문자 a 0.139, joh 뒤 n 0.377과 a 0.228(덱 0.38, 0.23), ma 뒤 r 0.301과 T = 0.5의 0.697(덱 0.30, 0.70), emma와 dasaem의 문자별 확률.

### 이름 하나 만들기 (12,000걸음, 시드 1)

- [`?view=generate&step=12000&t=1.0&seed=1`](https://jdasam.github.io/mas1004/demos/lecture13-names-rnn/?view=generate&step=12000&t=1.0&seed=1)에서 `Finish the name`: zaylahn(새 이름). 문자가 붙을 때의 확률은 z 0.029, a 0.514, y 0.157, l 0.356, a 0.284, h 0.291, n 0.036, 끝 `.` 0.234다. 첫 자리 난수가 0.990이라 띠의 오른쪽 끝, 확률 0.029인 z에 떨어진다.
- 이어서 `Start over`를 누르고 `Most likely next`만 누르면 alexandra(학습 이름)가 된다. 뽑지 않으므로 시드와 상관없이 늘 같다.
- 앞부분 세 개를 T마다 본다. 이름 1번의 그 자리 난수가 어느 문자에 떨어지는지도 적었다. `Compare with training names`를 켜면 그 앞부분으로 시작하는 학습 이름의 비율이 회색으로 나온다.

[`. e m m`](https://jdasam.github.io/mas1004/demos/lecture13-names-rnn/?view=generate&step=12000&t=1.0&seed=1&prefix=emm): emm으로 시작하는 학습 이름 75개 뒤에는 a 0.533, e 0.253, i 0.133, o, r, y 각 0.027.

| T | a | e | i | o | y | 난수 0.434가 떨어지는 문자 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.602 | 0.285 | 0.101 | 0.0052 | 0.0049 | a |
| 1.0 | 0.421 | 0.289 | 0.172 | 0.039 | 0.038 | e |
| 1.5 | 0.321 | 0.250 | 0.177 | 0.066 | 0.064 | e |

[`. j o h`](https://jdasam.github.io/mas1004/demos/lecture13-names-rnn/?view=generate&step=12000&t=1.0&seed=1&prefix=joh&data=1): joh로 시작하는 학습 이름 42개 뒤에는 n 0.690(29개), a 0.310(13개). 모델은 n과 a 말고도 i, l, e 등에 확률을 나눠 준다.

| T | n | a | i | l | e | 난수 0.434가 떨어지는 문자 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.676 | 0.247 | 0.021 | 0.019 | 0.013 | n |
| 1.0 | 0.377 | 0.228 | 0.066 | 0.063 | 0.053 | l |
| 1.5 | 0.235 | 0.168 | 0.073 | 0.071 | 0.063 | k |

[`. m a`](https://jdasam.github.io/mas1004/demos/lecture13-names-rnn/?view=generate&step=12000&t=0.5&seed=1&prefix=ma): ma로 시작하는 학습 이름 1,296개 뒤에는 r 0.272, l 0.107, k 0.073, d 0.068, t 0.059, y 0.059.

| T | r | l | k | y | d | 난수 0.921이 떨어지는 문자 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.697 | 0.086 | 0.051 | 0.044 | 0.025 | r |
| 1.0 | 0.301 | 0.106 | 0.081 | 0.075 | 0.057 | y |
| 1.5 | 0.181 | 0.090 | 0.076 | 0.072 | 0.060 | y |

### 이름 20개씩 (12,000걸음, 시드 1)

[T = 0.5](https://jdasam.github.io/mas1004/demos/lecture13-names-rnn/?view=generate&step=12000&t=0.5&seed=1&list=20): 새 이름 5개(25%), 학습 이름 14개, 검증 이름에만 1개. 표시가 없는 것은 학습 이름이고, renna가 두 번 나온다.
zaylah(검증 이름에만), hensley, alexandra, zahar, juan, kayanna, renna, kailynn, ariel, leilon(새), renna, remington, kameri(새), sameer, belly(새), maryana, kalee, karyle(새), julianna, andrai(새)

[T = 1.0](https://jdasam.github.io/mas1004/demos/lecture13-names-rnn/?view=generate&step=12000&t=1.0&seed=1&list=20): 새 이름 13개(65%), 학습 이름 7개. 표시가 없는 것은 새 이름이다.
zaylahn, josya, alivia(학습), zadiy, kyan(학습), layanna(학습), shephia, lailynn(학습), doniel, niemi, theo(학습), shanvie, lanets, valla, eddiz, oley, maja(학습), limy, kymbert, chas(학습)

[T = 1.5](https://jdasam.github.io/mas1004/demos/lecture13-names-rnn/?view=generate&step=12000&t=1.5&seed=1&list=20): 새 이름 16개(80%), 학습 이름 4개. 표시가 없는 것은 새 이름이다.
zayleny, kwkyn, blisob, zachy, kyaq, may(학습), thiocl, madeyvia, evelia(학습), pkesselfa, timi, timotoeit, manish(학습), xehu, gadmyer, regwae, maha(학습), milyan, kymbert, cils

1번 이름은 같은 난수로 T에 따라 zaylah, zaylahn, zayleny가 된다.

`Generate 20 more`를 네 번 더 눌러 100개가 되면(`&list=100`으로 바로 열 수도 있다):

| T | 새 이름 | 학습 이름 | 검증 이름에만 | 빈 이름 |
| --- | --- | --- | --- | --- |
| 0.5 | 31 | 68 | 1 | 0 |
| 1.0 | 69 | 29 | 1 | 1 |
| 1.5 | 87 | 11 | 1 | 1 |

T = 1에서 1,000개([`&list=1000`](https://jdasam.github.io/mas1004/demos/lecture13-names-rnn/?view=generate&step=12000&t=1.0&seed=1&list=1000), 만드는 데 0.3초 이하): 새 이름 764개(76.4%), 학습 이름 218개, 검증 이름에만 15개, 빈 이름 3개. 덱 「New Names」는 이 값을 쓴다. 참고로 학습 스크립트의 PyTorch 표본 1,000개는 새 이름 762개, 학습 이름 218개, 검증 이름에만 20개였다.

### 이름 점수 (Score a name, 12,000걸음)

[dasaem](https://jdasam.github.io/mas1004/demos/lecture13-names-rnn/?view=score&step=12000&name=dasaem&count=1): 학습 이름에도 검증 이름에도 없다. 세기 열은 `count=1`(Compare with counting)일 때만 보인다.

| 위치 | 모델이 본 것 | 다음 문자 | RNN 확률 | −log p | 세기(앞 두 문자) | −log p |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `.` | `d` | 0.053 | 2.94 | 0.053 (1,526 / 28,829) | 2.94 |
| 2 | `. d` | `a` | 0.417 | 0.87 | 0.421 (642 / 1,526) | 0.87 |
| 3 | `. d a` | `s` | 0.027 | 3.61 | 0.030 (35 / 1,182) | 3.52 |
| 4 | `. d a s` | `a` | 0.048 | 3.03 | 0.056 (57 / 1,019) | 2.88 |
| 5 | `. d a s a` | `e` | 0.019 | 3.98 | 0.0092 (10 / 1,090) | 4.69 |
| 6 | `. d a s a e` | `m` | 0.0094 | 4.67 | 0.016 (10 / 624) | 4.13 |
| 7 | `. d a s a e m` | `.` | 0.903 | 0.10 | 0.168 (117 / 695) | 1.78 |

RNN 평균 2.74(이 모델의 검증 이름 평균 2.016), 세기 평균 2.97. 확률이 가장 낮은 것은 `. d a s a e` 뒤의 m(0.0094, −log p 4.67)이고, s, a, e, m이 모두 0.05 아래다. 끝의 `.`은 RNN이 0.903을 주고 세기는 0.168(em 뒤에 이름이 끝난 비율)이다.

[emma](https://jdasam.github.io/mas1004/demos/lecture13-names-rnn/?view=score&step=12000&name=emma&count=1): 학습 이름이다.

| 위치 | 모델이 본 것 | 다음 문자 | RNN 확률 | −log p | 세기(앞 두 문자) | −log p |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `.` | `e` | 0.048 | 3.03 | 0.048 (1,390 / 28,829) | 3.03 |
| 2 | `. e` | `m` | 0.179 | 1.72 | 0.186 (258 / 1,390) | 1.68 |
| 3 | `. e m` | `m` | 0.259 | 1.35 | 0.124 (86 / 695) | 2.09 |
| 4 | `. e m m` | `a` | 0.421 | 0.86 | 0.446 (66 / 148) | 0.81 |
| 5 | `. e m m a` | `.` | 0.062 | 2.78 | 0.068 (159 / 2,322) | 2.68 |

RNN 평균 1.95, 세기 평균 2.06. 확률이 가장 낮은 것은 첫 문자 e(0.048)다.

프리셋 전체(12,000걸음):

| 이름 | 데이터 | RNN 평균 −log p | 확률이 가장 낮은 문자 | 세기 평균 −log p |
| --- | --- | --- | --- | --- |
| emma | 학습 이름 | 1.95 | 첫 문자 e, 0.048 | 2.06 |
| olivia | 학습 이름 | 1.42 | 첫 문자 o, 0.012 | 2.27 |
| jennifer | 학습 이름 | 1.59 | `. j e n n i` 뒤 f, 0.027 | 2.10 |
| christiana | 검증 이름에만 | 1.00 | 첫 문자 c, 0.050 | 1.71 |
| dasaem | 없음 | 2.74 | `. d a s a e` 뒤 m, 0.0094 | 2.97 |
| minji | 없음 | 2.94 | `. m i n` 뒤 j, 0.0054 | ∞ (in 뒤의 j: 0 / 1,921) |
| seoyeon | 없음 | 3.41 | `. s e o` 뒤 y, 0.00099 | ∞ (eo 뒤의 y, ye 뒤의 o) |
| zzyzx | 학습 이름 | 5.17 | `. z z y z` 뒤 x, 0.0000051 | 3.55 |

zzyzx는 학습 이름인데도 RNN은 zzyz 뒤의 x에 0.0000051(−log p 12.19)을 준다. 세기는 학습 이름의 zzyzx 한 번을 그대로 세므로 손실이 RNN보다 낮다.

### 학습 단계별 (Training, 시드 1, T = 1)

[`?view=training&step=0&seed=1`](https://jdasam.github.io/mas1004/demos/lecture13-names-rnn/?view=training&step=0&seed=1)에서 걸음을 바꿔 간다.

| 학습 걸음 | 0 | 30 | 100 | 300 | 1,000 | 3,000 | 12,000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 학습 손실 | 3.343 | 2.487 | 2.314 | 2.231 | 2.141 | 2.042 | 1.903 |
| 검증 손실 | 3.343 | 2.489 | 2.318 | 2.237 | 2.159 | 2.091 | 2.016 |
| 첫 문자 a의 확률 | 0.027 | 0.056 | 0.126 | 0.115 | 0.136 | 0.114 | 0.139 |
| 첫 문자 `.`(빈 이름)의 확률 | 0.028 | 0.028 | 0.014 | 0.0062 | 0.0029 | 0.0017 | 0.00087 |
| 12개 중 새 이름 | 12 | 11 | 12 | 11 | 11 | 10 | 7 |

학습 이름 중 a로 시작하는 이름의 비율은 0.138(3,973 / 28,829)이다. 0걸음 모델은 첫 문자 27개에 0.027~0.056을 주는데(모두 같으면 1/27 = 0.037), a가 그중 가장 낮고 u가 가장 높다.

시드 1의 이름 1~12번. 0걸음의 7번과 11번은 30자까지 끝나지 않았다.

| 번호 | 0 | 30 | 100 | 300 | 1,000 | 3,000 | 12,000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | zeykfnzatjpa | zayiciy | zayiaht | zaykanu | zayeidu | zaylahn | zaylahn |
| 2 | lwryogqkenxrprgszb | ksryn | jrkyman | jusxich | josye | kriyanh | josya |
| 3 | dkrsofeva | choria | alirh | biophakr | amirh | borola | alivia |
| 4 | zablz | zaciy | zadiy | zadiy | zadey | zaciy | zadiy |
| 5 | nz | lzaran | kzasan | ky | kyarah | kyara | kyan |
| 6 | pdy | may | lay | lay | lay | layanna | layanna |
| 7 | wkroeudiyssfqidhcxbvgvglfjapel | tenman | shinal | shiram | shenai | shendre | shephia |
| 8 | pbcizyseniqwscyozt | madeyxe | laelyva | laelysh | lahayna | ladeysa | lailynn |
| 9 | gwkkgcgpjyuw | frild | drilea | drige | dopei | donica | doniel |
| 10 | spgssglmhlhkiod | qkeroa | niglia | niflia | nikki | niklle | niemi |
| 11 | wlolbjebzfvkrfelipjxbywkoztlfo | vemiaa | thgia | thin | then | tennaa | theo |
| 12 | wnnswqopycjqwiafd | vhlosieer | tillyna | timlona | theori | therrikan | shanvie |

12,000걸음 열의 학습 이름은 alivia, kyan, layanna, lailynn, theo이고 나머지 7개는 새 이름이다.

## 여는 법

- `index.html`을 더블클릭해(`file://`) 열어도 되고 http로 열어도 된다. 데이터와 모델은 모두 `<script>`로 읽는 JS 파일이라 `file://`에서도 읽힌다. ONNX Runtime Web 1.30.0은 jsdelivr CDN에서 받으므로 인터넷이 필요하다. 받지 못하면 화면에 "The demo could not start. ONNX Runtime Web, which runs the model, could not be loaded from cdn.jsdelivr.net. This page needs an internet connection. Check the connection and reload the page."가 나온다.
- 받는 파일: `index.html` 69 KB, `data.js` 232 KB(학습 이름, 검증 이름, 손실 곡선), `model/step*.js` 일곱 개 각 126 KB(ONNX 94 KB의 base64), CDN의 `ort.wasm.min.js` 17 KB, `ort-wasm-simd-threaded.mjs` 9 KB, `ort-wasm-simd-threaded.wasm` 3.1 MB(압축한 크기, 풀면 14 MB). 모델 파일 일곱 개는 처음에 모두 읽지만 ONNX 세션은 그 걸음을 처음 쓸 때 만들고 다시 쓴다.
- GitHub Pages 게시는 교수가 한다: `mas1004-2026/tools/publish_demo.sh lecture13-names-rnn` → https://jdasam.github.io/mas1004/demos/lecture13-names-rnn/ (`test/`는 빠진다). 게시한 뒤 `mas1004/README.md`의 데모 목록에 이 데모의 줄을 따로 더한다.
- 폴더를 옮길 때는 `test/`만 빼고 전부(`model/` 포함) 옮긴다.

## URL 옵션

화면이 바뀔 때마다 주소창의 주소도 바뀌므로, 주소를 복사하면 같은 화면을 다시 열 수 있다.

| 옵션 | 뜻 | 기본 |
| --- | --- | --- |
| `view` | `generate`, `score`, `training` | `generate` |
| `step` | 모델의 학습 걸음: 0, 30, 100, 300, 1000, 3000, 12000 | 12000 |
| `t` | 온도 T, 0.2~2.0(0.1 단위) | 1 |
| `seed` | 시드(0 이상의 정수) | 1 |
| `no` | Generate에서 만드는 이름의 번호 | 1 |
| `prefix` | Generate에서 이미 붙인 문자들(예: `emm`) | 없음 |
| `done` | `1`이면 prefix 뒤에 `.`이 붙어 끝난 상태 | 없음 |
| `list` | 오른쪽 목록에 미리 만들어 둘 이름 수(0~1,000) | 0 |
| `name` | Score a name의 이름 | dasaem |
| `sort`, `scores`, `data`, `hidden`, `count` | `1`이면 Sort by probability, Show scores, Compare with training names, Hidden values, Compare with counting을 켠다 | 꺼짐 |
| `axis` | Training의 가로축: `log`, `linear` | `log` |

## 화면 캡처용 훅

`window.RNN_DEMO.ready`는 시작이 끝나면 풀리는 Promise다. `window.RNN_DEMO.hooks`의 함수는 화면을 다시 그린 뒤 풀리는 Promise를 돌려준다(`state`만 값을 바로 돌려준다).

- 공통: `setView(view)`, `setStep(step)`, `setTemperature(t)`, `setSeed(seed)`, `render()`
- Generate: `appendChar(c)`(`'.'`도 된다), `sampleNext()`, `mostLikely()`, `finish()`, `undo()`, `startOver()`, `setPrefix(letters)`, `generateMany(n = 20)`(n개 더 만들고 목록을 돌려준다), `clearList()`
- Score a name: `setName(text)`
- 계산만: `probsAfter(step, prefix, T = 1)`(`.`과 문자들을 읽은 뒤의 확률 27개), `sampleName(step, T, seed, no)`
- `state()`: 화면에 나온 값. `gen`(토큰, 토큰별 확률, 다음 문자의 점수와 확률, 이 자리의 난수와 그 난수가 고를 문자, 켜져 있으면 학습 이름의 비율), `list`(목록의 이름과 표시), `score`(줄마다 RNN 확률과 −log p, 세기의 횟수와 확률, 두 평균, 검증 이름에서의 세기 결과), `training`(첫 문자의 모델 확률과 학습 이름 비율, 고른 걸음의 이름 12개와 표시), `timing`. ∞는 문자열 `'Infinity'`로 들어 있다.

예: `. e m m` 화면을 찍으려면 `await RNN_DEMO.ready; for (const c of 'emm') await RNN_DEMO.hooks.appendChar(c);`

## 실측 (이 컴퓨터의 headless Chromium, ONNX Runtime Web 1.30.0 WASM 단일 스레드)

`test/smoke.py`를 여러 번 돌려 잰 범위다.

- 열고 나서 첫 모델이 준비되기까지 0.4~2.9초. 테스트마다 새 브라우저 프로필이라 CDN에서 WASM 3.1 MB를 받는 시간이 들어 있다(처음 잰 때는 6.4초). 그중 첫 세션(WASM 컴파일 포함)이 0.3~0.8초, 다른 걸음의 세션은 각 1~32 ms.
- RNN 한 걸음(ONNX 실행 한 번): 평균 0.04~0.12 ms. 브라우저 시계가 0.1 ms 단위라 600번 정도의 평균으로 쟀다.
- 이름 20개 만들기 10~65 ms, 1,000개 0.13~0.28초. Training 화면을 처음 열 때(세션 여섯 개와 이름 84개) 35~115 ms.
- PyTorch와의 차이: 기준 이름 여섯 개(emma, olivia, dasaem, minji, zzyzx, jennifer)의 문자별 확률과 상위 5개 확률이 `names-rnn.json`(소수 4~5자리로 반올림한 값)과 최대 5e-5 다르고, 첫 문자 확률 27개는 최대 4.5e-6 다르다. 세기 통계(앞 두 문자 조합 600개, 세 문자 조합 5,939개, 검증 이름에서 처음 보는 문자 135개, 나머지 평균 2.1915)는 Python 값과 같다.

## 데이터와 라이선스

- 이름: makemore의 `names.txt`(https://github.com/karpathy/makemore, Andrej Karpathy, MIT 라이선스). 미국 사회보장국(SSA)이 공개한 아기 이름 자료에서 나온 32,033줄이다(서로 다른 이름 29,494개, 2~15자). 화면 맨 아래 줄에 출처를 적었다.
- 학습과 검증: 시드 42로 섞어 앞 90%(28,829줄)를 학습에, 나머지 10%(3,204줄)를 검증에 썼다. `data.js`의 `TRAIN_NAMES`, `VAL_NAMES`(정렬, 중복 포함), `RNN_META`(문자, 은닉 크기, 파라미터 수, 체크포인트별 학습·검증 손실, 50걸음마다의 배치 손실).
- 모델: h_t = tanh(x_t W_x + h_{t−1} W_h + b), scores_t = h_t W_o + c. x_t는 27차원 one-hot, 은닉 128, h_0 = 0, 파라미터 23,451개. Adam 학습률 0.003(7,200걸음부터 0.0003), 배치 64, 12,000걸음, CPU로 85초. 체크포인트는 0, 30, 100, 300, 1,000, 3,000, 12,000걸음.
- `model/step<N>.js`: 한 걸음(입력 `x` 1 × 27, `h` 1 × 128, 출력 `scores` 1 × 27, `h_next` 1 × 128)의 ONNX(opset 17)를 base64로 `window.RNN_MODELS[N]`에 넣는다. 만든 스크립트가 PyTorch와 3e-6 안에서 같은지 확인했다.
- ONNX Runtime Web 1.30.0(MIT 라이선스)은 jsdelivr에서 받는다.

## 구조와 테스트

- `index.html`: `<script id="nr-core">`(전역 `NR`, DOM 없음: softmax(scores / T), 시드 난수, 뽑기, 이름 정리, 세 문자 세기, 앞부분별 다음 문자 비율, 이름 표시)와 `<script id="nr-ui">`(ONNX 세션, 화면, 훅).
- 모델 실행: 걸음마다 ONNX 세션 하나를 처음 쓸 때 만들어 둔다. 실행은 한 번에 하나씩 차례로 한다. 앞부분(`.`과 문자들)마다 점수와 h를 기억해 두어 같은 앞부분은 다시 계산하지 않는다.
- `test/`(게시에서 빠진다):

```bash
node --test test/core.test.js                                         # 9개: softmax와 T, 뽑기, 난수, 이름 정리, 세기 통계와 첫 문자 비율이 Python 값과 같은지, 이름 표시 규칙
uv run --no-project --with playwright python test/smoke.py            # file://로 열어 콘솔 오류 없음, 기준 이름과 첫 문자 확률을 PyTorch와 대조(1e-4), 시드 재현(두 번, 다시 열기, 주소만으로), 세기 열은 켤 때만 보이고 켜면 ∞, 마우스와 키보드, CDN을 막았을 때의 안내, http, 캡처, README 예시 대조
uv run --no-project --with playwright python test/smoke.py --update   # 데모를 일부러 바꿨을 때 test/examples.json을 다시 쓴다
```

- `test/shots/`(1600 × 1000, 배율 2): `generate-emm.png`(`. e m m`과 이름 20개), `generate-emm-hidden-data.png`(은닉값과 학습 이름 비교), `score-dasaem.png`, `score-minji.png`(세기를 켜고 본 ∞), `training-step0.png`, `training-step12000.png`, `offline.png`(CDN을 막았을 때).
- `test/examples.json`: 위 「수업에서 보일 예시」의 값.

## 남은 것

- 두 13차시 데모의 차이 가운데 강조색과 확률 표기는 n-gram 데모에 맞췄다. 남은 차이는 세는 이름(이 데모 학습 이름 28,829개, n-gram 전체 32,033개)과 Undo 뒤의 난수(이 데모 같은 난수, n-gram 다음 난수)다. 맞출지 정하면 한쪽을 고친다.
- 게시는 하지 않았다(교수가 publish_demo.sh로 한다).
