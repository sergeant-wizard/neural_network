## 今回の目標
[前回](https://github.com/sergeant-wizard/neural_network/tree/master/back_propagation_20150630)は
2層のネットワークであったが、
今回は3層のネットワークでより自由度の高い学習が可能であるかを検証する。
具体的には、教師あり学習により恒等写像の実現性を検証する。

## 前回との差分
- 基本的に入力/出力の差分のみ。
- `LastLayer`を導入
- 収束の様子がわかるように残差を導入
