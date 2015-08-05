## RBM with CD

### 背景

[前回](https://github.com/sergeant-wizard/neural_network/tree/master/rbm)
のRBMにContrastive Divergenceを導入

### 結果

入力データを列方向に並べた行列 `input` を幾つか用意し、
それぞれの入力に対してネットワークが示す確率が正しいかを検証する。

input | Test No.
----- | --------
0 0 0 | 0
1 0 0 | 1
1 0 0, 0 1 0 | 2
1 0 0, 0 1 0, 0 0 1 | 3

#### Test No. 0

単一の入力に対してのみ高い確率を示すことを確認。

input | actual | expected
----- | ------ | --------
000   |  0.932 | 1
100   |  0.021 | 0
010   |  0.018 | 0
110   |  0.002 | 0
001   |  0.020 | 0
101   |  0.003 | 0
011   |  0.002 | 0
111   |  0.002 | 0

#### Test No. 1

単一の入力に対してのみ高い確率を示すことを確認。

input | actual | expected
----- | ------ | --------
000   | 0.007  | 0
100   | 0.961  | 1
010   | 0.001  | 0
110   | 0.014  | 0
001   | 0.001  | 0
101   | 0.016  | 0
011   | 0.000  | 0
111   | 0.001  | 0

#### Test No. 2
確率を示さないはずのパターンに対して高い確率を示してしまっている。
勾配消失の可能性があるので、今後モメンタムや重み減衰を試してみる。

input | actual | expected
----- | ------ | --------
000   | 0.343  | 0
100   | 0.238  | 0.5
010   | 0.217  | 0.5
110   | 0.149  | 0
001   | 0.020  | 0
101   | 0.013  | 0
011   | 0.012  | 0
111   | 0.008  | 0

#### Test No. 3

ネットワークが想起するパターンは正しいものの、
`000`と`111`の確率に差がでた。
極端な入力の違いに対し、ネットワークの自由度が不足しているのではないかと考察。

input | actual | expected
----- | ------ | --------
000   | 0.340  | 0.5
100   | 0.002  | 0
010   | 0.002  | 0
110   | 0.007  | 0
001   | 0.002  | 0
101   | 0.007  | 0
011   | 0.007  | 0
111   | 0.635  | 0.5

#### Test No. 4

２種類の入力に対して等しい確率を示し、想定通りの挙動を示すことを確認。

input | actual | expected
----- | ------ | --------
000   | 0.492  | 0.5
100   | 0.001  | 0
010   | 0.001  | 0
110   | 0.000  | 0
001   | 0.505  | 0.5
101   | 0.001  | 0
011   | 0.001  | 0
111   | 0.000  | 0

#### Test No. 5

2:1の比率で出現する入力に対し、想定通りの挙動を示すことを確認

input | actual | expected
----- | ------ | --------
000   | 0.010 | 0
100   | 0.000 | 0
010   | 0.325 | 0.333
110   | 0.002 | 0
001   | 0.652 | 0.667
101   | 0.000 | 0
011   | 0.010 | 0
111   | 0.000 | 0

#### Test No. 6

他のテストケースに比べるとやや入力していないパターンの確率が高いが、
反復回数を増やせば期待される確率に近づくことを確認。

input | actual | expected
----- | ------ | --------
000   | 0.000  | 0
100   | 0.032  | 0
010   | 0.006  | 0
110   | 0.302  | 0.333
001   | 0.031  | 0
101   | 0.283  | 0.333
011   | 0.287  | 0.333
111   | 0.059  | 0

### まとめ

7ケース中、6ケースで期待通りの挙動となることを確認した。
今後は contrastive divergence の効果や、
DBN, DBMへの拡張をしてみる。