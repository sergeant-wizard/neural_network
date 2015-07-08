# オートエンコーダー（５章）の実装

## 前回との差分
- `DeltaB`の行列の大きさ修正
- MNISTで入出力ができるようになった
- 重み行列の初期値重みをガウス分布に近くなるように修正（まだ暫定で、今後ガウス分布そのものにする予定）
- バイアスも乱数で初期化するように修正

## 結果
教科書の通り、学習の結果入力画像の復元結果が少しぼやけたものになった。
784(28X28)画素を中間層で100成分に凝集したにも関わらず特徴を捉えられている点が興味深い。
テキストに書いてあり通り、主成分分析に対応する作用の恩恵。

入力
![residue](https://github.com/sergeant-wizard/neural_network/blob/master/auto_encoder/input.png)
出力
![residue](https://github.com/sergeant-wizard/neural_network/blob/master/auto_encoder/100_100.png)
