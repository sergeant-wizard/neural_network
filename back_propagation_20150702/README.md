## 今回の目標
5章の例題でモメンタムと重み減衰を導入していたので、とりあえずこの段階で実装しておいた

## 実装
`Layer::GradientDescent`を重みとバイアスの更新に分け、
重みについては減衰とモメンタム、
バイアスについてはモメンタムの項を追加。
係数はテキストで推奨されている値をそのまま使用。
大きすぎると返って収束性能が悪くなることを確認。

## 結果
問題が単純すぎるせいか、収束性能の劇的な変化はなかった。
![residue](https://github.com/sergeant-wizard/neural_network/blob/master/back_propagation_20150702/residue.png)

## 次回の目標
オートエンコーダー！！！
