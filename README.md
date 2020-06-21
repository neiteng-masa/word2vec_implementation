# word2vec_implementation

## 使い方 (Wikipedia コーパス)
### 1. コーパスデータ整形
WikiExtractor で整形済み `text/` 以下ののコーパスデータを `cat text/*/* > wiki.txt` で結合したファイル `wiki.txt`を用意する。

### 2. バイナリデータに変換

`wiki.txt`をバイナリデータに変換する。
~~~
python wiki_to_corpus_data.py -p wiki.txt -o wiki
~~~
`wiki/`フォルダ以下に次のファイルが生成される。

    corpus            # 単語ID(int32)の列
    word_to_id.pkl    # key: word, value: 単語ID の dictionary
    id_to_word.pkl    # ID順にならんだ単語(str)列
    
### 3. 学習

以下のコマンドで学習が実行される。

```
python train.py wiki/corpus vec/wiki -d 300 -t 30 # 300次元, 30 スレッド
```
ハイパーパラメータはコマンドライン引数によって与えられる。詳細は `python train.py --help`

学習器はC++で実装されている。以下の流れで学習が進む。
1. 学習器をビルドに必要なパラメータを設定する。
2. 学習器をビルドする。
3. 学習器を実行する。

単語ベクトルのバイナリは `vec/wiki` に出力される。バイナリの構造は、
```
v11, v12, v13, ..., v1n, v21, ..., v2n, ...
```
nは次元数、vij は i 番目の単語ベクトル j 成分。各 vij は double。

### 4. 遊んでみる

python インタープリタ上で
```
from vector_operator import VectorOperator
vec = VectorOperator(300, "vec/wiki", "wiki/word_to_id.pkl", "wiki/id_to_word.pkl")
```
で単語ベクトルデータを読み込める。

"tokyo" のcos類似単語 :
```
vec.nearest_words(vec.vec("tokyo"))
#出力
(['tokyo', 'nagoya', 'osaka', 'shinagawa', 'japan', 'yoshijirō', 'ichigaya', 'kōtō', 'keio', 'seizo'],
[1.0, 0.6413310494175718, 0.6309932972537823, 0.5916276861299022, 0.5885590022607644, 0.5850446537666311, 0.5740446555447388, 0.5720594778884815, 0.5704665349140614, 0.569436419162514])
```

"hanako" のcos類似単語 :
```
(['hanako', 'natsuki', 'kyoko', 'misaki', 'kaori', 'yūna', 'nanami', 'haruka', 'makoto', 'takehito'],
[0.9999999999999999, 0.6255682458512841, 0.6195052149402723, 0.6076497471916825, 0.6069407216755482, 0.6001549156295506, 0.5998331419282437, 0.5990212646548976, 0.5968303084693412, 0.5947656988827921])
```

king - woman + man = queen :
```
vec.nearest_words(vec.vec("king") - vec.vec("man") + vec.vec("woman"))
#出力
(['king', 'woman', 'queen', 'she', 'daughter', 'coronation', 'regnant', 'herself', 'archduchess', 'elisabeth'],
[0.705318026855759, 0.5814277758184596, 0.4805892471782488, 0.47106896398597803, 0.46796439829993675, 0.45618244060582597, 0.43850040481795466, 0.4337381074258372, 0.4267931177590809, 0.42228663337944594])
```
