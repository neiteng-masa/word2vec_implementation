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

まずベクトルの次元数やコーパスの最大サイズなどを `src/configure.hpp` で設定し、ビルドする。
ハイパーパラメータは src/train.cpp を直接編集する。

`make` すると実行ファイル `bin/train` が生成される。

以下のコマンドで学習が実行される。
```
./bin/train wiki/corpus vec/wiki
```
単語ベクトルのバイナリは `vec/wiki` に出力される。バイナリの構造は、
```
v11, v12, v13, ..., v1n, v21, ..., v2n, ...
```
nは次元数、vij はi番目の単語ベクトルj成分。各 vij は double。

### 4. 実験

```
from vector_operator import VectorOperator
vec = VectorOperator(次元数, "vec/wiki", "wiki/word_to_id.pkl", "wiki/id_to_word.pkl")
```
で単語ベクトルデータを読み込める。
```
vec.nearest_words(vec.vec("father"))
#出力

```

```
vec.nearest_words(vec.vec("father") - vec.vec("man") + vec.vec("woman"))
#出力

```
