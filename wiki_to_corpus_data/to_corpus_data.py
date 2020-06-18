import argparse
import re
from tqdm import tqdm
import pickle as pkl
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pages", help = "Wikipedia全文書へのパス(WikiExtractor による <doc> タグを含んでいて良い)")
    parser.add_argument("-o", "--out")

    args = parser.parse_args()

    in_path = args.pages
    out_dir = args.out

    word_to_id = {}
    corpus = []

    print("Making vocabulary ...")
    with open(in_path, "r") as f:
        is_title_line = False
        for line in tqdm(f):
            if "<doc" in line:
                is_title_line = True
                continue
            if is_title_line:
                is_title_line = False
                continue
            if "</doc>" in line:
                continue
            line = line.lower()
            words = re.split(r"\W+", line)

            for word in words:
                if len(word) == 0:
                    continue
                word_id = word_to_id.get(word)
                if word_id is None:
                    word_id = len(word_to_id)
                    word_to_id[word] = word_id

                corpus.append(word_id)

    print("Just a minute ...")
    id_to_word = [ "" for _ in range(len(word_to_id)) ]
    for word, i in tqdm(word_to_id.items()):
        id_to_word[i] = word

    os.makedirs(out_dir, exist_ok = True)

    np.array(corpus, dtype="int32").tofile(os.path.join(out_dir, "corpus"))
    with open(os.path.join(out_dir, "word_to_id.pkl"), "wb") as f:
        pkl.dump(word_to_id, f)
    with open(os.path.join(out_dir, "id_to_word.pkl"), "wb") as f:
        pkl.dump(id_to_word, f)

