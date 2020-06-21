import pickle as pkl
import numpy as np

class VectorOperator:
    def __init__(self, dim, vec_path, word_to_id_path, id_to_word_path):
        print("Loading vector binary file ... ")
        self.vector = np.fromfile(vec_path, "float64").reshape(-1, dim)
        with open(word_to_id_path, "rb") as f:
            self.to_id = pkl.load(f)
        with open(id_to_word_path, "rb") as f:
            self.to_word = pkl.load(f)
        print("Preparing word vectors ...")
        self.norm = np.linalg.norm(self.vector, axis = -1)
        self.norm = np.maximum(self.norm, 1e-8)

    def vec(self, word):
        return self.vector[self.to_id[word]]

    def nearest_words(self, vec, n = 10):
        # size: vocab_num
        cos = np.dot(self.vector, vec) / (np.linalg.norm(vec) * self.norm)
        # top n index
        index = np.argpartition(-cos, n)[:n]
        top = index[np.argsort(-cos[index])]
        word_list = []
        sim_list = []
        for word_id in top:
            word = self.to_word[word_id]
            sim = cos[word_id]
            word_list.append(word)
            sim_list.append(sim)
        return word_list, sim_list

