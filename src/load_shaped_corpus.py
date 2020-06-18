import sys

def load_shaped_corpus(in_path):
    vocab_num = int.from_bytes(byte[0:4], sys.byteorder)
    cooccurrence_matrix_view = memoryview(byte[4:])
    cooccurrence_matrix = np.array(cooccurrence_matrix_view.cast("I").tolist()).reshape(vocab_num, -1)
    return cooccurrence_matrix

