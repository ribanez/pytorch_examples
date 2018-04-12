import sys
from src.DNN import Word2vec

if __name__ == '__main__':
    w2v = Word2Vec(input_file_name=sys.argv[1], output_file_name=sys.argv[2])
    w2v.train()