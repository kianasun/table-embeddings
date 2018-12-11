import os
import sys

from semantic_labeling.models.cnn_lstm import ClassificationCNNLSTM
from semantic_labeling.models.hatt import HierarchicalAttentionNetwork
from semantic_labeling.models.sim_cnn_lstm import SimilarityCNNLSTM
from semantic_labeling.models.tfidf import TFIDFModel

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='evaluate tagging results using CoNLL criteria',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg = parser.add_argument
    arg('-t', '--train', metavar='STR', default='../data/train.pkl',
        help='path to training data file')
    arg('-e', '--test', metavar='STR', default='../data/test.pkl',
        help='path to testing data file')
    arg('-m', '--model', metavar='INT', default=3,
        help='NER learning model (0 for CNN-LSTM, 1 for Sim CNN-LSTM, 2 for HATT, 3 for TF-IDF)')

    args = parser.parse_args(sys.argv[1:])
    args.model = int(args.model)
    if args.model == 0:
        model = ClassificationCNNLSTM(100, os.path.join("..", "embeddings", "glove.6B.50d.txt"))
    elif args.model == 1:
        model = SimilarityCNNLSTM(100, os.path.join("..", "embeddings", "glove.6B.50d.txt"))
    elif args.model == 2:
        model = HierarchicalAttentionNetwork(20, 100, os.path.join("..", "embeddings", "glove.6B.50d.txt"))
    else:
        model = TFIDFModel()

    train_file_path = args.train
    test_file_path = args.test
    results = model.run_experiments(train_file_path, test_file_path)

    true_count = 0
    total_count = 0
    mrr_count = 0.0
    for example, result in results.items():
        if example.semantic_type in result:
            true_count += 1
            mrr_count += 1.0 / (result.index(example.semantic_type) + 1)
        total_count += 1

    print("Hit@10:", true_count * 1.0 / total_count)
    print("MRR:", mrr_count / total_count)
