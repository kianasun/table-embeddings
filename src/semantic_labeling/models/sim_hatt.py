import codecs
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
from keras import Input, Model
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, \
    Lambda, K
from keras.optimizers import Adadelta
from keras.utils import plot_model
from keras_preprocessing.text import text_to_word_sequence

from data.variable import Variable
from preprocess.pickle import read_variables_from_pickle
from semantic_labeling.custom_layers.attn_layer import AttLayer


def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


class SimHierarchicalATTN:
    def __init__(self, max_cell: int, max_len: int, w2v_file: str):
        """
        :param max_len: maximum length of sentences
        :param w2v_file: path to word2vec file
        """
        self.max_cell = max_cell
        self.max_len = max_len

        self.model = None

        self.word_to_index = {}
        self.vocab = set()

        self.position_to_index = {}

        self.label_to_index = {}
        self.index_to_label = {}
        self.w2v_file = w2v_file
        self.word_embeddings = None

        self.char_to_index = {"PADDING": 0}

        for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|<>":
            self.char_to_index[c] = len(self.char_to_index)

    def init_params(self, train_examples: List[Variable]):
        """
            Build vocabulary for words, positions and labels.
        :param train_examples: training examples
        """
        for example in train_examples:
            if example.semantic_type not in self.label_to_index:
                self.label_to_index[example.semantic_type] = len(self.label_to_index)
                self.index_to_label[len(self.index_to_label)] = example.semantic_type

            for value in example.values:
                words = text_to_word_sequence(value)
                for word in words:
                    self.vocab.add(word)

    def load_w2v(self):
        """
            Load pre-trained word embeddings
        """
        word_to_embedding = {}
        length = 0

        with codecs.open(self.w2v_file, "r", encoding="utf-8") as reader:
            for line in reader.readlines():
                values = line.split()
                word = values[0]
                weights = np.asarray(values[1:], dtype='float32')
                length = len(weights)
                word_to_embedding[word] = weights

        embedding_matrix = np.zeros((len(self.vocab) + 2, length))
        for index, word in enumerate(self.vocab):
            embedding_vector = word_to_embedding.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[index] = embedding_vector
                self.word_to_index[word] = index

        self.word_to_index["PADDING_TOKEN"] = len(self.word_to_index)
        embedding_matrix[len(self.word_to_index)] = np.zeros(len(embedding_matrix[0]))

        self.word_to_index["UNKNOWN_TOKEN"] = len(self.word_to_index)
        embedding_matrix[len(self.word_to_index)] = np.random.uniform(-0.25, 0.25, len(embedding_matrix[0]))

        self.word_embeddings = embedding_matrix

    def preprocess(self, examples: List[Variable]):
        """
            Create input matrices for words from sentences. Position features of words are also included
        :param examples: list of relation example that need to be processed
        :return: tuple of 4 ndarray (numpy) for indices of words, labels, distances to first and second mentions
        """
        word_indices_for_column_cells = np.full((len(examples), self.max_cell, self.max_len),
                                                self.word_to_index["UNKNOWN_TOKEN"])
        label_indices_for_values = []

        for idx1, example in enumerate(examples):
            for idx2, cell in enumerate(example.values[:self.max_cell]):
                words = text_to_word_sequence(cell)
                for idx3, word in enumerate(words[:self.max_len]):
                    index = self.word_to_index["UNKNOWN_TOKEN"]
                    if word in self.word_to_index:
                        index = self.word_to_index[word]
                    elif word.lower() in self.word_to_index:
                        index = self.word_to_index[word.lower()]
                    word_indices_for_column_cells[idx1][idx2][idx3] = index

            label_indices_for_values.append([self.label_to_index[example.semantic_type]])

        return word_indices_for_column_cells, np.array(label_indices_for_values)

    def build_embed(self):
        word_input = Input(shape=(self.max_len,), dtype='int32')
        word_embed = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                               weights=[self.word_embeddings], trainable=False)(word_input)
        return word_input, word_embed

    def build_model(self):
        """
            Build LSTM model for relation classification with or without position features
        """
        word_input = Input(shape=(self.max_len,), dtype='int32', name='words_input')
        word_embed = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                               weights=[self.word_embeddings], trainable=False)(word_input)

        cell_output = Bidirectional(LSTM(units=50, dropout=0.75, return_sequences=True,
                                         recurrent_dropout=0.25), name="BiLSTM1")(word_embed)

        cell_att = AttLayer(100)(cell_output)
        cell_encoder = Model(word_input, cell_att)

        word_column_input = Input(shape=(self.max_cell, self.max_len,), dtype='int32')

        column_encoder = TimeDistributed(cell_encoder)(word_column_input)
        column_output = Bidirectional(LSTM(units=50, dropout=0.75, return_sequences=True,
                                           recurrent_dropout=0.25), name="BiLSTM2")(column_encoder)

        column_att = AttLayer(100)(column_output)

        shared_model = Model(inputs=word_column_input, outputs=column_att)

        left_word_column_input = Input(shape=(self.max_cell, self.max_len,), dtype='int32')
        right_word_column_input = Input(shape=(self.max_cell, self.max_len,), dtype='int32')

        left_column_att = shared_model(left_word_column_input)
        right_column_att = shared_model(right_word_column_input)

        distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                          output_shape=lambda x: (x[0][0], 1))([left_column_att, right_column_att])

        self.model = Model(inputs=[left_word_column_input, right_word_column_input],
                           outputs=[distance])

        optimizer = Adadelta(clipnorm=1.25)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        plot_model(self.model, to_file='sim_hatt.png')

        return column_att

    def train(self, train_vectors, train_labels):
        """
                Run training algorithm of Relation CNN model
                :param train_vectors: training feature vectors of shape (num_samples, max_len)
                :param train_labels: training labels in one-hot encoding form (num_samples, num_labels)
                :return:
                """
        history = self.model.fit(train_vectors, train_labels, batch_size=32, epochs=20,
                                 verbose=2)
        return history

    def predict_sim(self, test_vectors):
        """
            Predict labels of given feature vectors
        :param test_vectors: training feature vectors of shape (num_samples, max_len)
        :return:
        """
        sim_for_pair_values = self.model.predict(test_vectors, verbose=1)
        return sim_for_pair_values

    def run_experiments(self, train_file: str, test_file: str):
        """
            Train the model using data in train file and classify the relations provided in test file
        :param train_file: training file
        :param test_file: testing file
        :return:
        """

        # train_examples = read_variables_from_folder(Path(train_file))
        # test_examples = read_variables_from_folder(Path(test_file))

        train_examples = read_variables_from_pickle(Path(train_file))
        test_examples = read_variables_from_pickle(Path(test_file))

        self.init_params(train_examples)
        self.load_w2v()
        self.build_model()

        train_pairs = []
        train_labels = []

        train_vectors, _ = self.preprocess(train_examples)

        for idx1, train_example1 in enumerate(train_examples):
            for idx2, train_example2 in enumerate(train_examples):
                if len(train_pairs) > 50000:
                    break
                if train_example1.semantic_type == train_example2.semantic_type:
                    train_labels.append(1)
                else:
                    train_labels.append(0)
                train_pairs.append((train_example1, train_example2))
        test_to_train = defaultdict(list)

        for test_example in test_examples:
            for train_example in train_examples:
                test_to_train[test_example].append(train_example)

        print(len(train_examples), len(train_pairs))
        left_vectors, _ = self.preprocess([x[0] for x in train_pairs])
        right_vectors, _ = self.preprocess([x[1] for x in train_pairs])

        self.train([left_vectors, right_vectors], train_labels)
        self.model.save_weights("model.h5")
        del self.model
        # self.model.load_weights("model.h5")

        groundtruth_to_predictions = {}

        for test_example in test_examples:
            test_vectors, _ = self.preprocess([test_example] * len(train_examples))
            train_vectors, _ = self.preprocess(train_examples)
            test_indices_for_value = [x[0] for x in
                                      self.predict_sim([test_vectors, train_vectors])]
            best_indices = np.array(test_indices_for_value).argsort()[::-1]
            best_semantic_types = []

            for index in best_indices:
                if train_examples[index].semantic_type not in best_semantic_types:
                    best_semantic_types.append(train_examples[index].semantic_type)
            groundtruth_to_predictions[test_example] = best_semantic_types[:10]

        return groundtruth_to_predictions


if __name__ == "__main__":
    model = SimHierarchicalATTN(100, 20, "embeddings/glove.6B.50d.txt")
    # model = TFIDFModel()
    #
    # model = Doc2VecModel()
    train_file_path = "data/pkl/train.pkl"
    test_file_path = "data/pkl/test.pkl"
    results = model.run_experiments(train_file_path, test_file_path)

    true_count = 0
    total_count = 0
    mrr_count = 0.0
    for example, result in results.items():
        print(example, result)
        if example.semantic_type in result:
            true_count += 1
            mrr_count += 1.0 / (result.index(example.semantic_type) + 1)
        total_count += 1

    print(true_count * 1.0 / total_count)
    print(mrr_count / total_count)
    #
    # with open("prediction.txt", "w") as writer:
    #     for idx, relation in result_relations.items():
    #         writer.write(f"{str(idx)} {relation}\n")
    #
    # y_true, y_pred = read_test_data('key.txt', 'prediction.txt')
    # evaluate(y_true, y_pred)
