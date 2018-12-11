import codecs
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
from keras import Input, Model
from keras.models import load_model
from keras.initializers import RandomUniform
from keras.layers import Embedding, Dropout, Dense, GlobalMaxPooling1D, Bidirectional, LSTM, TimeDistributed, \
    MaxPooling1D, Conv1D, Flatten, concatenate, Lambda, K
from keras.optimizers import Adadelta
from keras.utils import plot_model
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import text_to_word_sequence

from data.variable import Variable
from preprocess.pickle import read_variables_from_pickle


def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


class SimilarityCNNLSTM:
    def __init__(self, max_len: int, w2v_file: str):
        """
        :param max_len: maximum length of sentences
        :param w2v_file: path to word2vec file
        """
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
        word_indices_for_values = []
        label_indices_for_values = []
        char_indices_for_values = np.zeros(shape=(len(examples), self.max_len, 52))

        for idx1, example in enumerate(examples):
            word_indices = []
            words = text_to_word_sequence(" ".join(example.values))
            for idx2, word in enumerate(words[:self.max_len]):
                if word in self.word_to_index:
                    word_indices.append(self.word_to_index[word])
                elif word.lower() in self.word_to_index:
                    word_indices.append(self.word_to_index[word.lower()])
                else:
                    word_indices.append(self.word_to_index["UNKNOWN_TOKEN"])

                for idx3, char in enumerate(word[0][:52]):
                    char = char.encode("ascii", "ignore")
                    if char in self.char_to_index:
                        char_indices_for_values[idx1][idx2][idx3] = self.char_to_index[char]

            label_indices_for_values.append([self.label_to_index[example.semantic_type]])
            word_indices_for_values.append(word_indices)

        word_indices_for_values = pad_sequences(word_indices_for_values, maxlen=self.max_len,
                                                padding="post", value=self.word_to_index["PADDING_TOKEN"])

        return np.array(word_indices_for_values), np.array(label_indices_for_values), char_indices_for_values

    def build_char_cnn(self):
        # character input layer
        character_input = Input(shape=(self.max_len, 52,))

        # embedding for characters
        embed_char_out = TimeDistributed(
            Embedding(len(self.char_to_index), 30,
                      embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)))(character_input)

        dropout = Dropout(0.5)(embed_char_out)

        # Build character CNN
        conv_1d_out = TimeDistributed(
            Conv1D(kernel_size=3, filters=30, padding='same',
                   activation='tanh', strides=1))(dropout)
        max_pool_out = TimeDistributed(MaxPooling1D(52))(conv_1d_out)
        char_model = TimeDistributed(Flatten())(max_pool_out)
        char_model = Dropout(0.5)(char_model)
        return character_input, char_model

    def build_embed(self):
        char_input, char_cnn = self.build_char_cnn()

        word_input = Input(shape=(self.max_len,), dtype='int32')
        word_embed = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                               weights=[self.word_embeddings], trainable=False)(word_input)

        output = concatenate([char_cnn, word_embed])
        return char_input, word_input, output

    def build_model(self):
        """
            Build LSTM model for relation classification with or without position features
        """
        left_char_input, left_word_input, left_embed = self.build_embed()
        right_char_input, right_word_input, right_embed = self.build_embed()

        shared_lstm = Bidirectional(LSTM(units=100, dropout=0.75,
                                         recurrent_dropout=0.25), name="BiLSTM")

        left_output = shared_lstm(left_embed)
        right_output = shared_lstm(right_embed)

        distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                          output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        self.model = Model(inputs=[left_char_input, left_word_input, right_char_input, right_word_input],
                           outputs=[distance])

        optimizer = Adadelta(clipnorm=1.25)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        plot_model(self.model, to_file='sim_cnn.png')
        self.model.summary()

    def train(self, train_vectors, train_labels):
        """
                Run training algorithm of Relation CNN model
                :param train_vectors: training feature vectors of shape (num_samples, max_len)
                :param train_labels: training labels in one-hot encoding form (num_samples, num_labels)
                :return:
                """
        history = self.model.fit(train_vectors, train_labels, batch_size=256, epochs=20,
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

        train_vectors, _, train_chars = self.preprocess(train_examples)

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
        left_vectors, _, left_chars = self.preprocess([x[0] for x in train_pairs])
        right_vectors, _, right_chars = self.preprocess([x[1] for x in train_pairs])

        # self.train([left_chars, left_vectors, right_chars, right_vectors], train_labels)
        # self.model.save_weights("model.h5")
        # del self.model
        self.model.load_weights("model.h5")

        groundtruth_to_predictions = {}

        for test_example in test_examples:
            test_vectors, _, test_chars = self.preprocess([test_example] * len(train_examples))
            train_vectors, _, train_chars = self.preprocess(train_examples)
            test_indices_for_value = [x[0] for x in
                                      self.predict_sim([test_chars, test_vectors, train_chars, train_vectors])]
            best_indices = np.array(test_indices_for_value).argsort()[::-1]
            best_semantic_types = []

            for index in best_indices:
                if train_examples[index].semantic_type not in best_semantic_types:
                    best_semantic_types.append(train_examples[index].semantic_type)
            groundtruth_to_predictions[test_example] = best_semantic_types[:10]

        return groundtruth_to_predictions


if __name__ == "__main__":
    model = SimilarityCNNLSTM(100, "embeddings/glove.6B.50d.txt")
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
