import codecs
from pathlib import Path
from typing import List

import numpy as np
from keras import Input, Model
from keras.layers import Embedding, Dropout, Dense, GlobalMaxPooling1D, Bidirectional, LSTM
from keras.utils import plot_model
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import text_to_word_sequence

from data.variable import Variable
from preprocess.pickle import read_variables_from_pickle


class ClassificationLSTM:
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

        self.position_dims = 5

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

        for idx1, example in enumerate(examples):
            word_indices = []
            words = text_to_word_sequence(" ".join(example.values))
            for idx2, word in enumerate(words[:100]):
                if word in self.word_to_index:
                    word_indices.append(self.word_to_index[word])
                elif word.lower() in self.word_to_index:
                    word_indices.append(self.word_to_index[word.lower()])
                else:
                    word_indices.append(self.word_to_index["UNKNOWN_TOKEN"])

            label_indices_for_values.append([self.label_to_index[example.semantic_type]])
            word_indices_for_values.append(word_indices)

        word_indices_for_values = pad_sequences(word_indices_for_values, maxlen=self.max_len,
                                                padding="post", value=self.word_to_index["PADDING_TOKEN"])

        return np.array(word_indices_for_values), np.array(label_indices_for_values)

    def build_model(self):
        """
            Build LSTM model for relation classification with or without position features
        """
        words_input = Input(shape=(self.max_len,), dtype='int32', name='words_input')
        words = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                          weights=[self.word_embeddings], trainable=False)(words_input)

        output = words

        output = Bidirectional(LSTM(units=50, dropout=0.75, return_sequences=True,
                                    recurrent_dropout=0.25), name="BiLSTM")(output)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(0.75)(output)
        output = Dense(len(self.label_to_index), activation='softmax')(output)

        self.model = Model(inputs=[words_input], outputs=output)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

        plot_model(self.model, to_file='lstm.png')
        self.model.summary()

    def train(self, train_vectors, train_labels):
        """
                Run training algorithm of Relation CNN model
                :param train_vectors: training feature vectors of shape (num_samples, max_len)
                :param train_labels: training labels in one-hot encoding form (num_samples, num_labels)
                :return:
                """
        history = self.model.fit(train_vectors, train_labels, batch_size=256, epochs=50,
                                 verbose=2)
        return history

    def predict(self, test_vectors):
        """
            Predict labels of given feature vectors
        :param test_vectors: training feature vectors of shape (num_samples, max_len)
        :return:
        """
        test_results_for_values = self.model.predict(test_vectors, verbose=1)
        return [np.argmax(pros) for pros in test_results_for_values]

    def predict_top_k(self, test_vectors, k: int):
        test_results_for_values = self.model.predict(test_vectors, verbose=1)
        print(len(test_results_for_values[0]))
        return [pros.argsort()[::-1][:k] for pros in test_results_for_values]

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

        print(len(train_examples), len(test_examples))

        self.init_params(train_examples)
        self.load_w2v()
        self.build_model()

        train_vectors, train_labels = self.preprocess(train_examples)
        test_vectors, test_labels = self.preprocess(test_examples)

        self.train([train_vectors], train_labels)
        test_indices_for_values = self.predict_top_k([test_vectors], 50)

        return {test_examples[i]: [self.index_to_label[label_index] for label_index in label_indices]
                for i, label_indices in enumerate(test_indices_for_values)}


if __name__ == "__main__":
    model = ClassificationLSTM(100, "embeddings/glove.6B.50d.txt")
    # model = TFIDFModel()
    #
    # model = Doc2VecModel()
    train_file_path = "data/pkl/train.pkl"
    test_file_path = "data/pkl/test.pkl"
    results = model.run_experiments(train_file_path, test_file_path)

    print(results)
    true_count = 0
    total_count = 0
    mrr_count = 0.0
    for example, result in results.items():
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
