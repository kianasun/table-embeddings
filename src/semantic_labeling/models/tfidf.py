from pathlib import Path
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from data.variable import Variable
from preprocess.pickle import read_variables_from_pickle


class TFIDFModel:
    def __init__(self):
        self.id_to_label: Dict[int, str] = {}
        self.label_to_id: Dict[str, int] = {}
        self.model = TfidfVectorizer()
        self.train_vectors = None
        self.train_labels = []

    def train(self, variables: List[Variable]):
        documents = []
        for variable in variables:
            if variable.semantic_type not in self.label_to_id:
                id = len(self.id_to_label)
                self.id_to_label[id] = variable.semantic_type
                self.label_to_id[variable.semantic_type] = id
                text = " ".join(variable.values)
                documents.append(text)
                self.train_labels.append(id)

        self.train_vectors = self.model.fit_transform(documents)

    def predict_variables(self, variables: List[Variable]) -> List[str]:
        return [self.predict(variable.values) for variable in variables]

    def predict_top_k_variables(self, variables: List[Variable], k: int) -> List[List[str]]:
        return [self.predict_top_k(variable.values, k) for variable in variables]

    def predict_proba(self, values: List[str]) -> Dict[str, float]:
        vector = self.model.transform([" ".join(values)])
        cosine_similarities = linear_kernel(vector[0:1], self.train_vectors).flatten()
        related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
        return {self.id_to_label[self.train_labels[i]]: cosine_similarities[i] for i in related_docs_indices}

    def predict(self, values: List[str]) -> str:
        label_to_score = self.predict_proba(values)
        return max(label_to_score.items(), key=lambda x: x[1])[0]

    def predict_top_k(self, values: List[str], k: int) -> List[str]:
        label_to_score = self.predict_proba(values)
        return [x[0] for x in sorted(label_to_score.items(), key=lambda x: x[1], reverse=True)[:k]]

    def run_experiments(self, train_file: str, test_file: str):
        train_examples = read_variables_from_pickle(Path(train_file))
        test_examples = read_variables_from_pickle(Path(test_file))

        self.train(train_examples)

        return {test_examples[i]: result for i, result in
                enumerate(self.predict_top_k_variables(test_examples, 10))}


if __name__ == "__main__":
    model = TFIDFModel()
    #
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
