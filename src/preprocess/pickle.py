import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

from data.variable import Variable


def read_from_pickle_folder(folder_path: Path):
    predicate_to_column = defaultdict(list)

    for file in folder_path.iterdir():
        print(file.name)
        temp_dict = pickle.load(file.open("rb"))
        predicate_to_column.update(temp_dict)

    return predicate_to_column


def fix_pickle_files(input_path: Path, output_path: Path):
    predicate_to_columns = defaultdict(list)

    for file in input_path.iterdir():
        print(file.name)
        temp_dict = pickle.load(file.open("rb"))
        predicate_to_columns.update(temp_dict)

    train_predicate_to_columns = defaultdict(list)
    test_predicate_to_columns = defaultdict(list)
    for predicate, columns in predicate_to_columns.items():
        test_predicate_to_columns[predicate] = [[]]
        for idx, column in enumerate(columns):
            if idx < len(columns) * 0.7:
                if idx % 1000 == 0:
                    train_predicate_to_columns[predicate].append([])
                train_predicate_to_columns[predicate][-1].append("".join(column))
            else:
                if idx % 1000 == 0:
                    test_predicate_to_columns[predicate].append([])
                test_predicate_to_columns[predicate][-1].append("".join(column))

    pickle.dump(train_predicate_to_columns, (output_path / "train.pkl").open("wb"))
    pickle.dump(test_predicate_to_columns, (output_path / "test.pkl").open("wb"))


def read_variables_from_pickle(pkl_file: Path) -> List[Variable]:
    variables = []
    predicate_to_columns: Dict[str, List[List[str]]] = pickle.load(pkl_file.open("rb"))

    for predicate, columns in predicate_to_columns.items():
        for column in columns:
            variables.append(Variable(predicate, column))
    return variables


if __name__ == "__main__":
    fix_pickle_files(Path("data") / "pickle", Path("data") / "pkl")
