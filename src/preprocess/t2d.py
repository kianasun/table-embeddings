import csv
import json
from collections import defaultdict
from typing import Dict, List


def read_tables_from_t2d_folder(t2d_folder) -> Dict[str, List[List[str]]]:
    def read_t2d_table_with_properties(table_file, property_file):
        text = table_file.read_text(encoding="utf-8", errors="ignore")
        json_obj = json.loads(text)
        if not property_file.exists():
            return
        with property_file.open() as f:
            csv_reader = csv.reader(f)
            name_to_predicate = {}
            for row in csv_reader:
                name_to_predicate[row[1]] = row[0]
            for column in json_obj["relation"]:
                column_name = column[0].lower().strip()
                if column_name in name_to_predicate:
                    predicate_to_columns[name_to_predicate[column_name]].append(column)

    table_folder = t2d_folder / "tables"
    property_folder = t2d_folder / "property"
    predicate_to_columns: Dict[str, List[List[str]]] = defaultdict(lambda: defaultdict(list))

    for t_file in table_folder.iterdir():
        name = t_file.stem.replace(".", "_")
        p_file = property_folder / (name + ".csv")

        read_t2d_table_with_properties(t_file, p_file)

    return predicate_to_columns
