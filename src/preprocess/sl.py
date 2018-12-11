import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree

from data.variable import Variable


def read_variables_from_folder(folder: Path) -> List[Variable]:
    variables = []
    if (folder / "model").exists():
        for file in (folder / "data").iterdir():
            model_file = folder / "model" / (file.stem + ".json")
            name_to_values = read_data_from_data_model_files(file, model_file)
            variables.extend(Variable.create_variable_list_from_dict(name_to_values))
    else:
        for file in (folder / "data").iterdir():
            if file.suffix == ".txt":
                name_to_values = Variable.create_variable_list_from_dict(read_data_from_text_file(file))
                variables.extend(name_to_values)

    return variables


def read_data_from_text_file(txt_file: Path) -> Dict[str, List[str]]:
    name_to_values: Dict[str, List[str]] = defaultdict(list)
    with txt_file.open("r") as f:
        num_types = int(f.readline().strip())
        f.readline()
        for num_type in range(num_types):
            semantic_type = f.readline().strip()

            semantic_type = "---".join(
                [part.split("/")[-1] for part in semantic_type.replace("#", "").split("|")])
            num_values = int(f.readline())
            for num_val in range(num_values):
                name_to_values[semantic_type].append(f.readline().split(" ", 1)[1])
            f.readline()
    return name_to_values


def read_data_from_data_model_files(data_file: Path, model_file: Path):
    col_to_name: Dict[str, str] = {}
    name_to_values: Dict[str, List[str]] = defaultdict(list)
    with model_file.open("r") as f:
        data = json.load(f)
        node_array = data["graph"]["nodes"]

        for node in node_array:
            if "userSemanticTypes" in node:
                semantic_object = node["userSemanticTypes"]
                name = node["columnName"]
                domain = semantic_object[0]["domain"]["uri"].split("/")[-1]
                predicate = semantic_object[0]["type"]["uri"].split("/")[-1]
                col_to_name[name] = domain + "---" + predicate

    col_to_values: Dict[str, List[str]] = {}
    if data_file.suffix == ".csv":
        col_to_values = read_data_from_csv(data_file)
    elif data_file.suffix == ".xml":
        col_to_values = read_data_from_xml(data_file)
    elif data_file.suffix == ".json":
        col_to_values = read_data_from_json(data_file)

    for col, values in col_to_values.items():
        name_to_values[col_to_name[col]] = values
    return name_to_values


def read_data_from_csv(csv_file: Path) -> Dict[str, List[str]]:
    col_to_values: Dict[str, List[str]] = defaultdict(list)
    with csv_file.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for header in row.iterkeys():
                if header:
                    col_to_values[header.replace(" ", "")].append(row[header])
    return col_to_values


def read_data_from_xml(xml_file: Path) -> Dict[str, List[str]]:
    col_to_values: Dict[str, List[str]] = defaultdict(list)
    xml_tree = ElementTree.parse(str(xml_file))
    root = xml_tree.getroot()
    for child in root:
        for attrib_name in child.attrib.keys():
            col_to_values[attrib_name].append(child.attrib[attrib_name])
        for attrib in child:
            col_to_values[attrib.tag].append(attrib.text)
    return col_to_values


def read_data_from_json(json_file: Path) -> Dict[str, List[str]]:
    col_to_values: Dict[str, List[str]] = defaultdict(list)
    with json_file.open("r") as f:
        json_array = json.load(f)
        for node in json_array:
            for field in node.keys():
                if isinstance(node[field], list):
                    for value in node[field]:
                        col_to_values[field].append(str(value))
                elif isinstance(node[field], dict):
                    for field1 in node[field].keys():
                        col_to_values[field1].append(str(node[field][field1]))
                else:
                    col_to_values[field].append(str(node[field]))
    return col_to_values
