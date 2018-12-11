from typing import List, Dict


class Variable:
    def __init__(self, semantic_type: str, values: List[str]):
        self.semantic_type: str = semantic_type
        self.values: List[str] = values

    @staticmethod
    def create_variable_list_from_dict(name_to_values: Dict[str, List[str]]) -> List['Variable']:
        return [Variable(name, values) for name, values in name_to_values.items()]
