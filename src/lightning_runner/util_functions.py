import json
from pathlib import Path
from typing import Union


def save_dict_to_json(dictionary: dict, output_path: Union[str, Path]):
    with open(Path(output_path), 'w') as file:
        json.dump(dictionary, file)

