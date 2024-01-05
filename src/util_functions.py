from datetime import datetime
from pathlib import Path

def create_output_dir(parent_path, run_name: str, add_timestamp: bool) -> str:
    """
    Creates output path based on run name and timestamp and returns path string
    :param parent_path: name of project path to create the output folder in
    :param run_name: str: name of run / config
    :param add_timestamp: bool: add timestamp to folder name if true
    :return: output folder path string
    """
    if add_timestamp:
        timestamp_str = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
        relative_output_dir = Path(f'{run_name}_{timestamp_str}')
    else:
        relative_output_dir = Path(f'{run_name}')

    output_path = Path(parent_path) / Path('outputs') / relative_output_dir

    output_path.mkdir(parents=True, exist_ok=True)
    print(f'Run output path: {output_path}')

    return str(output_path)