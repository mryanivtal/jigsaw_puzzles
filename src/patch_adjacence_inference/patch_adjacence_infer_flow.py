from pathlib import Path

from src import env_constants
from src.datasets.dogs_vs_cats_patch_infer_dataset import DogsVsCatsPatchInferDataset
from src.util_functions.util_functions import create_output_dir

if __name__ == '__main__':
    # TODO:Yaniv: continue from here!

    project_path = env_constants.PROJECT_PATH
    test_data_path = env_constants.TEST_DATA_PATH

    output_path = Path(project_path) / Path('output')
    run_name = 'patch__adj_inference'



    outputs_path = create_output_dir(project_path, run_name, add_timestamp=True)

    dataset = DogsVsCatsPatchInferDataset()