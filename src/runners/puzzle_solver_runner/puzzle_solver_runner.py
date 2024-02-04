import argparse
from pathlib import Path
from src import env_constants
from src.runners.puzzle_solver_runner.puzzle_solver_flow import execute_infer_flow
from src.util_functions.util_functions import load_dict_from_json

if __name__ == '__main__':
    # ---  Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', type=str, help='Run configuration json file to use')
    parser.add_argument('--project_path', type=str, help='Project path.  The outputs folder will be created in here')
    parser.add_argument('--test_data', type=str, help='Test data folder path')
    args = parser.parse_args()

    default_test_config = Path(__file__).parent / Path('run_configs/patch_adj_inference_12x12_resize360.json')
    # default_test_config = Path(__file__).parent / Path('run_configs/patch_adj_inference_10x10_resize300.json')

    run_config_path = args.run_config if args.run_config is not None else default_test_config
    project_path = args.project_path if args.project_path is not None else env_constants.PROJECT_PATH
    test_data_path = args.test_data if args.test_data is not None else env_constants.TEST_DATA_PATH

    run_params = load_dict_from_json(run_config_path)

    execute_infer_flow(run_params, project_path, test_data_path)





