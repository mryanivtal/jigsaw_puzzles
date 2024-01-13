import argparse
from pathlib import Path
from src import env_constants
from src.jigsaw_trainer.trainer.execute_experiment import execute_experiment
from src.util_functions.util_functions import load_dict_from_json

if __name__ == '__main__':
    # ---  Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', type=str, help='Run configuration json file to use')
    parser.add_argument('--project_path', type=str, help='Project path.  The outputs folder will be created in here')
    parser.add_argument('--train_data', type=str, help='Train data folder path')
    parser.add_argument('--test_data', type=str, help='Test data folder path')
    args = parser.parse_args()

    run_config_path = args.run_config if args.run_config is not None else (Path(__file__).parent / Path(
        'run_configs/image_classification/test_run.json'))
    project_path = args.project_path if args.project_path is not None else env_constants.PROJECT_PATH
    train_data_path = args.train_data if args.train_data is not None else env_constants.TRAIN_DATA_PATH
    test_data_path = args.test_data if args.test_data is not None else env_constants.TEST_DATA_PATH

    run_params = load_dict_from_json(run_config_path)

    execute_experiment(run_params, project_path, train_data_path, test_data_path)





