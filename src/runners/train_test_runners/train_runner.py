import argparse
from pathlib import Path
from src import env_constants
from src.trainer.trainer_modules.train_flow import execute_train_flow
from src.util_functions.util_functions import load_dict_from_json

if __name__ == '__main__':
    # ---  Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', type=str, help='Run configuration json file to use')
    parser.add_argument('--project_path', type=str, help='Project path.  The outputs folder will be created in here')
    parser.add_argument('--train_data', type=str, help='Train data folder path')
    parser.add_argument('--test_data', type=str, help='Test data folder path')
    args = parser.parse_args()

    default_test_config = Path(__file__).parent / Path(
        'run_configs/image_classification_vit_without_pos_emb/jigsaw_vitb16pretrainedpatch16_resize224_7x7_noPosEmb.json')

    # default_test_config = Path(__file__).parent / Path('run_configs/image_classification_vit/vit_custom_classification_template.json')
    # default_test_config = Path(__file__).parent / Path('run_configs/image_classification_vit/vit_pretrained_classification_template.json')

    # default_test_config = Path(__file__).parent / Path('run_configs/patch_adjacence_resnet18/patch_adgacence_template.json')
    # default_test_config = Path(__file__).parent / Path('run_configs/image_classification_resnet18/jigsaw_classification_template.json')

    run_config_path = args.run_config if args.run_config is not None else default_test_config
    project_path = args.project_path if args.project_path is not None else env_constants.PROJECT_PATH
    train_data_path = args.train_data if args.train_data is not None else env_constants.TRAIN_DATA_PATH
    test_data_path = args.test_data if args.test_data is not None else env_constants.TEST_DATA_PATH

    run_params = load_dict_from_json(run_config_path)

    execute_train_flow(run_params, project_path, train_data_path, test_data_path)





