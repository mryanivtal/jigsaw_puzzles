from pathlib import Path

ENVIRONMENT = 'Local'

PROJECT_PATH = Path.cwd().parent.parent
TRAIN_DATA_PATH = PROJECT_PATH / Path('data/train')
TEST_DATA_PATH = PROJECT_PATH / Path('data/test')

