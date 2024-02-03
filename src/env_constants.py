from pathlib import Path

ENVIRONMENT = 'Local'

PROJECT_PATH = Path(__file__).parent.parent.parent
# TRAIN_DATA_PATH = PROJECT_PATH / Path('data/train')
TRAIN_DATA_PATH = PROJECT_PATH / Path('data/animals_and_more')

TEST_DATA_PATH = PROJECT_PATH / Path('data/test')

