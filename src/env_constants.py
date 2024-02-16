from pathlib import Path

ENVIRONMENT = 'Local'

# ------------- Local computer
PROJECT_PATH = Path(__file__).parent.parent.parent
TRAIN_DATA_PATH = PROJECT_PATH / Path('data/train')
TEST_DATA_PATH = PROJECT_PATH / Path('data/test')

# -------------- RUNI server
# PROJECT_PATH = '/home/mlds_user/yaniv'
# TRAIN_DATA_PATH = PROJECT_PATH / Path('/home/mlds_user/yaniv/data/train')
# TEST_DATA_PATH = PROJECT_PATH / Path('/home/mlds_user/yaniv/data/test')


