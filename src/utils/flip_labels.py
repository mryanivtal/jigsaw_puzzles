import glob

import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    test_data_path = Path(r'D:\docs\Study\DSML_IDC\Final project\project\data\test')
    uncertain_folder = test_data_path / Path('uncertain')

    files = glob.glob(str(uncertain_folder / Path('*.jpg')))

    for file in files:
        filename = file.split('\\')[-1]

        assert len(filename.split('.')) == 3
        label, fid, ext = filename.split('.')
        label = 'dog' if label == 'cat' else 'cat'
        new_filename = '.'.join([label, fid, ext])

        old_path = uncertain_folder / Path(filename)
        new_path = uncertain_folder / Path(new_filename)

        if old_path.exists():
            old_path.rename(new_path)

        print()

