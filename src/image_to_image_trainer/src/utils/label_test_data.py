import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    csv_path = Path(__file__).parent / Path('sample_predictions_log.csv')
    test_data_path = Path(r'/data/test')
    uncertain_folder = test_data_path / Path('uncertain')
    uncertain_folder.mkdir(exist_ok=True)

    labels = pd.read_csv(csv_path)
    labels['uncertain'] = labels['probability'].apply(lambda x: True if 0.08 < x < 0.92 else False)

    uncertain_dogs = len(labels[(labels['uncertain'] == True) & (labels['prediction'] == 1)])
    uncertain_cats = len(labels[(labels['uncertain'] == True) & (labels['prediction'] == 0)])
    dogs = len(labels[(labels['uncertain'] == False) & (labels['prediction'] == 1)])
    cats = len(labels[(labels['uncertain'] == False) & (labels['prediction'] == 0)])

    print(f'uncertain_dogs: {uncertain_dogs}')
    print(f'uncertain_cats: {uncertain_cats}')
    print(f'dogs: {dogs}')
    print(f'cats: {cats}')

    for _, row in tqdm(labels.iterrows(), total=len(labels)):
        filename = row['path'].split('/')[-1]
        label = row['prediction_str']
        uncertain = row['uncertain']

        assert len(filename.split('.')) == 2

        new_name = '.'.join(([label] + filename.split('.')))
        if uncertain:
            new_path = test_data_path / uncertain_folder / Path(new_name)
        else:
            new_path = test_data_path / Path(new_name)

        file_path = test_data_path / Path(filename)

        if file_path.exists():
            file_path.rename(new_path)

        print()

