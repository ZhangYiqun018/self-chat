from utils import compute_rouge, save_results
from datasets import load_dataset
import os
import concurrent.futures
from tqdm.auto import tqdm

origin = load_dataset(
    'json',
    data_files = os.path.join('data', 'data_seeds.json'),
    split      = 'train'
)

origin_datas = []

for data in origin:
    text = data['persona'] + data['situation']

    origin_datas.append(text)

machine_data = load_dataset(
    'json',
    data_files=os.path.join('data', 'machine_generate.json'),
    split='train'
)

filter_datas = []

for data in tqdm(machine_data):
    text = data['persona'] + data['situation']
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                compute_rouge,
                text,
                label
            ) for label in origin_datas
        ]

        scores = [f.result() for f in futures]

        if max(scores) <= 0.7:
            filter_datas.append(data)
            origin_datas.append(text)

save_results(filter_datas, 'filter_data.json')