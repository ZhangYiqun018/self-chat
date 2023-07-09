import concurrent.futures
import configparser
import json
import os
import re

from datasets import concatenate_datasets, load_dataset
from tqdm.auto import tqdm

from utils import get_azure_response


def init_data(seeds_path: str, machine_path: str):
    machine_data = load_dataset(
        'json',
        data_files = os.path.join('data', seeds_path),
        split      = 'train'
    )

    mental_health_data = load_dataset(
        'json',
        data_files = os.path.join('data', machine_path),
        split      = 'train'
    )

    dataset = concatenate_datasets([machine_data, mental_health_data])

    templates = []

    fp = open(data_path, 'a')

    for data in tqdm(dataset):
        for flag in flags:
            template_data = template.format(
                flag           = flag,
                ai_persona     = ai_persona,
                user_persona   = data['persona'],
                user_situation = data['situation']
            )

            templates.append(template_data)

            temp = {
                'template'      : template_data,
                'ai_persona'    : ai_persona,
                'user_persona'  : data['persona'],
                'user_situation': data['situation']
            }

            fp.write(json.dumps(temp) + '\n')

def post_process(response):
    return response

def check_dialog_turns(response):
    pattern = r'\[A\]:|\[B\]:|\<A\>:|\<B\>:'
    matches = re.findall(pattern, response)
    count = len(matches)
    print(count)
    return count >= 10

def run(content):
    while True:
        response = get_azure_response(url, apikey, content)
        
        if check_dialog_turns(response):
            break

        response = post_process(response)

    return response

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    url    = config.get('AZURE', 'url')
    apikey = config.get('AZURE', 'apikey')


    template_path = os.path.join('templates', 'dialog_prompt.json')
    data_path = os.path.join('data', 'dialog_init_data_zh.json')
    seeds_path = 'machinehealth_seeds_zh.json'
    machine_path = 'machine_generate_mentalhealth_zh.json'
    result_path = os.path.join('data', 'machine_generate_dialog_zh.json')
    if 'zh' in result_path:
        flags = ["<A>", "<B>"]
    else:
        flags = ["[A]", "[B]"]

    with open(template_path, 'r') as r:
        template   = json.load(fp=r)['prompt_zh']
        ai_persona = json.load(fp=r)['ai_persona_zh']

    
    if not os.path.exists(data_path):
        init_data(
            seeds_path   = seeds_path,
            machine_path = machine_path
        )

    datas = load_dataset(
        'json',
        data_files = data_path,
        split      = 'train'
    )
    
    print(datas)

    fp = open(result_path, 'a')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                run,
                data['template']
            ) for data in datas
        ]

        with tqdm(total=len(futures)) as pbar:
            for future, data in zip(concurrent.futures.as_completed(futures), datas):
                try:
                    data['response'] = future.result()
                    fp.write(json.dumps(data) + '\n') 
                    pbar.update(1)
                except Exception as e:
                    print(e)
                    pbar.update(1)

    