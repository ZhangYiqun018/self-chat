from utils import get_azure_response
from datasets import load_dataset, concatenate_datasets
import os
from tqdm.auto import tqdm
import json
import concurrent.futures
import configparser
import re


template = """The conversation between [A] and [B]. [A] and [B] comply with the following control information. According to the control information, {flag} initiates the conversation.

## [A] persona:
{ai_persona}

## [B] persona:
{user_persona}

## [B] situation:
{user_situation}

## conversation:
"""

def init_data():
    machine_data = load_dataset(
        'json',
        data_files = os.path.join('data', 'machine_generate.json'),
        split      = 'train'
    )

    mental_health_data = load_dataset(
        'json',
        data_files = os.path.join('data', 'mental_health_persona.json'),
        split      = 'train'
    )

    dataset = concatenate_datasets([machine_data, mental_health_data])

    ai_persona = "I am PICA, the empathetic chatbot from the Neu Datamining Lab. With advanced algorithms, I am designed to understand and respond to human emotions with compassion and understanding."

    flags = ["[A]", "[B]"]

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
    pattern = r'\[A\]:|\[B\]:'
    matches = re.findall(pattern, response)
    count = len(matches)

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

    data_path = os.path.join('data', 'dialog_init_data.json')
    if not os.path.exists(data_path):
        init_data()

    datas = load_dataset(
        'json',
        data_files = data_path,
        split      = 'train'
    )
    
    print(datas)

    result_path = os.path.join('data', 'machine_generate_dialog.json')

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

    