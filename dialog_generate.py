import concurrent.futures
import configparser
import json
import os
import re

from datasets import concatenate_datasets, load_dataset
from tqdm.auto import tqdm

from utils import get_api2d_response, get_azure_response, get_openai_response


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
        template_data = template.format(
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

        fp.write(json.dumps(temp, ensure_ascii=False) + '\n')

def post_process(conversation):
    fp = open('train_psyqa.json', 'a')
    history = []
    for conv in conversation:
        prompt = re.sub(r'^[^a-zA-Z\u4e00-\u9fff]+', '', conv[0])
        reply = re.sub(r'^[^a-zA-Z\u4e00-\u9fff]+', '', conv[1])
        if len(reply) == 0:
            return False
        if len(history) == 0 and (reply== history[-1][1] or prompt == history[-1][0]):
            return False
        data = {
            'prompt'  : prompt,
            'response': reply,
            'history' : history
        }
        
        fp.write(
            json.dumps(data, ensure_ascii=False) + '\n'
        )

        history.append([prompt, reply])
    
    return True

def check_dialog_turns(text):
    if "<Round 10>" not in text:
        return False

    pattern = r"<[AB]>(.+?)\n(?:<[AB]>(.+?)\n)*"
    conversation = re.findall(pattern=pattern, string=text)
    if len(conversation) < 10:
        return False
    
    # return post_process(conversation)
    return True

def run(content):
    while True:
        if api_style == 'azure':
            response = get_azure_response(
                url if not use_16k else url16k, 
                apikey if not use_16k else apikey16k, 
                content           = content,
                temperature       = 0.1,
                _verbose          = False,
                frequency_penalty = 0.6,
                use_16k           = use_16k,
            )
        elif api_style == 'api2d':
            response = get_api2d_response(
                url,
                apikey,
                content  = content,
                _verbose = False,
            )
        elif api_style == 'openai':
            response = get_openai_response(
                url, 
                apikey,
                content           = content,
                temperature       = 0.1,
                _verbose          = False,
                frequency_penalty = 0.6,
                use_16k           = use_16k,
            )
        if check_dialog_turns(response):
            break
        else:
            print(response)
            continue

    return response

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    api_style = 'azure'
    use_16k   = False
    if api_style == 'azure':
        print("Use AZURE")
        url    = config.get('AZURE', 'url')
        apikey = config.get('AZURE', 'apikey')
        url16k    = config.get('AZURE', 'url16k')
        apikey16k = config.get('AZURE', 'apikey16k')
    elif api_style == 'api2d':
        print("Use API2D")
        url       = config.get('API2D', 'url')
        apikey    = config.get('API2D', 'apikey')
    elif api_style == 'openai':
        print("Use OPENAI")
        url       = config.get('OPENAI', 'url')
        apikey    = config.get('OPENAI', 'apikey')

    seeds_path    = 'mentalhealth_seeds_zh.json'
    machine_path  = 'machine_generate_mentalhealth_zh.json'
    template_path = os.path.join('templates', 'dialog_prompt.json')
    
    data_path     = os.path.join('data', 'psyqa_data.json')
    result_path   = os.path.join('data', 'machine_generate_dialog_psyqa.json')

    with open(template_path, 'r') as r:
        template_data = json.load(fp=r)
        print(template_data['prompt_zh'])
        template   = template_data['prompt_zh']
        ai_persona = template_data['ai_persona_zh']

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
    print(datas[0]['template'])

    fp = open(result_path, 'a')
    with concurrent.futures.ThreadPoolExecutor() as executor:
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
                    fp.write(json.dumps(data, ensure_ascii=False) + '\n') 
                    pbar.update(1)
                except Exception as e:
                    print(e)
                    pbar.update(1)