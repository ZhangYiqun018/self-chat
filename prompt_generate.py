import concurrent.futures
import configparser
import json
import os
import random
import re
from typing import List, Optional

from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm.auto import tqdm

from utils import get_azure_response

template="""[persona]: The persona is the role or image that an individual presents in a social context based on societal expectations and self-construction.
[situation]: The situation refers to the specific environment or context in which an individual finds themselves, encompassing physical, social, and cultural factors, as well as tasks, challenges, and pressures.
Combining the concepts of [persona] and [situation], come up with a series of diversified personas and situation:
{prompt}
"""

def make_template(
    template: str,
    origin: List,
    machine: List,
    num_seed: int,
    num_machine: int,
) -> str:
    origin = random.sample(origin, k=num_seed)
    machine = random.sample(machine, k=num_machine)

    datas = origin + machine
    prompt = ""
    for idx, data in enumerate(datas):
        prompt += f"{str(idx+1)}. {data}\n"

    prompt += f"{str(idx+2)}. "

    return template.format(prompt=prompt)

def save_results(results: List[str]):
    path = os.path.join('data', 'machine_generate.json')
    fp = open(path, 'a')
    for result in results:
        fp.write(json.dumps(result) + '\n')

def find_word_in_string(w, s):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)

def post_process(text):
    raw_texts = re.split(r"\n\d+\s?\. ", text)
    texts = []

    for text in raw_texts:
        text = re.sub(r"\s+", " ", text).strip()
        text = text.strip().capitalize()
        if text == "":
            continue

        if any(find_word_in_string(word, text) for word in ["i'm sorry", "i am sorry", "I am sorry", "I'm sorry"]):
            continue
        
        try:
            persona_match = re.search(r"'persona': '([^']+)'", text)
            persona = persona_match.group(1) if persona_match else ""

            situation_match = re.search(r"'situation': '([^']+)'", text)
            situation = situation_match.group(1) if situation_match else ""
            
            if persona == "" or situation == "":
                continue

            text = {
                'persona': persona,
                'situation': situation
            }
        except:
            continue

        texts.append(text)

    return texts

def process_dataset(dataset: Dataset) -> List:
    results = []
    for data in dataset:
        data = {
            'persona': data['persona'],
            'situation': data['situation']
        }
        results.append(data)

    return results

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    url    = config.get('AZURE', 'url')
    apikey = config.get('AZURE', 'apikey')

    dataset = load_dataset(
        'json',
        data_files = os.path.join('data', 'data_seeds.json'),
        split      = 'train'
    )
    origin = process_dataset(dataset=dataset)

    machine_dataset_path = os.path.join('data', 'machine_generate.json')
    if os.path.exists(machine_dataset_path):
        machine_dataset = load_dataset(
            'json',
            data_files=machine_dataset_path,
            split='train'
        )
        machine = process_dataset(machine_dataset)
    else:
        machine = []

    num_seed = 1
    num_machine = 2

    num_generate = 20000

    results = []

    with tqdm(total=num_generate) as pbar:
        while True:
            content = make_template(
                template    = template,
                origin      = origin,
                machine     = machine,
                num_seed    = num_seed,
                num_machine = num_machine
            )
            responses = get_azure_response(
                url     = url,
                apikey  = apikey,
                content = content
            )

            responses = post_process(responses)
            
            # extend machine pools
            machine.extend(responses)

            save_results(responses)

            pbar.update(len(responses))
            
            if len(machine) >= num_generate:
                break