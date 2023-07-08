import concurrent.futures
import configparser
import json
import os
import random
import re
import string
from typing import List

from datasets import load_dataset
from tqdm.auto import tqdm

from utils import compute_rouge, get_azure_response


def get_persona_template(persona_pool: List, num_pool: int):
    persona_template = []

    persona_pool = random.sample(persona_pool, k=num_pool)

    persona_template.extend(persona_pool)

    random.shuffle(persona_template)

    template = ""
    for idx, persona in enumerate(persona_template):
        template += f"{str(idx+1)}. {persona}\n"
    
    template += f"{str(idx+2)}. "

    return template

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

    return texts[0]

def save_results(results: List):
    path = os.path.join('data', 'mental_health_data.json')

    with open(path, 'a') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    template = """Combining the concepts of [persona] and [situation], the given text will be disassembled and expanded into persona and situation.

Requirement:
1. output should be writen in the first person
2. make appropriate associations to generate specific situations and personas, instead of vague descriptions.
3. output format: {'persona': 'your persona output here', 'situation': 'your situation output here'}

the concept of [persona]: The persona is the role or image that an individual presents in a social context based on societal expectations and self-construction.
the concept of [situation]: The situation refers to the specific environment or context in which an individual finds themselves, encompassing physical, social, and cultural factors, as well as tasks, challenges, and pressures.

input: """

    config = configparser.ConfigParser()
    config.read('config.ini')
    url    = config.get('AZURE', 'url')
    apikey = config.get('AZURE', 'apikey')

    dataset_persona_pool = load_dataset(
        'json',
        data_files = os.path.join('data', 'mental_health_persona.json'),
        split      = 'train'
    )
    persona_pool = dataset_persona_pool['persona']


    results = []

    start_num = len(persona_pool)

    with tqdm(total=len(persona_pool)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    get_azure_response,
                    url,
                    apikey,
                    template + persona + '\n'
                ) for persona in persona_pool
            ]

            results = [f.result() for f in futures]

            results = [post_process(result) for result in results]

            save_results(results)

            pbar.update(len(results))