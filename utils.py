import concurrent.futures
import configparser
import json
import random
import os

import openai
import requests
import tenacity
from rouge_score import rouge_scorer


@tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=200),
        stop=tenacity.stop_after_attempt(10),
        reraise=True)
def get_api2d_response(url: str, apikey: str, content: str):
    headers = {
        'Authorization': f'Bearer {apikey}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    payload = json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "safe_mode": False
    })

    response = requests.request("POST", url, headers=headers, data=payload)
    response = eval(response.text)
    response = response['choices'][0]['message']['content']

    return response

@tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=200),
        stop=tenacity.stop_after_attempt(2),
        reraise=True)
def get_azure_response(url: str, apikey: str, content: str, _verbose: bool = False):
    openai.api_type    = "azure"
    openai.api_base    = url
    openai.api_version = "2023-03-15-preview"
    openai.api_key = apikey

    if _verbose:
        print(content)

    temperature_list = [0.0, 0.1, 0.2, 0.3]
    temperature = random.sample(temperature_list, k=1)[0]
    # print(temperature)
    response = openai.ChatCompletion.create(
        # engine = "deployment_name"
        engine = "gpt-35-turbo",
        messages = [
            {
                "role"   : "system",
                "content": "You are an AI assistant that helps people find information."
            },
            {
                "role"   : "user",
                "content": content
            }
        ],
        temperature = temperature,
        max_tokens  = 800,
        top_p       = 0.95,
    )

    response = response['choices'][0]['message']['content']

    return response

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    
    scores = scorer.score(
        target=references,
        prediction=predictions
    )

    return scores['rougeL'].fmeasure

def save_results(results, filename: str):
    path = os.path.join('data', filename)
    fp = open(path, 'a')
    for result in results:
        fp.write(json.dumps(result) + '\n')

if __name__ == '__main__':
    urls = [
        "https://oa.api2d.net/v1/chat/completions",
        "https://openai.api2d.net/v1/chat/completions",
        "https://stream.api2d.net/v1/chat/completions"
    ]

    apikey = 'fk189078-yD7FKy1YlGCriUZxxfJhWzo026etYqpt'

    contents = ["response 1", "response 2", "response 3"]


    config = configparser.ConfigParser()
    config.read('config.ini')


    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = []

        for url, content in zip(urls, contents):
            url = config.get('AZURE', 'url')
            apikey = config.get('AZURE', 'apikey')

            print(url, apikey)

            future = executor.submit(get_azure_response, url=url, apikey=apikey, content=content)

            results.append(future)
        
        for result in concurrent.futures.as_completed(results):
            print(result.result())
