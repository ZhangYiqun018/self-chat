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
        stop=tenacity.stop_after_attempt(3),
        reraise=True)
def get_api2d_response(
    url: str, 
    apikey: str, 
    content: str,
    _verbose: bool=False,
    temperature: float=0.7,
):
    headers = {
        'Authorization': f'Bearer {apikey}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    if _verbose:
        print(content)

    payload = json.dumps({
        "model": "gpt-3.5-turbo-0613",
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": temperature,
        "max_tokens" : 2000,
        "safe_mode"  : False
    })

    response = requests.request("POST", url, headers=headers, data=payload)
    response = eval(response.text)
    response = response['choices'][0]['message']['content']

    return response


@tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=200),
        stop=tenacity.stop_after_attempt(3),
        reraise=True)
def get_openai_response(
    url              : str,
    apikey           : str,
    content          : str,
    _verbose         : bool  = False,
    temperature      : float = 0.7,
    frequency_penalty: float = 0.0,
    presence_penalty : float = 0.0,
    use_16k          : bool  = False,
):
    openai.api_type = "openai"
    openai.api_base = url
    openai.api_key = apikey

    if _verbose:
        print(content)

    response = openai.ChatCompletion.create(
        # engine = "deployment_name"
        model = "gpt-3.5-turbo" if not use_16k else "ChatGPT16k",
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
        temperature       = temperature,
        max_tokens        = 3000 if not use_16k else 10000,
        top_p             = 0.95,
        frequency_penalty = frequency_penalty,
        presence_penalty  = presence_penalty,
    )
    response = response['choices'][0]['message']['content']

    return response


@tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=200),
        stop=tenacity.stop_after_attempt(3),
        reraise=True)
def get_azure_response(
    url              : str,
    apikey           : str,
    content          : str,
    _verbose         : bool  = False,
    temperature      : float = 0.7,
    frequency_penalty: float = 0.0,
    presence_penalty : float = 0.0,
    use_16k          : bool  = False,
):
    openai.api_type    = "azure"
    openai.api_base    = url
    openai.api_version = "2023-03-15-preview"
    openai.api_key = apikey

    if _verbose:
        print(content)

    response = openai.ChatCompletion.create(
        # engine = "deployment_name"
        engine = "gpt-35-turbo" if not use_16k else "ChatGPT16k",
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
        temperature       = temperature,
        max_tokens        = 3000 if not use_16k else 10000,
        top_p             = 0.95,
        frequency_penalty = frequency_penalty,
        presence_penalty  = presence_penalty,
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
    try:
        response = get_openai_response(
            url = "http://107.148.42.172:8000/v1",
            apikey="fk-zhangyiqun",
            _verbose=True,
            content="下面请你扮演心理医生的角色，疏导我的情绪，倾听我的问题。\n当然，我很愿意扮演心理医生的角色来倾听您的问题并疏导情绪。请告诉我您当前的情绪和问题，我会尽力提供支持和建议。"
        )
        print(response)
    except Exception as e:
        print(e)
    