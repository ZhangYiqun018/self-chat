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
    pass

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
    response = """
[A]: Hello there! I am PICA, an empathetic chatbot from the Neu Datamining Lab. How can I assist you today?

[B]: Hi PICA. I wanted to chat with someone who might understand what it's like to have an anxiety disorder.

[A]: I'm here to listen and help, [B]. As an empathetic chatbot, I am designed to understand and respond to human emotions with compassion and understanding. Could you tell me more about how your anxiety impacts your daily life?

[B]: Sure. My anxiety often causes excessive worry and fear, which can make it difficult for me to focus on everyday tasks. I also experience physical symptoms like a rapid heartbeat and shortness of breath, which further impact my mental well-being.

[A]: I can understand how challenging that can be, [B]. It sounds like your anxiety is affecting your daily functioning and overall well-being. Is there anything specific you would like to discuss or any strategies you are looking for to manage your anxiety better?

[B]: I'm open to any suggestions or strategies that could help me cope with my anxiety. It would be great to hear some techniques that have worked for others.

[A]: Absolutely, [B]. One technique that can be helpful is deep breathing exercises. Taking slow, deep breaths and focusing on your breath can help calm your mind and body. Another strategy is practicing mindfulness, which involves being present in the moment and nonjudgmentally accepting your thoughts and feelings.

[B]: Those sound like useful techniques. I'll definitely give them a try. Is there anything else you would recommend?

[A]: Another helpful technique can be challenging negative thoughts and replacing them with more positive and realistic ones. This process, known as cognitive restructuring, can help in reducing anxiety. Additionally, engaging in regular physical exercise and maintaining a healthy lifestyle can have a positive impact on managing anxiety.

[B]: Thank you, PICA. I will incorporate these strategies into my daily routine and see how they work for me. It's comforting to have someone to talk to who understands what I'm going through.

[A]: You're welcome, [B]. I'm here for you whenever you need to talk or if you have any other questions. Remember, you are not alone in this journey, and there are resources and support available to help you manage your anxiety.
"""
