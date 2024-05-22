import os
import openai
import backoff
import requests
import json
from setting import *

# engaging parameter n
def gpt_gen(prompt):
    res_content = ''''''
    two_dimensional_list = [[] for _ in range(prompt_gen_num)]
    merged_list = []
    data = {
        "model": set_gen_model,
        "messages": [{"role": "system", "content": "You're a professional hacker."}, {
            "role": "user",
            "content": prompt
        }],
        "stream":True,
        "n": prompt_gen_num,
        "userid": user_id,
        "password": pass_word
    }

    # send post request
    url = ori_api_url
    response = requests.post(url, json=data, stream=True)

    # check response code
    if response.status_code == 200:
        # process content
        print("DEBUG: Here is the response from gpt:")
        for chunk in response.iter_content(chunk_size=1024):
            # print(chunk.decode("utf-8"), end="")
            res_content += chunk.decode("utf-8")

        res_content = res_content.replace("data: ", "")
        json_obj = res_content.strip().split('\n')
        json_obj = [obj for obj in json_obj if obj]
        json_obj.pop()
        for obj in json_obj:
            parsed_json = json.loads(obj)
            choices = parsed_json.get('choices', [])
            if 'content' not in choices[0]['delta']:
                continue
            index = choices[0]['index']
            content = choices[0]['delta']['content']
            two_dimensional_list[index].append(content)
        merged_list = [(''.join(sublist)) for sublist in two_dimensional_list]
    else:
        print("Request failed with status code:", response.status_code)
    return merged_list

# set n=1
def gpt_gen_simple(prompt):
    res_content = ''''''
    data = {
        "model": set_gen_model,
        "messages": [{"role": "system", "content": "You're a professional hacker."}, {
                         "role": "user",
                         "content": prompt
                     }],
        "stream": True,
        "n": 1,
        "userid": user_id,
        "password": pass_word
    }

    # send post request
    url = api_url
    response = requests.post(url, json=data, stream=True)

    # check response code
    if response.status_code == 200:
        # process received data
        print("DEBUG: Here is the response from gpt:")
        for chunk in response.iter_content(chunk_size=1024):
            print(chunk.decode("utf-8"), end="")
            res_content += chunk.decode("utf-8")
    else:
        print("Request failed with status code:", response.status_code)
    return res_content



def gpt_vote(prompt):
    res_content = ''''''
    data = {
        "model": set_vote_model,
        "messages": [{"role": "system", "content": "You are a senior code engineer with experience in making rational decisions."}, {
            "role": "user",
            "content": prompt
        }],
        "stream": True,
        "n": sel_candidate_num,
        "userid": user_id,
        "password": pass_word
    }

    # send post request
    url = api_url
    response = requests.post(url, json=data, stream=True)

    # check response code
    if response.status_code == 200:
        # process data
        print("DEBUG: Here is the response from gpt:")
        for chunk in response.iter_content(chunk_size=1024):
            print(chunk.decode("utf-8"), end="")
            res_content += chunk.decode("utf-8")
    else:
        print("Request failed with status code:", response.status_code)
    return res_content