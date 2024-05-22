import requests
import json
from tools import *

# prepare json data
n = 3
res_content = ''''''
two_dimensional_list = [[] for _ in range(n)]
merged_list = []
data = {
        "model":"gpt-3.5-turbo",
        "messages":[{"role": "system", "content": ""},{
            "role":"user",
            "content":'''Write a function that implements the sum of two numbers using the python language, your output should be the following format:
                        Code:
                        
                        Plan1:
                        Code corresponding to plan 1.
                        
                        Plan2:
                        Code corresponding to plan 2.
                        
                        ...
                        
                        PlanN:
                        Code corresponding to plan N.
                        '''

        }],
        "stream":True,
        "n":3,
        "userid":"jkpathfinder",
        "password":"tian20007188"
    }

# send post request
url = "https://www.hustgpt.com/api/OriginAPI"
response = requests.post(url, json=data, stream=True)

# check response status
if response.status_code == 200:
    # process response content
    for chunk in response.iter_content(chunk_size=1024):
        res_content += chunk.decode("utf-8")
        # print(chunk.decode("utf-8"), end=" kkk")
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

    # print("DEBUG: print two_dimensional_list", two_dimensional_list)
    merged_list = [(''.join(sublist)) for sublist in two_dimensional_list]
    print("DEBUG: print merged_list[0]\n", merged_list[0])
    print("DEBUG: print merged_list[1]\n", merged_list[1])
    print("DEBUG: print merged_list[2]\n", merged_list[2])
else:
    print("Request failed with status code:", response.status_code)