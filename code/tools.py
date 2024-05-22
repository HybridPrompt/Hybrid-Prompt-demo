import os
import json
import random
from extra_module import *

def read_folder_files(folder_path):
    content = ""  # string var initial null
    file_count = 0 # file counter

    # traverse all the files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)

            if file_count == 0:
                content += "Here is an example:\n"
            else:
                content += "\nHere is another example:\n"
            # print file content
            with open(file_path, "r") as file:
                file_content = file.read()
                content += file_content

            file_count += 1

    return content

def is_folder_empty(folder_path):
    if not os.path.exists(folder_path):
        print("ERROR: File path not exist")
        return 1

    if not os.path.isdir(folder_path):
        print("ERROR: Not correctly input file path")
        return 1

    for root, dirs, files in os.walk(folder_path):
        if files:
            return 0

    return 1

'''
def remove_element(lst, element):
    return [x for x in lst if x != element]
'''

def Select_Encryptlist():
    # reserve 'String obfuscation' and 'String concatenation' element
    selected_elements = ['String obfuscation', 'String concatenation']

    # select from 'String ROT13 encryption' and 'String BASE encryption'element
    random_element = random.choice(['String ROT13 encryption', 'String BASE encryption'])
    selected_elements.append(random_element)

    # reserve 'String XOR encryption' element
    selected_elements.append('String XOR encryption')

    return selected_elements

def get_file_names(directory):
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_name = os.path.basename(file_path)
            file_names.append(file_name)
    return file_names


