# Author: Mingrui Ma
from extra_module import *
import argparse

# Construct ArgumentParser

parser = argparse.ArgumentParser(description='Generate php webshell prompt')

# add pre-knowledge available parameter
parser.add_argument('-p', '--pre_knowledge', type=str, help='Pre-knowledge Module (If Necessary)')

# add Few-shot example available parameter
# parser.add_argument('-e', '--few_shotexample', type=str, help='Few-shot Example (If Necessary)')

# add php template parameter
parser.add_argument('temp_filename', type=str, help='File name of the template file (Required!)')

# add webshell escape keywords
parser.add_argument('--choices', nargs='+', choices=choices_list, help='Specify choices')

# add extra key prompts
parser.add_argument('-k', '--extra_keyprompting', type=str, help='Available key prompting keywords (Optional)')

# parse
args = parser.parse_args()



