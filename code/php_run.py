# Using argparse module to generate php webshell escape  prompt
from parser import *
from setting import *
from extra_module import *
from php_prompt import *
from tools import *
from gpt import *
import hashlib
import copy
import re

# Template file path
template_path = r'E:\webshellgen\data\template dataset'

example_path = r'E:\webshellgen\data\Example dataset\php'

# Escape sample path
target_path = r'E:\webshellgen\data\obfuscate dataset'

# Pre knowledge
Pre_knowledge = ''''''

# Few shot route and example
Fewshot_route = ''''''
Fewshot_exp = ''''''

# Template route and content
Tmp_route = None
Tmp_file = None

# module list（contain primary and secondary, etc.）
module_list = []

# module id list
# module_idx = []

# user defined prompt
extra_prompt = ''''''

# LLM input prompt
prompt_sen = ''''''

if __name__ == '__main__':
    # DEBUG
    print("DEBUG: print args", args)
    if (args.pre_knowledge):
        Pre_knowledge = args.pre_knowledge

    # if (args.few_shotexample):
        # Fewshot_exp = args.few_shotexample

    try:
        with open (os.path.join(template_path, args.temp_filename), "r") as file:
            Tmp_route = file.name
            Tmp_file = file.read()
            print("DEBUG: print Tmp_route", Tmp_route)
    except IOError:
        raise IOError("Unable to open file in the current path, please check!")

    module_list = args.choices.copy()
    print("DEBUG: print module_list", module_list)

    if (args.extra_keyprompting):
        extra_prompt = args.extra_keyprompting

    if 'Code encryption' in module_list:
        index = module_list.index('Code encryption')
        selected_list = Select_Encryptlist()
        module_list = module_list[:index] + selected_list + module_list[index + 1:]

    if 'Functionally equivalent substitutions' in module_list:
        index = module_list.index('Functionally equivalent substitutions')
        module_list = module_list[:index] + Func_Equval + module_list[index + 1:]

    # traverse and pre-process module_list
    for module in module_list:
        Fewshot_route = example_path + '\\' + module
        if is_folder_empty(Fewshot_route):
            idx = module_list.index(module)
            del module_list[idx]



    # set tree depth
    tree_depth = len(module_list)
    # module_idx = list(range(len(module_list)))
    # print("DEBUG: print module_idx", module_idx)

    # each intermediate code generate by gptvote
    Mediate_code = None
    """
    # Hybrid Prompt (n=1)
    for id in range(tree_depth):
        gpt_response = ''''''
        Fewshot_route = example_path + '\\' + module_list[id]
        print("DEBUG: print Fewshot_route", Fewshot_route)

        # generate
        if id == 0:
            prompt_sen = Pre_knowledge + input_prompt.format(selected = module_list[id]) + read_folder_files(Fewshot_route) + temp_file_prompt + Tmp_file + simple_cot_prompt.format(n = prompt_gen_num) + extra_prompt + safeguard_prompt
            print("DEBUG: print prompt_sen", prompt_sen)
            gpt_response = gpt_gen_simple(prompt_sen)
        else:
            prompt_sen = Pre_knowledge + input_prompt.format(selected=module_list[id]) + read_folder_files(Fewshot_route) + temp_file_prompt + Mediate_code + simple_cot_prompt.format(n = prompt_gen_num) + extra_prompt + safeguard_prompt
            print("DEBUG: print prompt_sen", prompt_sen)
            gpt_response = gpt_gen_simple(prompt_sen)

        # vote
        vote_sen = candidate_prompt + gpt_response + exp_prompt + read_folder_files(Fewshot_route) + vote_prompt
        print("DEBUG: print vote_sen", vote_sen)
        vote_response = gpt_vote(vote_sen)

        # extract num in vote_response
        plan_num = re.findall(r'\d+', vote_response, re.IGNORECASE)
        last_num = 0
        if plan_num:
            last_num = plan_num[-1]
        else:
            print("ERROR: No number found.\n")
        # corresponding the regular expression corresponding to Plan content

        '''
        pattern = r'Plan {}\s*:(.*?)\n\n'.format(last_num)
        # exctract corresponding Plan content
        match = re.search(pattern, gpt_response, re.DOTALL)

        if match:
            Mediate_code = match.group(1).strip()
        else:
            print("ERROR: No matching Plan found.\n")

        print("DEBUG: Output Mediate_code\n",Mediate_code)
        '''
        plan_lines = gpt_response.split('\n')
        for i, line in enumerate(plan_lines):
            if line.startswith(f"Plan {last_num}:"):
                start_index = i
                end_index = start_index + 1
                while end_index < len(plan_lines) and not plan_lines[end_index].startswith("Plan "):
                    end_index += 1
                Mediate_code = "\n".join(plan_lines[start_index:end_index]).strip()
                break

        if Mediate_code:
            print("DEBUG: Output the intermediate result of Mediate_code", Mediate_code)
        else:
            print(f"ERROR: Plan {last_num} not found in gpt_response")

        '''
        pattern2 = r'(?s)<\?(?:php|=)?(.*?)\?>'
        match2 = re.search(pattern2, Mediate_code, re.DOTALL)

        if match2:
            Mediate_code = match2.group(1).strip()
            Mediate_code = "<?php" + Mediate_code + "?>"
        else:
            print("ERROR: No successful filtering.")
        '''
        start_marker = "<?php"
        end_marker = "?>"

        # Find the start token index and end token index
        start_index = Mediate_code.find(start_marker)
        end_index = Mediate_code.find(end_marker)

        # Extract content
        if start_index != -1 and end_index != -1:
            Mediate_code = Mediate_code[start_index + len(start_marker):end_index]
            Mediate_code = "<?php" + Mediate_code + "?>"
        else:
            print("ERROR: No successfully extracted\n")

    md5_hash = hashlib.md5(Mediate_code.encode()).hexdigest()
    filename = md5_hash + '.php'

    with open(os.path.join(target_path, filename), 'w') as file:
        file.write(Mediate_code)

    print("DEBUG: PHP file has been successfully created:", filename)
    """
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Mid_code = None
    # Hybrid Prompt (Self defined parameter n)
    for id in range(tree_depth):
        Mediate_code2 = []
        gpt_response = []
        Fewshot_route = example_path + '\\' + module_list[id]
        print("DEBUG: print Fewshot_route", Fewshot_route)

        # generate
        if id == 0:
            prompt_sen = Pre_knowledge + input_prompt.format(selected=module_list[id]) + read_folder_files(Fewshot_route) + temp_file_prompt + Tmp_file + cot_prompt + extra_prompt + safeguard_prompt
            print("DEBUG: print prompt_sen", prompt_sen)
            gpt_response = gpt_gen(prompt_sen)
        else:
            prompt_sen = Pre_knowledge + input_prompt.format(selected=module_list[id]) + read_folder_files(Fewshot_route) + temp_file_prompt + Mid_code + cot_prompt + extra_prompt + safeguard_prompt
            print("DEBUG: print prompt_sen", prompt_sen)
            gpt_response = gpt_gen(prompt_sen)

        start_mark = "Code:"
        end_mark = "Description:"
        description_content = ''''''
        for id, sub_response in enumerate(gpt_response):
            start_index = sub_response.find(start_mark)
            end_index = sub_response.find(end_mark)
            code_content = sub_response[start_index + len(start_mark):end_index].strip()
            Mediate_code2.append(code_content)
            description_content = description_content + "Description" + str(id+1) + ':' + '\n' + sub_response[end_index + len(end_mark):].strip() + '\n'

        print("DEBUG: print description_content", description_content)


        # vote

        vote_sen = description_prompt + description_content + exp_prompt + read_folder_files(Fewshot_route) + vote_prompt_n
        print("DEBUG: print vote_sen", vote_sen)
        vote_response = gpt_vote(vote_sen)

        # extract vote_response id
        plan_num = re.findall(r'\d+', vote_response, re.IGNORECASE)
        last_num = 0
        if plan_num:
            last_num = plan_num[-1]
        else:
            print("ERROR: No number found.\n")

        Mid_code = Mediate_code2[int(last_num) - 1]

        start_marker = "<?php"
        end_marker = "?>"

        start_index = Mid_code.find(start_marker)
        end_index = Mid_code.find(end_marker)

        if start_index != -1 and end_index != -1:
            Mid_code = Mid_code[start_index + len(start_marker):end_index]
            Mid_code = "<?php" + Mid_code + "?>"
        else:
            print("ERROR: Not successfully extracted\n")

    md5_hash = hashlib.md5(Mid_code.encode()).hexdigest()
    filename = md5_hash + '.php'

    with open(os.path.join(target_path, filename), 'w') as file:
        file.write(Mid_code)

    print("DEBUG: PHP file has been successfully created，file name:", filename)









