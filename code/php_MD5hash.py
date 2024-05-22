# calculate php MD5 hash
import os
import hashlib
import shutil

# define file path
# source_path = r'E:\webshellgen\templates\original dataset\All_Datasets 2\All_Datasets\php_black'
# source_path = r'E:\webshellgen\templates\original dataset\AWVD数据集\黑\Php'
# source_path = r'E:\webshellgen\templates\original dataset\All_Datasets 2\All_Datasets\php_white'
# target_path = r'E:\webshellgen\templates\MD5filter dataset\php_black'
# target_path = r'E:\webshellgen\templates\MD5filter dataset\php_white'

# traverse all the php file in the given file path
for root, dirs, files in os.walk(source_path):
    for file in files:
        if file.endswith('.php'):
            # calculate MD5 hash
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                content = f.read()
            md5_hash = hashlib.md5(content).hexdigest()

            # construct file path and file name
            target_file_name = md5_hash + '.php'
            target_file_path = os.path.join(target_path, target_file_name)

            # check whether the target path exists
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            # check wheter the target file exists
            if os.path.exists(target_file_path):
                continue

            # rename and save files
            shutil.copy(file_path, target_file_path)