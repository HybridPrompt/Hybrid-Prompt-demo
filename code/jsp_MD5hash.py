# for filtering jsp files
import os
import hashlib
import shutil

# set path
# source_path = r'E:\webshellgen\templates\original dataset\All_Datasets 2\All_Datasets\jsp_black'
# source_path = r'E:\webshellgen\templates\original dataset\AWVD数据集\黑\Jsp'
source_path = r'E:\webshellgen\templates\original dataset\All_Datasets 2\All_Datasets\jsp_white'
# target_path = r'E:\webshellgen\templates\MD5filter dataset\jsp_black'
target_path = r'E:\webshellgen\templates\MD5filter dataset\jsp_white'

# traversing all jsp files
for root, dirs, files in os.walk(source_path):
    for file in files:
        if file.endswith('.jsp'):
            # calculate MD5 hash
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                content = f.read()
            md5_hash = hashlib.md5(content).hexdigest()

            # construct target file path and file name
            target_file_name = md5_hash + '.jsp'
            target_file_path = os.path.join(target_path, target_file_name)

            # check the existence of path
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            # check the target file
            if os.path.exists(target_file_path):
                continue

            # rename the file and save to the target path
            shutil.copy(file_path, target_file_path)