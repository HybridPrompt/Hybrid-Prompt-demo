# virustotal api scanning content
# www.virustotal.com/vtapi/v2/

import requests
import json
import os
from tools import *

# Virustotal report path
report_path = r'E:\webshellgen\result'

def getFileScanId(url, apikey, a, b):
    # /file/scan
    # limited by 32MB
    params = {'apikey': apikey}
    files = {'file': (a, open(b, 'rb'))}
    response = requests.post(url, files=files, params=params)
    my_scan_id = str(response.json()['scan_id'])
    return my_scan_id

def getFileReportResult(url, apikey, my_scan_id, report_name):
    # /file/report
    # scanning file report
    # resource can be MD5，SHA-1，SHA-256。
    get_params = {'apikey': apikey, 'resource': my_scan_id,'allinfo': '1'}
    response2 = requests.get(url, params=get_params)
    jsondata = json.loads(response2.text)
    with open(os.path.join(report_path, report_name), "w") as f:
        json.dump(jsondata, f, indent=4)
    return jsondata

def getResult(json, report_txt_name):
    result = {}
    Final_result = False
    for k,v in json["scans"].items():
        result[k] = v['detected']
    print(result)
    for value in result.values():
        if value:
            Final_result = True
            break
    print("contain {0} files".format(len(result)))
    print("final scanning results", Final_result)
    with open(os.path.join(report_path, report_txt_name), "w") as g:
        g.write(str(result))

def main():
    # file_name = input("please input filename:")
    apikey = "XXXXXX"
    file_src = r'E:\webshellgen\data\obfuscate dataset'
    # os.chmod(file_src, 0o755)

    file_names = get_file_names(file_src)
    for file_name in file_names:
        a = str(file_name)
        b = str(file_src)
        b = os.path.join(b, a)
        print("DEBUG: print the corresponding scan file name", a)
        # v2 api
        url1 = 'https://www.virustotal.com/vtapi/v2/file/scan'
        url2 = "https://www.virustotal.com/vtapi/v2/file/report"

        # v3 api
        # new_url1 = 'https://www.virustotal.com/api/v3/files'
        # new_url2 = ''

        # get file scan_id
        scan_id = getFileScanId(url1, apikey, a, b)
        # get return json result and write to result
        # getFileReportResult(url2, apikey, scan_id)
        rp_name = a.replace(".php", ".json")
        rp_txt_name = a.replace(".php", ".txt")
        json = getFileReportResult(url2, apikey, scan_id, rp_name)
        getResult(json, rp_txt_name)

if __name__ == '__main__':
    main()


