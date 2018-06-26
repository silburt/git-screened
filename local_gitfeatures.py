import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import json
from subprocess import call
import os
import glob

def get_readme_length(repo, GProfile):
    readme = glob.glob('%s/README*'%repo)
    #print(readme, len(readme))
    if len(readme) >= 1:
        try:
            text = open(readme[0], 'r', encoding='utf-8').read()
            while "\n\n" in text:
                text = text.replace("\n\n","\n")
            GProfile.readme_lines = len(text.splitlines())
        except:
            print('couldnt get readme')

# get frequency of comments vs. code
def get_comment_code_ratio(text, GProfile):
    print("HI")
    for sym_s, sym_e in [('#','\n')]:
        start = text.find(sym_s)
        end = text.find(sym_e, start + 1)
        comm_len = 0
        while start != -1:
            comm_len += len(text[start: end].split('\n'))
            start = text.find(sym_s, end + 1)
            end = text.find(sym_e, start + 1)

    print("HI2")
    for sym_s, sym_e in [('"""', '"""')]:
        start = text.find(sym_s)
        end = text.find(sym_e, start + 1)
        doc_len = 0
        while start != -1:
            doc_len += len(text[start: end].split('\n'))
            start = text.find(sym_s, end + 1)
            end = text.find(sym_e, start + 1)

    print("HI3")
    GProfile.comment_lines += comm_len
    GProfile.docstring_lines += doc_len

# get summary stats of pep8 errors
def get_pep8_errs(file, GProfile, show_source = True):
    name = file.split('.py')[0]
    
    call("pycodestyle --statistics -qq %s.py > %s.txt"%(name, name), shell=True)
    errs = open('%s.txt'%name, 'r').read().splitlines()
    for err in errs:
        val, label = err.split()[0], err.split()[1]
        label = label[0:2] # remove extra details
        GProfile.pep8[label] += int(val)


