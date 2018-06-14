# This is the general script used to download github repos
import os
import numpy as np
from requests.auth import HTTPBasicAuth
import requests
import json

auth = open("utils/auth.txt").read()
username, pw = auth.split()[0], auth.split()[1]

def main(dir, max_size):
    repos = open('repos/machine-learning_repos.txt', 'r').read().splitlines()
    for r in repos[0:100]:
        try:
            s = requests.get(r,
                             headers={"Accept":"application/vnd.github.mercy-preview+json"},
                             auth=HTTPBasicAuth(username, pw))
            summary = json.loads(s.text or s.content)
            if summary['language'] == 'Python' and summary['size'] < max_size:
                loc = r.split('repos/')[1]
                folder = loc.split('/')[-1]
                os.system('mkdir %s/%s'%(dir, folder))
                os.system('git clone https://github.com/%s %s/%s/'%(loc, dir, folder))
            else:
                print('skipped %s, language=%s, size=%.2f'%(r, summary['language'], summary['size']/float(max_size)))
        except:
            print('couldnt process %s'%r)

if __name__ == '__main__':
    max_size = 100000
    dir = 'repos'
    main(dir, max_size)
