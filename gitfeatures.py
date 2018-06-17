import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import json

# base directory is from ./run.py
auth = open('utils/auth.txt').read()
username, pw = auth.split()[0], auth.split()[1]

def scrape_readme_single(dir, r):
    git_path = r.split('repos/')[1]
    url = 'https://raw.githubusercontent.com/%s/master/README'%git_path
    try:
        readme = requests.get('%s.md'%url,
                              headers={"Accept":"application/vnd.github.mercy-preview+json"},
                              auth=HTTPBasicAuth(username, pw))
        if(readme.ok):
            readme = readme.text
        else:
            readme = requests.get('%s.rst'%url,
                                  headers={"Accept":"application/vnd.github.mercy-preview+json"},
                                  auth=HTTPBasicAuth(username, pw)).text
          
            # write to file
            readme_corpus = open('%s/%s.txt'%(dir, git_path.split('/')[1]), 'w')
            readme_corpus.write(str(readme.encode('utf8')))
            readme_corpus.close()
    except:
        print('No README file for %s. Skipping'%r)

# get frequency of desired packages in python code
def get_package_freq(text, GProfile):
    for p in GProfile.packages.keys():
        if text.find(p) != -1:    # is package even in script at all?
            
            # direct import, e.g. 'import numpy (as np)'
            string = 'import %s'%p
            loc = text.find(string)
            if loc != -1:
                # last word in line is call_name
                call_name = text[loc:].partition('\n')[0].split()[-1].split('\n')[0].split('\\')[0]
                GProfile.packages[p] += text.count('%s.'%call_name)
            
            # indirect import, e.g. 'from numpy.X import vectorize (as vec)'
            string_from = 'from %s'%p
            loc_from = text.find(string_from)
            ext = ['(', '.']  # check both X() and X.y()
            while loc_from != -1: # iterate same package imported mult. times on different lines
                try:
                    call_names = text[loc_from:].partition('\n')[0].split(' import ')[1]
                    if call_names.count(',') != 0:  # iterate mulitple imports on same line
                        call_names = call_names.replace('(','').replace(')','').replace(" ", "").split(',')
                        for call_name in call_names:
                            if call_name == "": # skip bad entries
                                continue
                            GProfile.packages[p] += sum(text.count('%s%s'%(call_name, x)) for x in ext)
                    else:
                        call_name = call_names.split()[-1]
                        GProfile.packages[p] += sum(text.count('%s%s'%(call_name, x)) for x in ext)
                except:
                    pass
                loc_from = text.find(string_from, loc_from + 1) # get next instance, e.g. 'from numpy.X import b'

# get frequency of comments vs. code
def get_comment_freq(text, GProfile):
    for sym_s, sym_e in [('#','\n'), ('"""', '"""')]:
        start = text.find(sym_s)
        end = text.find(sym_e, start + 1)
        while start != -1:
            GProfile.comment_words += len(text[start: end].split())
            start = text.find(sym_s, end + 1)
            end = text.find(sym_e, start + 1)

    allowed_characters = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    GProfile.code_words += len(''.join(c for c in text if c in allowed_characters ).split())

# get distribution of commits over time
def get_repo_commit_history(commits_url, GProfile):
    try:
        r = requests.get(commits_url, headers={"Accept":"application/vnd.github.mercy-preview+json"},
                     auth=HTTPBasicAuth(username, pw))
        commits = json.loads(r.text or r.content)
        for commit in commits:
            date_string = commit['commit']['author']['date']
            date = datetime.strptime(date_string.split("T")[0],'%Y-%m-%d')
            GProfile.commit_history.append(date)
    except:
        print('couldnt get commit history for %s'%commits_url)

