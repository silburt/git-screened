import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import json
from subprocess import call
import os

# base directory is from ./run.py
auth = open('utils/auth.txt').read()
username, pw = auth.split()[0], auth.split()[1]

def get_request(url, timeout=10):
    r = None
    i = 0
    while r == None and i < 3:
        try:
            r = requests.get(url, headers={"Accept":"application/vnd.github.mercy-preview+json"},
                             auth=HTTPBasicAuth(username, pw), timeout=timeout)
        except:
            print('tried request %d, no success'%i)
        i += 1
    return r

#def scrape_readme_single(dir, r):
#    git_path = r.split('repos/')[1]
#    url = 'https://raw.githubusercontent.com/%s/master/README'%git_path
#    try:
#        readme = requests.get('%s.md'%url,
#                              headers={"Accept":"application/vnd.github.mercy-preview+json"},
#                              auth=HTTPBasicAuth(username, pw))
#        if(readme.ok):
#            readme = readme.text
#        else:
#            readme = requests.get('%s.rst'%url,
#                                  headers={"Accept":"application/vnd.github.mercy-preview+json"},
#                                  auth=HTTPBasicAuth(username, pw)).text
#
#            # write to file
#            readme_corpus = open('%s/%s.txt'%(dir, git_path.split('/')[1]), 'w')
#            readme_corpus.write(str(readme.encode('utf8')))
#            readme_corpus.close()
#    except:
#        print('No README file for %s. Skipping'%r)

def get_readme_length(path, GProfile):
    git_path = path.split('repos/')[1]
    url = 'https://raw.githubusercontent.com/%s/master/README'%git_path
    try:
        readme = get_request('%s.md'%url)
        if(readme.ok):
            readme = readme.text
        else:
            readme = get_request('%s.md'%url).text
        
        GProfile.readme_lines = len(readme.split('\n'))
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
def get_comment_code_ratio(text, GProfile):
    for sym_s, sym_e in [('#','\n'),('"""', '"""')]:  #
        start = text.find(sym_s)
        end = text.find(sym_e, start + 1)
        while start != -1:
            GProfile.comment_lines += len(text[start: end].split('\n'))
            start = text.find(sym_s, end + 1)
            end = text.find(sym_e, start + 1)

# get summary stats of pep8 errors
def get_pep8_errs(text, GProfile, show_source = True):
    f = open('temp.py', 'w')
    f.write(text)
    f.close()

    call("pycodestyle --statistics -qq temp.py > temp.txt", shell=True)
    errs = open('temp.txt', 'r').read().splitlines()
    for err in errs:
        val, label = err.split()[0], err.split()[1]
        label = label[0:2] # remove extra details
        GProfile.pep8[label] += int(val)
    
    # cleanup
    #os.remove('temp.py')
    #os.remove('temp.txt')

# get distribution of commits over time
def get_repo_commit_history(commits_url, GProfile):
    try:
        r = get_request(commits_url.split('{/sha}')[0])
        commits = json.loads(r.text or r.content)
        GProfile.n_commits = len(commits)
        for commit in commits:
            date_string = commit['commit']['author']['date']
            date = datetime.strptime(date_string.split("T")[0],'%Y-%m-%d')
            GProfile.commit_history.append(date)
    except:
        print('couldnt get commit history for %s'%commits_url)

