import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import json
from subprocess import call
import os


def get_request(url, timeout=10):
    """
    Process Github API request.
    """
    r = None
    i = 0
    while r is None and i < 3:
        try:
            try:
                auth = open('../keys_passwords/auth.txt').read()
                username, pw = auth.split()[0], auth.split()[1]
                authtoken = open('../keys_passwords/token.txt').read()
                r = requests.get(url, headers={"Accept":
                    "application/vnd.github.mercy-preview+json",
                    "Authorization": "token %s"%authtoken},
                    auth=HTTPBasicAuth(username, pw), timeout=timeout)
            except:
                r = requests.get(url, headers={"Accept":
                    "application/vnd.github.mercy-preview+json"}, timeout=timeout)
        except:
            print('tried request %d, no success' % i)
        i += 1
    return r


def get_readme_length(contents_url, GProfile):
    """
    Get number of lines in readme if exists. 
    """
    r = get_request(contents_url)
    if r.ok:
        contents = json.loads(r.text or r.content)
        readme_url = None
        for c in contents:
            if 'README' in c['name']:
                readme_url = c['download_url']
                break
        if readme_url:
            readme = get_request(readme_url)
            if(readme.ok):
                text = readme.text
                while "\n\n" in text:
                    text = text.replace("\n\n", "\n")
                GProfile.readme_lines = len(text.splitlines())


def get_comment_code_ratio(text, GProfile):
    """
    Get frequency of comments vs. code. ***** CAN SHORTEN THIS 
    """
    for sym_s, sym_e in [('#','\n')]:
        start = text.find(sym_s)
        end = text.find(sym_e, start + 1)
        comm_len = 0
        while start != -1:
            comm_len += len(text[start: end].split('\n'))
            start = text.find(sym_s, end + 1)
            end = text.find(sym_e, start + 1)

    for sym_s, sym_e in [('"""', '"""')]:
        start = text.find(sym_s)
        end = text.find(sym_e, start + 1)
        doc_len = 0
        while start != -1:
            doc_len += len(text[start: end].split('\n'))
            start = text.find(sym_s, end + 1)
            end = text.find(sym_e, start + 1)

    GProfile.comment_lines += comm_len
    GProfile.docstring_lines += doc_len


def get_pep8_errs(text, GProfile, show_source=True):
    """
    Get summary stats of pep8 errors
    """

    ext = 'temp3'  # write pep8 to file.
    f = open('%s.py' % ext, 'w')
    f.write(text)
    f.close()

    call("pycodestyle --statistics -qq %s.py > %s.txt" % (ext, ext), shell=True)
    errs = open('%s.txt' % ext, 'r').read().splitlines()
    for err in errs:
        val, label = err.split()[0], err.split()[1]
        label = label[0: 2]  # remove extra details
        GProfile.pep8[label] += int(val)

    # cleanup
    os.remove('%s.py' % ext)
    os.remove('%s.txt' % ext)

def get_repo_commit_history(item, GProfile):
    """
    Get number and distribution of commits over time.
    """
    try:
        commits_url = item['commits_url'].split('{/sha}')[0]
        r = get_request(commits_url)
        commits = json.loads(r.text or r.content)
        GProfile.n_commits = len(commits)
        for commit in commits:
            date_string = commit['commit']['author']['date']
            date = datetime.strptime(date_string.split("T")[0],'%Y-%m-%d')
            GProfile.commit_history.append(date)

        # get commits/time
        created = datetime.strptime(item['created_at'].split("T")[0],'%Y-%m-%d')
        updatelast = datetime.strptime(item['updated_at'].split("T")[0],'%Y-%m-%d')
        GProfile.commits_per_time = (updatelast - created).days / float(GProfile.n_commits)
    except:
        print('couldnt get commit history for %s' % commits_url)
