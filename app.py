# https://stackoverflow.com/questions/45736656/how-to-use-a-button-to-trigger-callback-updates

import requests
from requests.auth import HTTPBasicAuth
import json
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import signal

##################
# base directory is from ./run.py
auth = open('utils/auth.txt').read()
username, pw = auth.split()[0], auth.split()[1]
authtoken = open('utils/token.txt').read()

##################
class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

##################
def get_request(url, timeout=10):
    r = None
    i = 0
    while r == None and i < 3:
        try:
            r = requests.get(url, headers={"Accept":"application/vnd.github.mercy-preview+json",
                             "Authorization": "token %s"%authtoken},
                             auth=HTTPBasicAuth(username, pw), timeout=timeout)
        except:
            print('tried request %d, no success'%i)
        i += 1
    return r

def get_readme_length(contents_url, GProfile):
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
                    text = text.replace("\n\n","\n")
                GProfile.readme_lines = len(text.splitlines())

# get frequency of comments vs. code
def get_comment_code_ratio(text, GProfile):
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

# get summary stats of pep8 errors
def get_pep8_errs(text, GProfile, show_source = True):
    ext = '10'
    f = open('temp%s.py'%ext, 'w')
    f.write(text)
    f.close()
    
    call("pycodestyle --statistics -qq temp%s.py > temp%s.txt"%(ext, ext), shell=True)
    errs = open('temp%s.txt'%ext, 'r').read().splitlines()
    for err in errs:
        val, label = err.split()[0], err.split()[1]
        label = label[0:2] # remove extra details
        GProfile.pep8[label] += int(val)
    
    # cleanup
    os.remove('temp%s.py'%ext)
    os.remove('temp%s.txt'%ext)

# get distribution of commits over time
def get_repo_commit_history(item, GProfile):
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
        GProfile.commits_per_time = (updatelast - created).days/float(GProfile.n_commits)
    except:
        print('couldnt get commit history for %s'%commits_url)

##################
class Github_Profile:
    def __init__(self):
        self.user = ''
        self.url = ''
        
        # metrics
        self.commit_history = []
        self.commits_per_time = 0
        self.n_commits = 0
        self.stargazers = 0
        self.forks = 0
        self.test_lines = 0
        self.docstring_lines = 0
        self.comment_lines = 0
        self.readme_lines = 0
        self.code_lines = 0
        self.n_pyfiles = 0
        self.pep8 = {}
        a = ['E1','E2','E3','E4','E5','E7','E9','W1','W2','W3','W5','W6']
        for p in a:
            self.pep8[p] = 0

##################
def get_metrics_per_file(item, GProfile, test_file):
    r = get_request(item['download_url'])
    if r.ok:
        text = r.text
        
        # metrics
        get_comment_code_ratio(text, GProfile)
        get_pep8_errs(text, GProfile)
        
        code_len = len(text.split('\n'))
        GProfile.code_lines += code_len
        if test_file:
            GProfile.test_lines += code_len

##################
# recursive
def digest_repo(repo_url, GProfile):
    r = get_request('%s'%repo_url)
    if r.ok:
        repoItems = json.loads(r.text or r.content)
        
        signal.signal(signal.SIGALRM, timeout_handler)
        for item in repoItems:
            signal.alarm(10)
            try:
                test_file = 0
                if item['type'] == 'file' and item['name'][-3:] == '.py':
                    GProfile.n_pyfiles += 1
                    if 'test' in item['name'].lower(): # pytest
                        test_file = 1
                    print(item['download_url'])
                    get_metrics_per_file(item, GProfile, test_file)
                elif item['type'] == 'dir':
                    digest_repo(item['url'], GProfile)
            except TimeoutException:
                print('%s timed out, skipping!'%item['download_url'])

##################
def get_features(item, GP):
    contents_url = '%s/contents'%item['url']
    
    # scrape commit history
    #get_repo_commit_history(item, GP)
    
    # scrape readme
    get_readme_length(contents_url, GP)
    
    # scrape file-by-file stats
    digest_repo(contents_url, GP)
    
    # scrape stargazers
    GP.stargazers = item['stargazers_count']
    
    # scrape forks
    GP.forks = item['forks_count']
    
    return GP


##################
##################
##################
app = dash.Dash()

app.layout = html.Div([
                       html.Label('Repository Name (format: user/repo)'),
                       dcc.Input(id='repo', value='', type="text"),
                       html.Button('Search', id='button'),
                       html.Div(id='my-div')
                       ])

@app.callback(
              Output(component_id='my-div', component_property='children'),
              [Input('button', 'n_clicks')],
              state=[State(component_id='repo', component_property='value')]
)
def update_output_div(n_clicks, input_value):
    GP = Github_Profile()
    r = get_request('https://api.github.com/repos/%s'%input_value)
    if r.ok:
        item = json.loads(r.text or r.content)
        GP = get_features(item, GP)
        
        return 'Stargazers for "{}" is {}'.format(input_value, GP.stargazers)
    else:
        return 'Couldnt get stats for "{}"'.format(input_value)

if __name__ == '__main__':
    app.run_server()
