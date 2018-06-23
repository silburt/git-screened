import requests
from requests.auth import HTTPBasicAuth
import json
import gitfeatures as gf
import signal
import numpy as np

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Class for each Github Profile
class Github_Profile:
    def __init__(self, pckgs):
        self.user = ''
        self.url = ''
        # metrics
        self.commit_history = []
        self.commits_per_time = 0
        self.n_commits = 0
        self.n_stars = 0
        self.n_forks = 0
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
        
        self.packages = {}
        for p in pckgs:
            self.packages[p] = 0

def get_metrics_per_file(item, GProfile, test_file):
    r = gf.get_request(item['download_url'])
    if r.ok:
        text = r.text
        
        # metrics
        #gf.get_package_freq(text, GProfile)
        gf.get_comment_code_ratio(text, GProfile)
        gf.get_pep8_errs(text, GProfile)
        
        code_len = len(text.split('\n'))
        GProfile.code_lines += code_len
        if test_file:
            GProfile.test_lines += code_len

# recursive
def digest_repo(repo_url, GProfile):
    r = gf.get_request('%s'%repo_url)
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

def get_features(item, GP):
    contents_url = '%s/contents'%item['url']
    
    # scrape commit history
    gf.get_repo_commit_history(item, GP)
        
    # scrape readme
    gf.get_readme_length(contents_url, GP)
        
    # scrape file-by-file stats
    digest_repo(contents_url, GP)
            
    # scrape stargazers
    GP.n_stars = item['stargazers_count']
            
    # scrape forks
    GP.n_forks = item['forks_count']

    return GP

def get_training_repos(repo_list_dir, output_dir):
    proc_repos = np.loadtxt(output_dir, delimiter=',', usecols=[0], dtype='str')
    repos = open(repo_list_dir, 'r').read().splitlines()
    # Change the behavior of SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)
    for repo in repos:
        if repo in proc_repos:
            print('already scanned %s'%repo)
            continue
        GP = Github_Profile([])
        GP.user = repo.split('repos/')[1].split('/')[0]
        r = gf.get_request(repo)
        if r.ok:
            item = json.loads(r.text or r.content)
            signal.alarm(60)
            try:
                if item['fork'] == False:  # for now ignore forks
                    GP = get_features(item, GP)

                    # save
                    string = '%s, %d, %d, %d, %d, %d, %d, %d, %f, %d, %d'
                    data = open(output_dir, 'a')
                    data.write(string%(repo, GP.n_pyfiles, GP.code_lines, GP.comment_lines,
                                       GP.docstring_lines, GP.test_lines,
                                       GP.readme_lines, GP.n_commits, GP.commits_per_time,
                                       GP.n_stars, GP.n_forks))
                                       
                    for key in GP.pep8.keys():
                        data.write(', %d'%GP.pep8[key])
                    data.write('\n')
                    data.close()
            except TimeoutException:
                print('%s timed out, skipping!'%repo)
            except:
                print('skipping repo %s'%repo)

if __name__ == '__main__':
    repo_dir = 'repo_data/top_stars_repos_Python.txt'
    output_dir = "repo_data/top_stars_stats_Python.txt"

    get_training_repos(repo_dir, output_dir)

# scrape global stats about user
#def scrape_by_user(git_user, packages, readme_dir='candidate'):
#    GProfile = Github_Profile(packages)
#    GProfile.url = 'https://api.github.com/users/%s/repos'%git_user
#    r = requests.get(GProfile.url, headers={"Accept":"application/vnd.github.mercy-preview+json"},
#                     auth=HTTPBasicAuth(username, pw))
#    userItems = json.loads(r.text or r.content)
#
#    for item in userItems:
#        if item['fork'] == False:  # for now ignore forks
#            if 'DeepMoon' in item['url']:   #### TEMP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                # scrape readme
#                gf.get_readme_length(item['url'], GProfile)
#
#                # scrape file-by-file stats
#                repo_url = '%s/contents'%item['url']
#                digest_repo(repo_url, GProfile)
#
#                # scrape commit history
#                gf.get_repo_commit_history(item['commits_url'], GProfile)
#
#                # scrape stargazers
#                GProfile.stargazers = item['stargazers_count']
#
#                # scrape forks
#                GProfile.forks = item['forks_count']
#    return GProfile

#    get_forks(repo_dir, output_dir)

#    User = scrape_by_user('silburt', ['numpy'], 'candidate')
#    print(User.n_pytests, User.code_lines, User.n_pyfiles, float(User.comment_lines)/User.code_lines,
#          float(User.readme_lines)/User.code_lines, User.code_lines/User.n_pyfiles)
#    print(User.pep8)

#def get_forks(repo_list_dir, output_dir):
#    repos = open(repo_list_dir, 'r').read().splitlines()
#    data = open(output_dir, 'a')
#    for repo in reversed(repos):
#        r = gf.get_request(repo)
#        if r is None:
#            continue
#        item = json.loads(r.text or r.content)
#        stargazer = item['forks_count']
#        data.write('%s, %d\n'%(repo, stargazer))
#        print(repo)
#    data.close()
