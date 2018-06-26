import requests
from requests.auth import HTTPBasicAuth
import json
import local_gitfeatures as gf
import matplotlib.pyplot as plt
import signal
import glob

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
        
        self.packages = {}
        for p in pckgs:
            self.packages[p] = 0

def get_metrics_per_file(text, GProfile, test_file):
        # metrics
        #gf.get_package_freq(text, GProfile)
        gf.get_comment_code_ratio(text, GProfile)
        
        code_len = len(text.split('\n'))
        GProfile.code_lines += code_len
        if test_file:
            GProfile.test_lines += code_len

# not recursive
def digest_repo(file, GProfile):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    test_file = 0
    try:
        GProfile.n_pyfiles += 1
        
        text = open(file, 'r').read()
        if 'test' in file.lower().split('/')[-1]: # pytest
            test_file = 1
        get_metrics_per_file(text, GProfile, test_file)
        gf.get_pep8_errs(file, GProfile)

    except TimeoutException:
        print('%s timed out, skipping!'%file)

def get_features(repo_list_dir, output_dir):
    #repos = glob.glob('%s/*'%repo_list_dir)
    repos = ['misc_scripts/']#, 'misc_plots/']
    # Change the behavior of SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)
    for repo in repos:
        print(repo)
        GP = Github_Profile([])
        # scrape readme
        gf.get_readme_length(repo, GP)
        
        files = glob.glob('%s/**/*.py'%repo, recursive=True)
        for f in files:
            print(f)
            signal.alarm(60)
            try:
                # scrape file-by-file stats
                digest_repo(f, GP)
            
            except TimeoutException:
                print('%s timed out, skipping!'%repo)
                continue
            except:
                print('skipping repo %s'%repo)
                continue

        print(GP.code_lines, GP.readme_lines)
        # save
        string = '%s, %d, %d, %d, %d, %d, %d'
        data = open(output_dir, 'a')
        data.write(string%(repo, GP.n_pyfiles, GP.code_lines, GP.comment_lines,
                           GP.docstring_lines, GP.test_lines,
                           GP.readme_lines))

        for key in GP.pep8.keys():
            data.write(', %d'%GP.pep8[key])
        data.write('\n')
        data.close()

if __name__ == '__main__':
    repotype = 'bottom'
    repo_dir = '%s_repos/'%repotype
    #output_dir = "repo_data/%s_stars_stats_Python_local.txt"%repotype
    output_dir = 'misc_scripts_testttttt.txt'
    get_features(repo_dir, output_dir)

