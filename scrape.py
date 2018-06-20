# Various important scraping functions here
import os
import numpy as np
from requests.auth import HTTPBasicAuth
import requests
import json

auth = open("utils/auth.txt").read()
username, pw = auth.split()[0], auth.split()[1]

########################################
def scrape_repos_by_specs(name, raw_links, n_pages=10, start_page=0):
    n_scraped = 0
    f = open('repo_data/%s_repos_Python.txt'%name, 'w')
    for raw_link in raw_links:
        for i in range(start_page, start_page + n_pages):
            link = '%s&per_page=100&page=%d'%(raw_link, i)
            r = requests.get(link,
                             headers={"Accept":"application/vnd.github.mercy-preview+json"},
                             auth=HTTPBasicAuth(username, pw))
            if(r.ok):
                repos = json.loads(r.text or r.content)
                for item in repos['items']:
                    f.write('%s\n'%item['url'])
                    n_scraped += 1 # number of repo urls scraped
            print(link)

    f.close()
    print("Scraped %d repos'"%n_scraped)  # why is it only scraping 4x fewer than I want?

########################################
def scrape_full_repo(dir = 'repos', max_size = 100000):
    repos = open('repos/machine-learning_repos.txt', 'r').read().splitlines()
    for r in repos:
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

########################################
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
        print('No README file for %s. Skipping'%repo)

def scrape_repo_readmes(repo_file, output_dir='readmes'):
    repos = open(repo_file, 'r').read().splitlines()
    for r in repos:
        scrape_readme_single(output_dir, r)

if __name__ == '__main__':
    #link = 'https://api.github.com/search/repositories?q=topic:%s+language:Python+sort:stars'%(t, i)
    #link = 'https://api.github.com/search/repositories?q=language:Python+stars:0+forks:0'%i
    link = 'https://api.github.com/search/repositories?q=language:Python+sort:stars'
    
    good_links = ['https://api.github.com/search/repositories?q=language:Python+stars:300..600',
                  'https://api.github.com/search/repositories?q=language:Python+stars:600..1000',
                  'https://api.github.com/search/repositories?q=language:Python+stars:1000..5000']
    good_name = 'top_stars'
    
    bad_links = ['https://api.github.com/search/repositories?q=language:Python+stars:0+forks:0',
                 'https://api.github.com/search/repositories?q=language:Python+stars:1+forks:0',
                 'https://api.github.com/search/repositories?q=language:Python+stars:0+forks:1']
    bad_name = 'bottom_stars'
    
    scrape_repos_by_specs(good_name, good_links)
    #scrape_repos_by_specs(bad_name, bad_links)

    #topics = ['visualization', 'statistics', 'programming', 'machine-learning']
#    for t in topics:
#        print('Getting READMEs for %s'%t)
#        repo_file = 'repo_data/%s_repos.txt'%t
#        output_dir = 'readme_out/%s'%t
#        if not os.path.exists(output_dir):
#            os.makedirs(output_dir)
#        scrape_repo_readmes(repo_file, output_dir)

