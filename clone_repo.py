# Small function that clones a repository directly from Github.

from subprocess import call


repos = open('repo_data/bottom_stars_repos_Python.txt','r').read().splitlines()
for r in repos:
    name = r.split('repos/')[1]
    user = name.split('/')[0]
    package = name.split('/')[1]
    call("git clone https://www.github.com/%s.git bottom_repos_full/%s_%s"%(name, user, package), shell=True)
