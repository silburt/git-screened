# Small function that clones a repository directly from Github.
import os
from subprocess import call

stars = 'top'

repos = open('repo_data/%s_stars_repos_Python.txt'%stars,'r').read().splitlines()
for r in repos:
    name = r.split('repos/')[1]
    loc = "clone_repo_%s/%s"%(stars, name.replace('/', '_'))
    if not os.path.isdir(loc):
        string = "git clone https://www.github.com/%s.git %s"%(name, loc)
        call(string, shell=True)
    else:
        print("already download %s, skipping"%name)
