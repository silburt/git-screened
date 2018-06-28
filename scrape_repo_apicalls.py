# Script that scrapes a collection of repositories fitting specified criteria
# (e.g. number of stars, coding language, etc.) and saves their Github API
# calls into a text file.
import os
from requests.auth import HTTPBasicAuth
import requests
import json
import gitfeatures as gf

def scrape_repos_by_specs(name, raw_links, n_pages=10, start_page=0):
    n_scraped = 0
    f = open('repo_data/%s_repos_Python.txt' % name, 'w')
    for raw_link in raw_links:
        for i in range(start_page, start_page + n_pages):
            link = '%s&per_page=100&page=%d' % (raw_link, i)
            r = gf.get_request(link)
            if(r.ok):
                repos = json.loads(r.text or r.content)
                for item in repos['items']:
                    f.write('%s\n' % item['url'])
                    n_scraped += 1  # number of repo urls scraped
            print(link)

    f.close()
    print("Scraped %d repos'" % n_scraped)


if __name__ == '__main__':
    #link = 'https://api.github.com/search/repositories?q=topic:%s+language:Python+sort:stars'%(t, i)
    #link = 'https://api.github.com/search/repositories?q=language:Python+stars:0+forks:0'%i
    
    #link = 'https://api.github.com/search/repositories?q=language:Python+sort:stars'
#    good_links = ['https://api.github.com/search/repositories?q=language:Python+stars:300..600',
#                  'https://api.github.com/search/repositories?q=language:Python+stars:600..1000',
#                  'https://api.github.com/search/repositories?q=language:Python+stars:1000..5000']
#    good_name = 'top_stars'

#    good_links = ['https://api.github.com/search/repositories?q=language:Python+stars:240..300',
#                  'https://api.github.com/search/repositories?q=language:Python+stars:200..240',
#                  'https://api.github.com/search/repositories?q=language:Python+stars:170..200']
#    good_name = 'top_stars2'

#    link = 'https://api.github.com/search/repositories?q=language:Python+sort:forks'
#    good_links = ['https://api.github.com/search/repositories?q=language:Python+stars:130..190',
#                  'https://api.github.com/search/repositories?q=language:Python+stars:190..350',
#                  'https://api.github.com/search/repositories?q=language:Python+stars:350..5000']
#    good_name = 'top_forks'

#    bad_links = ['https://api.github.com/search/repositories?q=language:Python+stars:0+forks:0',
#                 'https://api.github.com/search/repositories?q=language:Python+stars:1+forks:0',
#                 'https://api.github.com/search/repositories?q=language:Python+stars:0+forks:1']
#    bad_name = 'bottom_stars'

    bad_links = ['https://api.github.com/search/repositories?q=language:Python+stars:1+forks:0+created:<2018-01-01+created:>2017-01-01',
                 'https://api.github.com/search/repositories?q=language:Python+stars:1+forks:0+created:<2017-01-01+created:>2016-01-01',
                 'https://api.github.com/search/repositories?q=language:Python+stars:1+forks:0+created:<2016-01-01+created:>2015-01-01']
    bad_name = 'bottom_stars_created2'

    #scrape_repos_by_specs(good_name, good_links)
    scrape_repos_by_specs(bad_name, bad_links)
