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
    hi_star_links = ['https://api.github.com/search/repositories?q=language:Python+stars:130..190',
                     'https://api.github.com/search/repositories?q=language:Python+stars:190..350',
                     'https://api.github.com/search/repositories?q=language:Python+stars:350..5000']
    hi_star_name = 'top_stars'

    low_star_links = ['https://api.github.com/search/repositories?q=language:Python+stars:0+forks:0',
                      'https://api.github.com/search/repositories?q=language:Python+stars:1+forks:0',
                      'https://api.github.com/search/repositories?q=language:Python+stars:0+forks:1']
    low_star_name = 'bottom_stars'

    scrape_repos_by_specs(hi_star_name, hi_star_links)
    scrape_repos_by_specs(low_star_name, low_star_links)
