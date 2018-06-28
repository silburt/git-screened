## ```git-screened``` - Automating Github Repository Assessment.
### By: Ari Silburt
```git-screened``` is a [web-app](http://git-screened.icu/) that automatically scrapes summary statistics for an input Github repository, and classifies it relative to the "industry standard". By industry standard, I mean the most popular Python repositories on Github by star count. This web-app is intended to assist those screening candidates (i.e. HR, hiring managers, etc.) by providing a quick assessment of production-level code quality. This project was completed over the Summer 2018 [Insight Data Science](https://www.insightdatascience.com/) program.

### Method
First, 5000 Python Github repositories were randomly scraped using the GIthub-API with 200 stars or more, and labelled as the "industry standard" for production-level code (i.e. the positive class). In addition, 5000 Python Github repositories were scraped with 2 stars or less, labelled as the background class. It is assumed that the background class contains a mix of both production (positive class) *and* non-production level (negative-class) code. The basic rationale behind this is: if a Github repository is highly starred it is likely that it is production grade, however a low starred repository could still be of high quality and simply be unpopular. 

For each repository, the following features were generated (all except the last feature are normalized by lines of code):
- The various ```pep8``` [errors](https://pycodestyle.readthedocs.io/en/latest/intro.html#error-codes).
- Length of REAME file.
- Number of unit tests.
- Number of comments.
- Number of docstrings. 
- Lines of code / number of python files. 

A [One-Class SVM](http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html) was then trained on the positive and background class, using ```r*r/b``` as the metric to be maximized, where ```r``` is the recall of the positive class and ```b``` is the fraction of background samples classified as the positive class. This metric is a popular choice for the One-Class SVM, being used by, e.g. [Baldeck & Asner, 2015](https://ieeexplore.ieee.org/document/6891145/) and [Lee & Liu, 2003](https://www.aaai.org/Papers/ICML/2003/ICML03-060.pdf).

In addition, a constraint of $r > 85\%$ is added to prevent solutions of low positive-class recall, consistent with the intuition that highly starred repositories are of production-grade quality. 

### Usage
Below is a basic explanation of each python file:
- ```app.py``` - Contains the frontend website, hosted on AWS at [git-screened.icu](http://git-screened.icu/).
-```gitfeatures.py``` - Holds most of the backend functions to scrape repositories using the Github API and generate features for each repository.
-```gitscraper.py``` - Contains the top level functions that scrape repositories using the Github API and generate the features, storing the stats in csv files in repo_data/. 
-```modeling.py``` - Contains the code for pre-processing the scraped data and training the One-Class SVM model. 
-```scrape_repo_apicalls.py``` - Initial function that scrapes Github repository API calls based off desired criteria (e.g. number of stars, coding language, etc.).


