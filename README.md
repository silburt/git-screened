# ```git-screened``` - Automating Github Repository Assessment.
```git-screened``` is a [web-app](http://git-screened.icu:5000/) that automatically scrapes summary statistics for an input Github repository and classifies it relative to the industry standard. By industry standard, I mean the most popular 5000 Python repositories on Github by star count. This web-app is intended to assist those screening candidates (HR, hiring managers, etc.), and give quick metrics to aid the process. This project was completed over the Summer 2018 Insight Data Science program by Ari Silburt.

## Method
5000 Python Github repositories were scraped using the GIthub-API with 200 stars or more, labelled as the industry standard for good code (i.e. the positive class). In addition, 5000 Python Github repositories were scraped with 0 stars and 0 forks, labelled as the background class, containing both industry standard non-production level (i.e. the negative-class) repositories. The following features were generated for each repository:
- ```pep8``` errors.



