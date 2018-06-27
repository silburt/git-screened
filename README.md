## ```git-screened``` - Automating Github Repository Assessment.
### By: Ari Silburt
```git-screened``` is a [web-app](http://git-screened.icu/) that automatically scrapes summary statistics for an input Github repository, and classifies it relative to the "industry standard". By industry standard, I mean the most popular Python repositories on Github by star count. This web-app is intended to assist those screening candidates (i.e. HR, hiring managers, etc.) by providing a quick assessment of production-level code quality. This project was completed over the Summer 2018 [Insight Data Science](https://www.insightdatascience.com/) program.

### Method
First, 5000 Python Github repositories were randomly scraped using the GIthub-API with 200 stars or more, and labelled as the industry standard for good code (i.e. the positive class). In addition, 5000 Python Github repositories were scraped under 1 star and 1 fork, labelled as the background class. It is assumed that the background class contains both industry standard *and* non-production level (i.e. the negative-class) code. In short, if a Github repository is highly starred it is likely that it is well-maintained and production grade, however an unstarred repository could still be of high quality and simply be unpopular. 

For each repository, the following features were generated:
- ```pep8``` errors.
- Length of REAME.md file.
- *more features*

A [One-Class SVM](http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html) was then trained on the scraped repositories/features, using the positive and background classes. The following metric, M, was maximized, used by [Baldeck & Asner, 2015](https://ieeexplore.ieee.org/document/6891145/) and [Lee & Liu, 2003](https://www.aaai.org/Papers/ICML/2003/ICML03-060.pdf) was used to train the model:

$$
M = r^2 / f_b
$$

Where:
-$r$ is the recall of the positive class.
-$f_b$ is the fraction of the background samples classified as the positive class.

In addition, a constraint of $r > 85\%$ is added, preventing solutions of low recall from being found, which would go against the intuition that most highly starred repositories are of production-grade quality. 

### Results


