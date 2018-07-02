#!/usr/bin/env python
import dash
import flask
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import gitscraper as gs
import gitfeatures as gf
import modeling as mod
import json
import numpy as np
import os.path
from scipy.stats import percentileofscore
from sklearn.externals import joblib
from textwrap import dedent

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)
server = app.server

my_css_url = "https://codepen.io/chriddyp/pen/bWLwgP.css"
my_css_button = "https://codepen.io/chriddyp/pen/brPBPO.css"
app.css.append_css({"external_url": my_css_url})
app.css.append_css({"external_url": my_css_button})

app.layout = html.Div([
    html.Div([
        html.H3('git-screened', style={'font-family': 'Courier New',
            'background-color': '#CCE5FF', 'text-align': 'center', 'font-size': 45}),
        html.H3('Automating Github Repository Assessment', style={'font-size': 25, 
            'font-style': 'italic', 'text-align': 'center'}),
        html.H3('By: Ari Silburt', style={'font-size': 20,
        'font-style': 'normal', 'text-align': 'center'}),
        dcc.Markdown(dedent('''
         ```git-screened``` is a tool for recruiters and hiring managers to
         quickly determine if a candidate writes clear, beautiful, and
         productionized Python code. It can also be used by developers to
         polish their portfolios to match the "industry standard" on GitHub.
         Here, "industry standard" refers to the 6000 most starred Python
         repositories on Github. This project was completed over the Summer
         2018 [Insight Data Science](https://www.insightdatascience.com/)
         program.The code, along with additional details, are available on
         [Github](https://github.com/silburt/git-screened).

         **Instructions**: Enter a Github repository in the search box below
         and click "search". Summary statistics for the repository will be
         scraped (in real time from Github) and output on the screen, along
         with a pass/fail classification for the overall production quality
         of the repository (generated from a pre-trained One-Class SVM model).
         Ticking 'Detailed Metrics' will output histograms that
         visualize/contextualize the searched repository to the industry
         standard (note: histograms are dislpayed in log scale). Lastly, a
         festive meme can also be output ;).

         **Final Notes**: This app works best for repositories where the
         majority of code is written in Python. In addition, since the
         statistics are scraped in real time, large repositories will take a
         moment.

        ------
        ''')), ],
        style={'background-color': 'WhiteSmoke'}),

    html.H2('Repository Name (Format: user/repository)',
            style={'font-size': 20, 'font-style': 'normal'}),
    dcc.Input(value='', type='text', id='repo'),
    html.Button('Search', id='button'),
    dcc.Checklist(options=[{'label': 'Detailed Metrics', 'value': 'metrics'},
                           {'label': 'Festive Meme', 'value': 'meme'}],
                  values=[], id='checklist'),
    html.Div(id='my-div', children='Enter a value and press Search')])


# SCRAPING/PROCESSING FUNCTIONS
def digest_repo(repo_url, GProfile):
    """
    Look through each file and directory, extract metrics from
    each python file. Recursive function.
    """
    r = gf.get_request('%s' % repo_url)
    if r.ok:
        repoItems = json.loads(r.text or r.content)

        for item in repoItems:
            try:
                if item['type'] == 'file' and item['name'][-3:] == '.py':
                    GProfile.n_pyfiles += 1
                    print(item['download_url'])
                    gs.get_metrics_per_file(item, GProfile)
                elif item['type'] == 'dir':
                    digest_repo(item['url'], GProfile)
            except:
                print('%s timed out, skipping!' % item['download_url'])


def get_features(item):
    """
    Top-level function that scrapes features for each python file
    and stores it in a Github_Profile class.
    """
    GP = gs.Github_Profile()
    contents_url = '%s/contents' % item['url']

    # scrape readme
    gf.get_readme_length(contents_url, GP)

    # scrape file-by-file stats
    digest_repo(contents_url, GP)

    # scrape commit history
    gf.get_repo_commit_history(item, GP)

    # scrape stargazers
    GP.n_stars = item['stargazers_count']

    # scrape forks
    GP.n_forks = item['forks_count']
    
    return GP


# OUTPUT FUNCTIONS
def get_quality(pcnt):
    """
    Maps percentileofscore value to quality descriptor.
    """
    if pcnt < 10:
        return 'POOR', 'red'
    elif pcnt > 10 and pcnt < 30:
        return 'FAIR', 'orange'
    elif pcnt > 30 and pcnt < 50:
        return 'GOOD', 'limegreen'
    elif pcnt > 50:
        return 'GREAT', 'green'


def output_feature(Xp, Xr, feat, repo_name, graph_flag=False,
                   pep8=False, nbins=30):
    """
    Outputs histogram and percentile score for given feature for the
    pre-scraped "Industry Standard" repos, as well as for the repo in question.
    """
    features = ['code/files', 'comment/code lines', 'test/code lines',
                'readme/code lines', 'docstring/code lines',
                'pep8 errors/code lines']
    HR_feature = ['Code Distribution', 'Commenting', 'Unit Test', 'Readme',
                  'Docstring', 'pep8 Error (more=worse)']

    if pep8:  # for pep8 errors fewer is better
        Xr_ = 0
        Xp_ = 0
        for i in range(feat, feat + 10):
            Xr_ += 10**Xr[:, i]
            Xp_ += 10**Xp[:, i]
        pcnt = percentileofscore(np.log10(Xp_), np.log10(Xr_))
        pl_P = Xp_[Xp_ <= 1.2]
        pl_R = Xr_
        quality_label, color = get_quality(100 - pcnt)
    else:
        pcnt = percentileofscore(Xp[:, feat], Xr[:, feat])
        pl_P = Xp[:, feat]
        pl_R = Xr[:, feat]
        quality_label, color = get_quality(pcnt)

    if graph_flag:
        max_bin = np.max(np.histogram(Xp[:, feat], bins=nbins)[0])
        return html.Div([html.H3('{} quality is {}.'.format(HR_feature[feat], quality_label),
                                 style={'color': color}),
                         dcc.Graph(id='basic-interactions{}'.format(feat),
                                   figure={'data': [{'x': pl_P, 'nbinsx': nbins,
                                                     'name': 'industry standard',
                                                     'type': 'histogram'},  # 'histnorm':'probability'},
                                                    {'x': pl_R[0] * np.ones(2), 'y':[0, max_bin],
                                                     'name': repo_name,
                                                     'type': 'line', 'mode': 'lines',
                                                     'line': {'width': 5}}],
                                           'layout': {'title': '%.0fth percentile of industry standard repos' % pcnt,
                                           'xaxis': dict(title='log10({})'.format(features[feat])), 'barmode':'overlay'}}
                                   )])
    else:
        return html.Div([html.H3('{} Quality is {}'.format(HR_feature[feat], quality_label), style={'color':color})])


def output(input_value, GP, Xr, score, checklist, modeltype='OC-SVM'):
    """
    Top-level function that outputs the pass/fail classification score,
    quality scores for each metric, meme, and histograms.
    """
    # classification score
    meme = None
    rand = np.random.randint(1, 5)
    if score == 1:
        outcome = 'YES'
        color = 'green'
        if 'meme' in checklist:
            meme = ('https://raw.githubusercontent.com/silburt/'
                    'git-screened/master/app_images/happy_{}.jpg'.format(rand))
    else:
        outcome = 'NO'
        color = 'red'
        if 'meme' in checklist:
            meme = ('https://raw.githubusercontent.com/silburt/'
                    'git-screened/master/app_images/sad_{}.jpg'.format(rand))

    graph_flag = False
    if 'metrics' in checklist:
        graph_flag = True

    X_pos = np.load('models/X_pos_unscaled_%s.npy' % modeltype)
    return html.Div([html.H1('Results for Repository: "{}":'.format(input_value)),
                     html.Div([
                              html.H2('Production-Level Code: {}'.format(outcome),
                                      style={'color': color}),
                              html.Img(src=meme)
                              ]),
                     output_feature(X_pos, Xr, 0, input_value, graph_flag),
                     output_feature(X_pos, Xr, 1, input_value, graph_flag),
                     output_feature(X_pos, Xr, 2, input_value, graph_flag),
                     output_feature(X_pos, Xr, 3, input_value, graph_flag),
                     output_feature(X_pos, Xr, 4, input_value, graph_flag),
                     output_feature(X_pos, Xr, 5, input_value, graph_flag, True)
                     ])


@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input('button', 'n_clicks')],
    state=[State(component_id='repo', component_property='value'),
           State(component_id='checklist', component_property='values')])
def update_output_div(n_clicks, input_value, checklist):
    """
    Main App Callback. Takes a user/repository as input, scrapes the stats
    using the Github API, classifies the repository using the pre-trained
    One-Class SVM model, and sends all the information to output() to be
    output on the screen. Saves a querried Github_Profile for faster loading
    subsequent times.
    """
    repo_path = 'saved_repo_profiles/GP_%s.pkl' % (input_value.replace('/', '_'))
    if os.path.isfile(repo_path):  # if profile already exists, don't re-scrape
        GP = joblib.load(repo_path)
    else:
        r = gf.get_request('https://api.github.com/repos/%s' % input_value)
        if r.ok:
            item = json.loads(r.text or r.content)
            GP = get_features(item)
            joblib.dump(GP, repo_path)
        else:
            return html.Div([html.H2('Couldnt find: "{}" on Github'.format(input_value),
                                     style={'font-style': 'normal', 'font-size': 15})])

    score, Xr = mod.classify_repo(GP)  # r for repo
    return output(input_value, GP, Xr, score, checklist)


if __name__ == '__main__':
    app.server.run(port=8000, host='0.0.0.0')
