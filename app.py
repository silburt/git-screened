#!/usr/bin/env python
import dash
import flask
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import dash_table_experiments as dt
import gitscraper as gs
import gitfeatures as gf
import modeling as mod
import json
import numpy as np
import os.path
from scipy.stats import percentileofscore
from sklearn.externals import joblib

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)
server = app.server

#my_css_url = "https://unpkg.com/normalize.css@5.0.0"
my_css_url = "https://codepen.io/chriddyp/pen/bWLwgP.css"
app.css.append_css({"external_url": my_css_url})

app.layout = html.Div([
                       html.H1('git-screened', style={'font-style': 'Courier New',
                               'font-size': 25, 'text-align': 'center'}),
                       html.H2('Repository Name (Format: user/repo)'),
                       dcc.Input(value='', type='text', id='repo'),
                       html.Button('Search', id='button'),
                       dcc.Checklist(options=[{'label': 'Detailed Metrics', 'value': 'metrics'},
                                              {'label': 'Festive Meme', 'value': 'meme'}],
                                     values=[], id='checklist'),
                       html.Div(id='my-div', children='Enter a value and press Search'),
                       
#                       dcc.Upload(
#                                  id='upload-data',
#                                  children=html.Div([
#                                                     'Drag and Drop or ',
#                                                     html.A('Select Files')
#                                                     ]),
#                                  style={
#                                  'width': '100%',
#                                  'height': '60px',
#                                  'lineHeight': '60px',
#                                  'borderWidth': '1px',
#                                  'borderStyle': 'dashed',
#                                  'borderRadius': '5px',
#                                  'textAlign': 'center',
#                                  'margin': '10px'
#                                  },
#                                  # Allow multiple files to be uploaded
#                                  multiple=True
#                                  ),
#                       html.Div(id='output-data-upload'),
#                       html.Div(dt.DataTable(rows=[{}]), style={'display': 'none'})
                       
                       ])

#######Scraping/Processing Functions
# So, it looks like I can't call this function from gitscraper.py for some reason.
# Hypothesis: From the error: ValueError: signal only works in main thread, seems like
# you can't reference an outside function that then references a different outside func.
def digest_repo(repo_url, GProfile):
    r = gf.get_request('%s'%repo_url)
    if r.ok:
        repoItems = json.loads(r.text or r.content)

        for item in repoItems:
            try:
                test_file = 0
                if item['type'] == 'file' and item['name'][-3:] == '.py':
                    GProfile.n_pyfiles += 1
                    if 'test' in item['name'].lower(): # pytest
                        test_file = 1
                    print(item['download_url'])
                    gs.get_metrics_per_file(item, GProfile, test_file)
                elif item['type'] == 'dir':
                    digest_repo(item['url'], GProfile)
            except:
                print('%s timed out, skipping!'%item['download_url'])

def get_features(item):
    GP = gs.Github_Profile()
    contents_url = '%s/contents'%item['url']
    
    # scrape commit history
    gf.get_repo_commit_history(item, GP)
    
    # scrape readme
    gf.get_readme_length(contents_url, GP)
    
    # scrape file-by-file stats
    digest_repo(contents_url, GP)
    
    # scrape stargazers
    GP.n_stars = item['stargazers_count']
    
    # scrape forks
    GP.n_forks = item['forks_count']
    return GP

####### Output Functions #######
def get_quality(pcnt): # Need to fix - for pep8 errors, more is worse...
    if pcnt < 10:
        return 'POOR', 'red'
    elif pcnt > 10 and pcnt < 30:
        return 'FAIR', 'orange'
    elif pcnt > 30 and pcnt < 50:
        return 'GOOD', 'limegreen'
    elif pcnt > 50:
        return 'GREAT', 'green'

def output_feature(Xp, Xr, feat, repo_name, graph_flag=False, reverse_metric=False, nbins=30):
    features = ['code/files','comment/code','test/code','readme/code','docstring/code',
                'commits_per_time','E1/code','E2/code','E3/code','E4/code','E5/code',
                'E7/code','W1/code','W2/code','W3/code','W6/code','code_lines']
    HR_feature = ['Code Distribution', 'Commenting', 'Unit Test', 'Readme', 'Docstring', 'Commits', 'Style']
    pcnt = percentileofscore(Xp[:,feat], Xr[:,feat])
    if reverse_metric:  # e.g. for pep8 errors fewer is better
        pcnt = 100 - pcnt
    quality_label, color = get_quality(pcnt)
    if graph_flag:
        max_bin = np.max(np.histogram(Xp[:, feat], bins=nbins)[0])
        return html.Div([html.H3('{} quality is {}.'.format(HR_feature[feat], quality_label), style={'color':color}),
                         dcc.Graph(
                         id='basic-interactions{}'.format(feat),
                         figure={
                                'data': [
                                         {'x': Xp[:, feat], 'nbinsx':nbins ,'name': 'industry standard', 'type': 'histogram'},#, 'histnorm':'probability'},
                                 #{'x': Xb[:, feat], 'name': 'bad', 'type': 'histogram', 'opacity':0.7},  # 0 star/fork results
                                 {'x': Xr[:, feat][0]*np.ones(2), 'y':[0, max_bin],
                                 'name': repo_name, 'type': 'line', 'mode': 'lines', 'line': {'width': 5}}
                                 ],
                         'layout': {'title': '%.0fth percentile of industry standard repos'%pcnt,
                                  'xaxis':dict(title='log10({})'.format(features[feat])),
                                  'barmode':'overlay'}
                               })])
    else:
        return html.Div([html.H3('{} Quality is {}'.format(HR_feature[feat], quality_label), style={'color':color})])


def output(input_value, GP, Xr, score, checklist):
    # classification score
    meme = None
    if score == 1:
        outcome = 'PASS'
        color = 'green'
        if 'meme' in checklist:
            meme = ('https://raw.githubusercontent.com/silburt/'
                    'git-screened/master/app_images/happy_{}.jpg'.format(np.random.randint(1,5)))
    else:
        outcome = 'FAIL'
        color = 'red'
        if 'meme' in checklist:
            meme = ('https://raw.githubusercontent.com/silburt/'
                    'git-screened/master/app_images/sad_{}.jpg'.format(np.random.randint(1,5)))

    graph_flag = False
    if 'metrics' in checklist:
        graph_flag = True

    X_pos = np.load('models/X_pos.npy')
    #X_neg = np.load('models/X_neg.npy')
    return html.Div([html.H1('Results for Repository: "{}":'.format(input_value)),
                    html.Div([
                              html.H2('Status: {}'.format(outcome), style={'color':color}),
                              #html.H2('Checklist: {}'.format(checklist)),
                              html.Img(src=meme)
                              ]),
                     output_feature(X_pos, Xr, 0, input_value, graph_flag),
                     output_feature(X_pos, Xr, 1, input_value, graph_flag),
                     output_feature(X_pos, Xr, 2, input_value, graph_flag),
                     output_feature(X_pos, Xr, 3, input_value, graph_flag),
                     output_feature(X_pos, Xr, 4, input_value, graph_flag),
                     #output_feature(X_pos, Xr, 5, input_value, graph_flag),
                     output_feature(X_pos, Xr, 6, input_value, graph_flag, True),
                   ])

####### Main App Callback
@app.callback(
              Output(component_id='my-div', component_property='children'),
              [Input('button', 'n_clicks')],
              state=[State(component_id='repo', component_property='value'),
                     State(component_id='checklist', component_property='values')])
def update_output_div(n_clicks, input_value, checklist):
    repo_path = 'saved_repo_profiles/GP_%s.pkl'%(input_value.replace('/','_'))
    if os.path.isfile(repo_path):  # if profile already exists, don't re-scrape
        GP = joblib.load(repo_path)
    else:
        r = gf.get_request('https://api.github.com/repos/%s'%input_value)
        if r.ok:
            item = json.loads(r.text or r.content)
            GP = get_features(item)
            joblib.dump(GP, repo_path)
        else:
            return html.Div([html.H2('Couldnt find: "{}" on Github'.format(input_value))])

    score, Xr = mod.classify_repo(GP)
    return output(input_value, GP, Xr, score, checklist)

if __name__ == '__main__':
    app.server.run(port=8000, host='0.0.0.0')
    #app.run_server(host='0.0.0.0', debug=True, port=8000)
    #app.run(debug=True, use_reloader=False, port=5000, host='0.0.0.0')



#### Extra Code #####

#app = dash.Dash()
#app.title = "git-screened"
#server = app.server

#        GP = pickle.load(open(repo_path, 'rb'))
#            with open(repo_path, 'wb') as output_:
#                pickle.dump(GP, output_)

#                           dcc.Graph(
#                                     id='basic-interactions{}'.format(dim),
#                                     figure={
#                                     'data': [go.Box(
#                                                     x = X[:, dim],
#                                                     y = ["A", "A", "A", "A"],
#                                                     line = dict(color = 'gray'),
#                                                     name = "A",
#                                                     orientation = "h"
#                                                     ),
##                                              {'x': Xr[:, dim][0]*np.ones(2), 'y':[0, 0.5],
##                                              'name': input_value, 'type': 'line', 'mode': 'lines',
##                                              'line': {'width': 5}}
#                                              ],
#                                     'layout': {'title':features[dim], 'xaxis':dict(title='Value'), 'barmode':'overlay'}
#                                     }
#                                     ),

#
#'shapes':[{
#          'type': 'line',
#          'x0': 1,
#          'y0': 0,
#          'x1': 1,
#          'y1': 1500,
#          'name': input_value,
#          'line': {
#          'color': 'green',
#          'width': 3,
#          },
#          }]
