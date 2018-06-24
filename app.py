#!/usr/bin/env python
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import gitscraper as gs
import gitfeatures as gf
import modeling as mod
import json
import pickle
import numpy as np

app = dash.Dash()
#app = dash.Dash(__name__)
server = app.server
#my_css_url = "https://unpkg.com/normalize.css@5.0.0"
#app.css.append_css({"external_url": my_css_url})

app.layout = html.Div([
                       html.H1('git-screened', style={'font-style': 'Courier New',
                               'font-size': 25, 'text-align': 'center'}),
                       html.Label('Repository Name (Format: user/repo)', style={'text-align': 'center'}),
                       dcc.Input(value='', type='text', id='repo'),
                       html.Button('Search', id='button'),
                       dcc.Checklist(options=[{'label': 'Detailed Metrics', 'value': 'on'}], values=[], id='metrics'),
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

####### Output Function
def output(input_value, GP, X, Xb, Xr, score, metrics):
    features = ['code/files','comment/code','test/code','readme/code','docstring/code',
                'commits_per_time','E1/code','E2/code','E3/code','E4/code','E5/code',
                'E7/code','W1/code','W2/code','W3/code','W6/code','code_lines']

    # classification score
    if score == 1:
        outcome = 'PASS'
        color = 'green'
        link = ('https://raw.githubusercontent.com/silburt/'
                'git-screened/master/app_images/baby.jpg')
    else:
        outcome = 'FAIL'
        color = 'red'
        link = ('https://raw.githubusercontent.com/silburt/'
                'git-screened/master/app_images/stormtrooper.jpg')
    
    dim = 2
    div_output = html.Div([
                       html.H1('Results for Repository: "{}"'.format(input_value)),
                       dcc.Graph(
                                 id='basic-interactions{}'.format(dim),
                                 figure={
                                 'data': [
                                          {'x': X[:, dim], 'name': 'good', 'type': 'histogram'},
                                          {'x': Xb[:, dim], 'name': 'bad', 'type': 'histogram'},
                                          {
                                          'x': X[:, dim][0]*np.ones(2),
                                          'y': np.linspace(0,200, 2),
                                          'name': input_value,
                                          'line': {'width': 3,'dash': 'dot'}}],
                                 'layout': {}
                                 }
                                 ),
                       html.Div([
                                 html.H2('Status: {}'.format(outcome), style={'color':color}),
                                 html.H2('Metrics: {}'.format(metrics)),
                                 html.Img(src=link)
                                 ]),
                       
                       html.Div([
                                 html.P('pep8 errors: {}'.format(GP.pep8)),
                                 html.P('commits per time: {}'.format(GP.commits_per_time))
                                 ]),
                       ])
    return div_output

####### Main App Callback
@app.callback(
              Output(component_id='my-div', component_property='children'),
              [Input('button', 'n_clicks')],
              state=[State(component_id='repo', component_property='value'),
                     State(component_id='metrics', component_property='values')])
def update_output_div(n_clicks, input_value, metrics):
    r = gf.get_request('https://api.github.com/repos/%s'%input_value)
    if r.ok:
        item = json.loads(r.text or r.content)
        GP = get_features(item)
        with open('users_test/GP_%s.pkl'%item['name'], 'wb') as output_:
            pickle.dump(GP, output_)
        score, Xr = mod.classify_repo(GP)
        X = np.load('models/X.npy')
        Xb = np.load('models/Xb.npy')
        return output(input_value, GP, X, Xb, Xr, score, metrics)

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug = True)
    #app.run(debug=True, use_reloader=False, port=5000, host='0.0.0.0')
