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

app = dash.Dash()
#app = dash.Dash(__name__)
server = app.server
#my_css_url = "https://unpkg.com/normalize.css@5.0.0"
#app.css.append_css({"external_url": my_css_url})

app.layout = html.Div([
                       html.H1('git-screened', style={'font-style': 'Courier New',
                               'font-size': 25, 'text-align': 'center'}),
                       html.Label('Repository Name (Format: user/repo)', style={'text-align': 'center'}),
                       dcc.Input(id='repo', value='', type="text"),
                       html.Button('Search', id='button'),
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
    GP = gs.Github_Profile([])
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
def output(input_value, GP, score):
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
    return html.Div([
                     html.H1('Results for Repository: "{}"'.format(input_value)),
                     html.Div([
                               html.H2('Status: {}'.format(outcome), style={'color':color}),
                               html.Img(src=link)
                              ]),

                     html.Div([
                               html.P('pep8 errors: {}'.format(GP.pep8)),
                               html.P('commits per time: {}'.format(GP.commits_per_time))
                               ]),
                    ])
#elif score == -1:


####### Main App Callback
@app.callback(
              Output(component_id='my-div', component_property='children'),
              [Input('button', 'n_clicks')],
              state=[State(component_id='repo', component_property='value')])
def update_output_div(n_clicks, input_value):
    r = gf.get_request('https://api.github.com/repos/%s'%input_value)
    if r.ok:
        item = json.loads(r.text or r.content)
        GP = get_features(item)
#        with open('GP_TC.pkl', 'wb') as output_:
#            pickle.dump(GP, output_)
        score = mod.classify_repo(GP)
        return output(input_value, GP, score)

if __name__ == '__main__':
    #app.run_server(host='0.0.0.0', debug = True)
    app.run(debug=True, use_reloader=False, port=5000, host='0.0.0.0')
