# https://stackoverflow.com/questions/45736656/how-to-use-a-button-to-trigger-callback-updates
# https://dash.plot.ly/dash-core-components/upload

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import gitscraper as gs
import gitfeatures as gf
import json

app = dash.Dash()

app.layout = html.Div([
                       html.Label('Repository Name (format: user/repo)'),
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
    GP.stargazers = item['stargazers_count']
    
    # scrape forks
    GP.forks = item['forks_count']
    return GP

@app.callback(
              Output(component_id='my-div', component_property='children'),
              [Input('button', 'n_clicks')],
              state=[State(component_id='repo', component_property='value')])
def update_output_div(n_clicks, input_value):
    r = gf.get_request('https://api.github.com/repos/%s'%input_value)
    if r.ok:
        item = json.loads(r.text or r.content)
        GP = get_features(item)
        string = ('Stats for "{}"\n:'
                  'pep8 errors: {}\n'
                  'commits per time: {}\n'
                  ).format(input_value, GP.pep8, GP.commits_per_time)
        return string
    else:
        return 'Couldnt get stats for "{}"'.format(input_value)

if __name__ == '__main__':
    app.run_server()
