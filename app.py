# https://stackoverflow.com/questions/45736656/how-to-use-a-button-to-trigger-callback-updates

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import gitfeatures as gf
import json

app = dash.Dash()

app.layout = html.Div([
                       html.Label('Repository Name (format: user/repo)'),
                       dcc.Input(id='repo', value='', type="text"),
                       html.Button('Search', id='button'),
                       html.Div(id='my-div')
                       ])

@app.callback(
              Output(component_id='my-div', component_property='children'),
              [Input('button', 'n_clicks')],
              state=[State(component_id='repo', component_property='value')]
              )
def update_output_div(n_clicks, input_value):
    r = gf.get_request('https://api.github.com/repos/%s'%input_value)
    if r.ok:
        repoItems = json.loads(r.text or r.content)
        return 'Stargazers for "{}" is {}'.format(input_value, repoItems["stargazers_count"])
    else:
        return 'Couldnt get stats for "{}"'.format(input_value)

if __name__ == '__main__':
    app.run_server()
