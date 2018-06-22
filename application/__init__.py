from flask import Flask
from application import views

app = dash.Dash(__name__)
server = app.server
