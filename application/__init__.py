from flask import Flask
from application import views

app = Flask(__name__)
server = app.server
