import flask
from flask import request, jsonify

import os
import urllib.request

UPLOAD_FOLDER = './uploads'

app = flask.Flask(__name__)
app.secret_key = "secret key"
app.config["DEBUG"] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 * 1024

