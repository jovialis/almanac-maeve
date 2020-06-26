from flask import Flask
from flask import request
import pymongo

import search

app = Flask(__name__)


@app.route('/')
def hello_world():
    res_string = ""

    if 'q' in request.args:
        query = request.args['q']

        for res in search.search(query):
            res_score, res_name, res_desc = res
            res_string += '<b>' + res_name + '</b><br>'
            res_string += '' + str(res_desc) + '<br>'
            res_string += str(res_score) + '<br>'
            res_string += '-------------<br>'

    return res_string


if __name__ == '__main__':
    app.run()
