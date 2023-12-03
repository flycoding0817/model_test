# -*- coding: UTF-8 -*-
from flask import Flask, render_template, request, make_response
from flask import jsonify
import threading
import time
from flask_cors import *
import responder


format_time = time.strftime("%Y-%m-%d", time.localtime(time.time()))

def heartbeat():
    timer = threading.Timer(60, heartbeat)
    timer.start()
timer = threading.Timer(60, heartbeat)
timer.start()
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import re
zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
app = Flask(__name__,static_url_path="/static")
CORS(app, supports_credentials=True)

respons = responder.Responder()

@app.route('/message', methods=['POST','GET'])
def reply():
    screen = request.form['screen']
    region = request.form['region']
    resDic = respons.requestDistribution(screen, region)
    print("resDic: {}".format(resDic))
    return jsonify(resDic)

# @app.route("/")
# def index():
#     # return render_template("index_bootstrap.html")
#     return render_template("index_focus_rotation.html")

# 启动APP
if (__name__ == "__main__"):
    app.run(host = '10.209.22.222', port = 8811)