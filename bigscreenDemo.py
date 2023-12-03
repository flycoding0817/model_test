# -*- coding: UTF-8 -*-
from flask import Flask, render_template, request, make_response
from flask import jsonify
import threading
import time

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

INPUT_CONTEXT = " "

# @app.route('/message', methods=['POST'])
# def reply():
#     req_msg = request.form['msg']
#     user_ip = request.remote_addr
#     res_new = 'aatest'
#     return jsonify({'text': res_new})

@app.route("/")
def index():
    # return render_template("index_focus_rotation.html")
    return render_template("index_flex.html")

# 启动APP
if (__name__ == "__main__"):
    # app.run(host = '10.209.22.222', port = 8810)
    app.run(host='localhost', port=8810)