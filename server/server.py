# -*- coding: utf-8 -*-
import os
import age_estimation
import algrithm
from flask import Flask, request, url_for, send_from_directory, render_template
from werkzeug import secure_filename
import time
import json

import base64
import cv2

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '../img/'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory("../img",
                               filename)


# web ver

@app.route('/cascade', methods=['GET', 'POST'])
def cascade():
    return upload_for_web(request, algrithm.cascade_japi_path, 2)


@app.route('/age', methods=['GET', 'POST'])
def age_estimation():
    return upload_for_web(request, algrithm.age_japi_path, 3)


@app.route('/gender', methods=['GET', 'POST'])
def gender_estimation():
    return upload_for_web(request, algrithm.gender_japi_path, 4)


def upload_for_web(request, func, type_id):
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            name = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], name)  # TODO：save file with unique name
            file.save(path)

            beg = time.time()
            jdata = json.dumps(func(path), indent=4)
            end = time.time()

            file_url = url_for('uploaded_file', filename=name)
            width, height = algrithm.get_img_size(path)
            return render_template("index.html", type_id=type_id, jdata=jdata, \
                                   width=width, height=height, \
                                   url=file_url, ok=True, time=end - beg)
    return render_template("index.html")


# api ver
@app.route('/api/v1.0/all/', methods=['GET', 'POST'])
def api_all():
    return upload_for_api(request, algrithm.all_japi_path)


@app.route('/api/v1.0/cascade/', methods=['GET', 'POST'])
def api_cascade():
    return upload_for_api(request, algrithm.cascade_japi_path)


@app.route('/api/v1.0/age/', methods=['GET', 'POST'])
def api_age():
    return upload_for_api(request, algrithm.age_japi_path)


@app.route('/api/v1.0/gender/', methods=['GET', 'POST'])
def api_gender():
    return upload_for_api(request, algrithm.gender_japi_path)


def upload_for_api(request, func):
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            name = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], name)
            file.save(path)

            beg = time.time()
            jdata = func(path)
            end = time.time()

            width, height = algrithm.get_img_size(path)

            jdata['time'] = 1.0 / (end - beg)
            jdata['width'] = width
            jdata['height'] = height

            print("cost:", 1.0 / (end - beg))
            return json.dumps(jdata, indent=4)
    return ""


@app.route('/api/test/', methods=['GET', 'POST'])
def upload_byte_for_api():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            name = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], name)
            file.save(path)
            return "saved"
    return "good"


@app.route('/', methods=['GET', 'POST'])
def indextest():
    print('hiro dayo!')
    return ""


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
