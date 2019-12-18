import flask
import werkzeug
from flask import request, redirect, jsonify
from werkzeug.utils import secure_filename

import os

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config["IMAGE_UPLOADS"] = "./uploads"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]
    print(ext.upper())
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_CONTENT_LENGTH"]:
        return True
    else:
        return False


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    result_array = []
    print("in upload image method")

    if request.method == "POST":

        print("request is POST")

        if request.files:

            if 'file' not in request.files:
                print("file not there")
                return jsonify(result_array)

            file = request.files['file']
            if file.filename == '':
                print("No selected file")
                return jsonify(result_array)

            print('Trying to save file')
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['IMAGE_UPLOADS'], filename)
            print("path - ", path)
            file.save(path)
            print("file saved")

            result_array.append("f_16514932")
            result_array.append("f_12753691")
            result_array.append("f_17028422")
            result_array.append("f_16824992")
            result_array.append("f_16514932")
            result_array.append("f_12753691")
            result_array.append("f_15936972")
            result_array.append("f_4167993")

            return jsonify(result_array)

        else:
            print("request does not have any file")

    else:
        print("request is not POST, its - ", request.method)


    print("Upload Failed.. Try again")
    return jsonify(result_array)

app.run()















# import flask
# import os
# from flask import Flask, flash, request, redirect, url_for
# from werkzeug.utils import secure_filename
#
# UPLOAD_FOLDER = '/uploads'
# ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
#
# app = flask.Flask(__name__)
# app.config["DEBUG"] = True
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
#
# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit an empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(url_for('uploaded_file',
#                                     filename=filename))
#     return '''
#         <!doctype html>
#         <title>Upload new File</title>
#         <h1>Upload new File</h1>
#         <form method=post enctype=multipart/form-data>
#           <input type=file name=file>
#           <input type=submit value=Upload>
#         </form>
#         '''
#
# app.run()
#
#
# # @app.route('/upload', methods=['POST'])
# # def upload_file():
# #     print("test", request.files)
# #
# #     if 'file' not in request.files:
# #         return "NO file found"
# #
# #     file = request.files['file']
# #     file.save("static/test.jpeg")
# #     return "file saved successfully"
# #
# # app.run()