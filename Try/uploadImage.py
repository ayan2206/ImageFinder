import flask
import werkzeug
from flask import request, redirect
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

    if request.method == "POST":

        if request.files:

            if 'file' not in request.files:
                print("file not there")

            file = request.files['file']
            if file.filename == '':
                print('No selected file')
                return "No selected file"

            # if file and allowed_image(file.filename):
            #     print("file allowed")
            #     return

            # imageFile = request.files['image']
            # filename = secure_filename(imageFile.filename)
            # path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # print("path - ", path)
            # file.save(path)
            # return "file saved"

            print('Trying to save file')
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['IMAGE_UPLOADS'], filename)
            print("path - ", path)
            file.save(path)
            return "file saved"






            if "filesize" in request.cookies:
                print("here")
                if not allowed_image_filesize(request.cookies["filesize"]):
                    print("Filesize exceeded maximum limit")
                    return redirect(request.url)

                image = request.files["image"]

                if image.filename == "":
                    print("No filename")
                    return redirect(request.url)

                if allowed_image(image.filename):
                    filename = secure_filename(image.filename)

                    image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                    print("Image saved")

                    return redirect(request.url)

                else:
                    print("That file extension is not allowed")
                    return redirect(request.url)

    return "try uploading image"

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