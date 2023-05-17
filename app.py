from flask import Flask, request, redirect, url_for, render_template, send_from_directory, jsonify, send_file
import os
from waitress import serve

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/image')
def get_image():
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'],'image.jpg'), mimetype='image/jpg')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file', filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return render_template('uploaded.html', filename=filename)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
   # app.run(debug=True)
    serve(app, host='0.0.0.0', port=8080)
