from flask import Flask, request, render_template, make_response, jsonify
import requests

app = Flask(__name__)


api_endpoint = 'http://localhost:5001/predict'

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image = uploaded_file
            payload = {"image": image}
            # url = 'http://localhost:5000/predict'
            r = requests.post(api_endpoint, files=payload).json()
            res = make_response(jsonify(r), 200)
            return res

    return render_template('upload.html')


if __name__ == "__main__":
    app.run()