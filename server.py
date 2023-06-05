import request, configparser
from flask import Flask, render_template, request

app = Flask(__name__, static_folder='./templates/images')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_music_route')
def generate_music():
    print("cheguei!")
    return render_template('generate.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)