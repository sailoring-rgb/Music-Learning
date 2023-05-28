from flask import Flask, render_template

app = Flask(__name__, static_folder='./templates/images')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_music_route')
def generate_music():
    print("cheguei!")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=8000)