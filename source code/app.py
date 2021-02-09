from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

def do():
    coco = "[TRUE]"

@app.route('/')
def home():
    return render_template('search-bar.html')

@app.route('/join', methods=['GET', 'POST'])
def post():
    coco = do()
    result = {
        "output": coco
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result = result)

if __name__ == "__main__":
    app.run()
