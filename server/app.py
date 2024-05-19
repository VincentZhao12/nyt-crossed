from flask import Flask, request
from connections import find_connections
from flask_cors import CORS, cross_origin

app = Flask(__name__)

cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def home():
    return "nyt crossed app backend"

@app.route("/connections", methods=['POST'])
@cross_origin()
def connections():
    if "words" not in request.json:
        return "No words passed to server", 401
    
    words = request.json["words"]
    
    print(f"Recieved: {words}")
    
    if type(words) is not list:
        return 'Invalid datatype for "words"', 402
    
    n = len(words)
    if n < 4 or n % 4 != 0:
        return 'Invalid length for "words"', 403
    
    try:
        print("Finding Connections")
        res = find_connections(words)
        return res
    except Exception as e:
        return f'Error: {e} occurred when finding connections', 500

@app.route("/wordle")
def wordle():
    return "idk"