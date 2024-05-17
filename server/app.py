from flask import Flask, request
# from connections import find_connections
import json

data = {}
with open('data.json') as json_file:
    data = json.load(json_file)

app = Flask(__name__)

@app.route("/")
def home():
    return "nyt crossed app backend"

@app.route("/connections", methods=['POST'])
def connections():
    if "words" not in request.json:
        return "No words passed to server", 401
    
    words = request.json["words"]
    
    print(f"Recieved: {words}")
    
    if type(words) is not list:
        return 'Invalid datatype for "words"', 402
    
    if len(words) != 16:
        return 'Invalid length for "words"', 403
    
    try:
        print("Finding Connections")
        # res = find_connections(words)
        return data
    except Exception as e:
        return f'Error: {e} occurred when finding connections', 500

@app.route("/wordle")
def wordle():
    return "idk"