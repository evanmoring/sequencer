from flask import Flask, render_template, jsonify, request
import threading
import time
import sys
sys.path.append('../')
import sequencer
import time

bpm = 250
beats = 8 
stop_flag = False

app = Flask(__name__)
@app.route("/")
def hello_world():
    return f"<p>Hello, World! {render_template('home.html')}</p>"

@app.route('/request', methods = ['POST'])
def request_rec():
    global seq

    t = request.json['type']
    x = request.json['x']
    a = request.json['action']
    print(t)
    if a  == "clear_all":
        print("CLEAR")
        seq.clear()
        
    if a == "remove":
        print("TYPE IS REMOVE")
        seq.remove_waveform(x * beats, sequencer.Snare())
    if a == "add":
        seq.place_waveform(x * beats, sequencer.Snare())
    return jsonify(isError= False,
                message= "Success",
                statusCode= 200,
                data= 0), 200

def toggle_start_stop():
    global stop_flag
    if not stop_flag:
        seq.stop()
    else:
        seq.play()


def background_task():
    seq.play()

if __name__ == "__main__":
    seq = sequencer.Sequencer(beats, bpm)
    thread = threading.Thread(target=background_task)
    thread.daemon = True
    thread.start()
    app.run()

