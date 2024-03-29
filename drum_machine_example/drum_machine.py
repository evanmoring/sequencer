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
instrument_map = {
"snare": sequencer.Snare(),
"hihat": sequencer.Hihat(),
"kick": sequencer.Kick()
}

app = Flask(__name__)
@app.route("/")
def fill_page():
    return f"{render_template('drum_machine.html')}"

@app.route('/request', methods = ['POST'])
def request_rec():
    global seq
    i = request.json['instrument']
    x = request.json['x']
    a = request.json['action']

    if a  == "clear_all":
        seq.clear()
    if a == "remove":
        seq.remove_waveform(x * beats, instrument_map[i])
    if a == "add":
        seq.place_waveform(x * beats, instrument_map[i])

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

def sequence_thread():
    seq.play()

if __name__ == "__main__":
    seq = sequencer.Sequencer(beats, bpm)
    thread = threading.Thread(target=sequence_thread)
    thread.daemon = True
    thread.start()
    app.run()

