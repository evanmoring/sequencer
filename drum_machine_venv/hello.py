from flask import Flask, render_template, jsonify, request
import threading
import time
import sys
sys.path.append('../')
import sequencer
import time

bpm = 250
beats = 16
seq = sequencer.Sequencer(beats, bpm)
#seq.place_waveform(4, sequencer.Snare())
#seq.place_waveform(8, sequencer.Snare())

app = Flask(__name__)
@app.route("/")
def hello_world():
    return f"<p>Hello, World! {render_template('home.html')}</p>"
#    return f"<p>Hello, World! {counter}</p>"

@app.route('/request', methods = ['POST'])
def request_rec():
    print(request.json['type'])
    print("X")
    print(request.json['x'])
    x = request.json['x']
    #print(request.headers)
    data = 0

    seq.place_waveform(x * beats, sequencer.Snare())
    print("target beat")
    print(x * beats)
    
    

    return jsonify(isError= False,
                message= "Success",
                statusCode= 200,
                data= data), 200

def background_task():
    seq.play()

#    global counter
#    while True:
#        time.sleep(1)
#        counter += 5
        #print(counter)

if __name__ == "__main__":
    thread = threading.Thread(target=background_task)
    thread.daemon = True
    thread.start()
    app.run()

