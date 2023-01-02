from flask import Flask, render_template, jsonify, request
import threading
import time
import sys
sys.path.append('../')
import sequencer
import time

app = Flask(__name__)
@app.route("/")
def hello_world():
    return f"<p>Hello, World! {counter} {render_template('home.html')}</p>"
#    return f"<p>Hello, World! {counter}</p>"

@app.route('/request', methods = ['POST'])
def request_rec():
    print(request.json['type'])
    print(request.json['x'])
    #print(request.headers)
    data = 0
    
    if sine.freq > 600:
        sine.freq = 300
    else:
        sine.freq = (sine.freq *1.5)
    sine.array = sine.generate_waveform()
    sine.apply_gain(1/ (sine.freq / 1000))
    print(sine.freq)

    return jsonify(isError= False,
                message= "Success",
                statusCode= 200,
                data= data), 200

def background_task():
    sine.apply_gain(1/ (sine.freq / 1000))
    sine.play()

#    global counter
#    while True:
#        time.sleep(1)
#        counter += 5
        #print(counter)

if __name__ == "__main__":
    sine = sequencer.Sine(300,sequencer.seconds_to_samples(2))
    counter = 5
    thread = threading.Thread(target=background_task)
    thread.daemon = True
    thread.start()
    app.run()

