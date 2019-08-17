# Credits: http://blog.miguelgrinberg.com/post/video-streaming-with-flask
# Mod by XC

from flask import Flask, render_template, Response, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import cv2
from time import time, sleep 
import pickle 
import logging 
from collections import defaultdict

from server.action_graph import ActGraph  


class ReusableForm(Form):
    name = TextField('Name:', validators=[validators.required()])
    definition = TextField('Def:', validators=[validators.required()])


def print_lines(lines):
    for line in lines:
        flash(line)


class WebServer:
    def __init__(self, data_queues, msg_queue, ip, port, display_fps=15):
        self.data_queues = data_queues
        self.ip = ip
        self.port = port
        self.msg_queue = msg_queue
        self.display_period = 1. / display_fps
        self.last_display_time = defaultdict(float)


    def run(self):
        self.log('start to run')
        app = Flask('__main__', template_folder='web/html', static_url_path = "", 
                                                        static_folder = "web/html")
        # app.config.from_object(__name__)
        app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

        def read_im(src):
            while True:
                im = self.data_queues[src].read()
                if im is None:
                    # print('empty input queue for %s' % src)
                    sleep(0.04)
                    continue 

                sleep(max(0, self.display_period - (time() - self.last_display_time[src])))
                self.last_display_time[src] = time() 

                ret, jpeg = cv2.imencode('.jpg', im)
                b = jpeg.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' 
                        + b + b'\r\n\r\n')


        @app.route('/', methods=['GET', 'POST'])
        def index():
            form = ReusableForm(request.form)
            if form.errors:
                print(form.errors)

            if request.method == 'POST':
                name = request.form['name']
                definition = request.form['definition']
         
                if form.validate():
                    act = ActGraph(name, definition)
                    print_lines(act.show())
                    self.msg_queue.write(pickle.dumps(act))
                else:
                    print_lines(['Error: All the fields are required.'])
         
            return render_template('index.html', form=form)

        ###### AUTO GENERATED #####

        self.log('app run')
        app.run(host=self.ip, port=self.port, threaded=True)


    def log(self, s):
        logging.debug('[WebServer] %s' % s)
