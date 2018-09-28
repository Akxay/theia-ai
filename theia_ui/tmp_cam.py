import cv2
from flask import Flask, Response, stream_with_context

app = Flask(__name__)

class VideoCamera(object):
    def __init__(self, index=0):
        self.video = cv2.VideoCapture(index)
        self.index = index
        print (self.video.isOpened())

    def __del__(self):
        self.video.release()

    def get_frame(self, in_grayscale=False):
        _, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame


def gen():
    camera = VideoCamera()
    while True :
        frame = camera.get_frame()
        yield (frame)


@app.route('/')
def home():
    return Response(stream_with_context(gen()))


if __name__ == "__main__":
    # app.secret_key = os.urandom(11)
    app.run(host='0.0.0.0', port=8082, debug=True, threaded=True, use_reloader=False)
