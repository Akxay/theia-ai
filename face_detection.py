import cv2
import numpy as np


class FaceDetector(object):
    def __init__(self, prototxt, caffeemodel):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffeemodel)

    def detect(self, image):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()
        return detections


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


def cut_faces(image, faces_coord):
    (x, y, w, h) = faces_coord
    return image[y:h, x: x + (w - x)]


def draw_rectangle(image, coords):
    (x, y, j, k) = coords
    cv2.rectangle(image, (x, y), (j, k), (0, 0, 255), 2)


def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("people/")]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("people/" + person):
            images.append(cv2.imread("people/" + person + '/' + image,
                                     0))
            labels.append(i)
    return (images, np.array(labels), labels_dic)
