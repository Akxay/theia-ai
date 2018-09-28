from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
from numpy import genfromtxt
from models import *
import pickle

np.set_printoptions(threshold=np.nan)


def who_is_it(image, database=database, model=FRmodel):
    """
    Implements face recognition by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """

    face, faces_cord = crop_face(image)
    # cv2.imwrite("./data/tmp/face.jpg", face)
    # print(face.shape)
    encoding = img_to_encoding(face, model)
    min_dist = 10
    identity = 'None'

    # for (name, db_enc) in database.items():
    #     for enc in db_enc:
    #         dist = np.linalg.norm(np.subtract(encoding, enc))
    #
    #         if dist < min_dist:
    #             min_dist = dist
    #             identity = name
    # print(encoding.shape)
    clf = pickle.load(open('clf', 'rb'))
    le = pickle.load(open('le', 'rb'))
    identity = clf.predict(encoding)
    identity = le.inverse_transform(identity)[0]
    min_dist = np.max(clf.predict_proba(encoding))
    print('----Predicted Name : ', identity, '   -----Pobability :', min_dist)

    pred_prob = clf.predict_proba(encoding)[0]
    if (len(pred_prob)):
        if all(pred_prob < 0.2):
            print("Not in the database.")
    # else:
        # print ("it's " + str(identity) + ", the distance is " + str(min_dist))

    if (isinstance(faces_cord, (list, np.ndarray)) == False):
        # faces_cord = (-1)
        return (min_dist, identity), faces_cord
    return (min_dist, identity), faces_cord[0]

