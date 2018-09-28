from keras.models import load_model
# import imgaug as ia
# from imgaug import augmenters as iaa
import shutil
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

from fr_utils import *
from inception_blocks_v2 import *

if (os.path.exists("./facenet_keras.h5") == False):
    os.system('sudo gsutil -m cp gs://theia-ai/data/facenet_keras.h5 ./')

# FRmodel = load_model('./facenet_keras.h5')
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

cascPath = "./haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

clf = None
le = None

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


class Register(object):
    def __init__(self, cascade_path, filepath):
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.filepath = os.path.abspath(filepath) + '/student'

    def capture_images(self, name='Unknown', n_img_per_person=20):
        video = cv2.VideoCapture(0)
        margin = 10
        counter = 0
        timer = 0

        # checks if the video camera is on
        if video.isOpened():
            is_capturing, _ = video.read()
        else:
            is_capturing = False

        if not os.path.exists('student'):
            os.mkdir('student')
        if os.path.isdir(self.filepath + '/' + name):
            return ('Student is already registered')
        else:
            os.mkdir(self.filepath + '/' + name)

        while counter < n_img_per_person:
            _, frame = video.read()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # faces = self.detector.detectMultiScale(frame,
            #                                        scaleFactor=1.1,
            #                                        minNeighbors=3,
            #                                        minSize=(100, 100))

            if len(frame) != 0 and timer % 300 == 50:  # timer - time between
            # saving

                face_cut, faces = crop_face(frame)
                # face = faces[0]
                # (x, y, w, h) = face
                # left = x - margin // 2
                # right = x + w + margin // 2
                # bottom = y - margin // 2
                # top = y + h + margin // 2
                # img = frame[bottom:top, left:right, :]
                if(len(face_cut) >0) :
                    cv2.imwrite(self.filepath + '/' + name + '/' + str(counter) + '.png', face_cut)
                counter += 1

            # cv2.rectangle(frame,
            #                               (left-1, bottom-1),
            #                               (right+1, top+1),
            #                               (255, 0, 0), thickness=2)
            #                 plt.imshow(frame)
            if counter == n_img_per_person:
                break
            timer += 50
        video.release()
        return ("Student is Registered")

    def recapture_images(self, name='Unknown'):
        if os.listdir(self.filepath + '/' + name):
            shutil.rmtree(self.filepath + '/' + name)
        x = self.capture_images(name)
        return ('Student has been re-registered')


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)

    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

    basic_loss = pos_dist - neg_dist + alpha

    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

    return loss


def cut_faces(image, faces_coord):
    (x, y, w, h) = faces_coord
    return image[y:h, x : x + (w-x)]


def plt_show(image, title=""):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, cmap="Greys_r")
    plt.show()


def crop_face(image):
    # imagePath = image_path

    # Read the image
    # image = cv2.imread(imagePath)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cut = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        face_cut,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(100, 100)
    )


    # print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        box = np.array([x, y, x + w, y + h])
        face_cut = cut_faces(image, box.astype("int"))
        # face_cut = prewhiten(face_cut)
        # plt_show(face_cut)
        break
    return face_cut, faces


def auger(images_array, names, times=4):
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                           sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           # convert images into their superpixel representation
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 7)),
                               # blur image using local means with kernel sizes between 2 and 7
                               iaa.MedianBlur(k=(3, 11)),
                               # blur image using local medians with kernel sizes between 2 and 7
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                           # search either for all edges or for directed edges,
                           # blend the result with the original image using a blobby mask
                           iaa.SimplexNoiseAlpha(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0.5, 1.0)),
                               iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           ])),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                           # add gaussian noise to images
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                           ]),
                           iaa.Invert(0.05, per_channel=True),  # invert color channels
                           iaa.Add((-10, 10), per_channel=0.5),
                           # change brightness of images (by -10 to 10 of original value)
                           iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                           # either change the brightness of the whole image (sometimes
                           # per channel) or change the brightness of subareas
                           iaa.OneOf([
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               iaa.FrequencyNoiseAlpha(
                                   exponent=(-4, 0),
                                   first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                   second=iaa.ContrastNormalization((0.5, 2.0))
                               )
                           ]),
                           iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           iaa.Grayscale(alpha=(0.0, 1.0)),
                           sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                           # move pixels locally around (with random strengths)
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           # sometimes move parts of the image around
                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )
    for time in range(times) :
        face_aug = seq.augment_images(images_array)
        for i in range(len(face_aug)):
            ia.seed(np.random.randint(10000, size=1)[0])
            face = face_aug[i]
            filename = names[i]
            cv2.imwrite("./data/augs/" + str(time) + '-' + filename, face)


def augment(data_path):
    temp = []
    count = 0
    names = []
    for filename in os.listdir(data_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            name = os.path.splitext(filename)[0]
            face = cv2.imread(data_path + filename)
            temp.append(face)
            names.append(filename)
    auger(temp, names)


# os.system('sudo gsutil -m cp -R gs://theia-ai/data_msan/ ./data/')
# os.system('sudo mv ./data/data_msan/* ./data/')
# os.system('sudo rm -r ./data/data_msan ')
# if (os.path.isdir("./data/crops/") == False):
#     os.system('sudo mkdir ./data/crops')
# if (os.path.isdir("./data/augs/") == False):
#     os.system('sudo mkdir ./data/augs')


# for filename in os.listdir('./data/'):
#     if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
#         name = os.path.splitext(filename)[0]
#         face,_ = crop_face(cv2.imread('./data/'+filename))
#         cv2.imwrite("./data/crops/" + filename, face)


# augment('./data/crops/')


FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

database = {}


# for filename in os.listdir('./data/crops/'):
#     imgs = []
#     if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
#         name = os.path.splitext(filename)[0]
#         # print ('images/crops/'+ filename)
#          im = cv2.imread('./data/crops/'+filename, 1)
#         imgs.append(img_to_encoding(im, FRmodel))
#         database[name] = np.array(imgs)

# for filename in os.listdir('./data/augs/'):
#     if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
#         name = os.path.splitext(filename)[0]
#         # print ('images/crops/'+ filename)
#           im = cv2.imread('./data/augs/'+filename, 1)
#         database[name] = img_to_encoding(im, FRmodel)

def create_encoding_db():
    print ('------recalculating embeddings-----')
    student_path = os.path.abspath('./student')
    for name in [name.split('/')[-2] for name in glob.glob(student_path+'/*/')]:
          if name not in database:
            imgs = []
            for image_path in glob.glob(student_path+'/'+name+'/*.png'):
                # print(image_path)
                if (os.path.getsize(image_path) > 0):
                    im = cv2.imread(image_path, 1)
                    # face, faces_cord = crop_face(im)
                    img = img_to_encoding(im, FRmodel)
                    imgs.append(img)
            database[name] = np.array(imgs)
    return


def train_svm():
    labels = []
    embs = []
    print ('database-----', len(database), database.keys())
    for name, imgs in database.items():
        labels.extend([name] * len(imgs))
        embs.extend(imgs)
    # print(imgs[0].shape)
    # print(len(embs))
    embs = np.concatenate(embs)
    le_loc = LabelEncoder().fit(labels)
    y = le_loc.transform(labels)
    print(embs.shape)
    print(y.shape)
    clf_loc = SVC(kernel='linear', probability=True).fit(embs, y)
    pickle.dump(clf_loc, open('clf', 'wb'))
    pickle.dump(le_loc, open('le', 'wb'))
    return clf_loc, le_loc


def retrain():
    print('---inside retrain--')
    global clf, le
    create_encoding_db()
    clf, le = train_svm()
    return


retrain()

# for folder in glob.glob('./student/*'):
#     c = 0
#     for filename in os.listdir(folder):
#         if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
#             name = os.path.splitext(filename)[0]
#             path = folder + filename
#               im = cv2.imread(path, 1)
#             database[str(name)+'-'+str(c)] = img_to_encoding(im, FRmodel)
#             c += 1
