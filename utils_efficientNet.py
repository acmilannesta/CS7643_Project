
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
import numpy as np
import cv2
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Example, Features, Feature
from sklearn.metrics import recall_score
from pydicom import dcmread
from io import BytesIO


NUM_IMAGES = 1000
BATCH_SIZE = 32
DROPOUT = 0.25
LR = 2e-4
OHEM_RATE = 0.5
NUM_EPOCHS = 5
# n_grapheme_root, n_vowel_diacritic, n_consonant_diacritic = 168, 11, 7

# def cutbox(h=HEIGHT, w=WIDTH):
#     cut_ratio = np.random.beta(a=0.4, b=0.4)
#     cut_ratio = np.clip(cut_ratio, 0.2, 0.8)
#     h_cut_size = ((h - 1) * cut_ratio).astype('int')
#     w_cut_size = ((w - 1) * cut_ratio).astype('int')
#     x1 = np.random.randint(0, (h - 1) - h_cut_size)
#     y1 = np.random.randint(0, (w - 1) - w_cut_size)
#     x2 = x1 + h_cut_size
#     y2 = y1 + w_cut_size
#     return x1, x2, y1, y2, cut_ratio
#
#
# def rotate_image(image):
#     image_center = tuple(np.array(image.shape[1::-1]) / 2)
#     matrix = cv2.getRotationMatrix2D(image_center, np.random.randint(0, 361, 1), 1.0)
#     result = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,
#                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
#     return result


# def resize_image(img):
#     img = 255 - img
#     img = (img * (255.0 / img.max())).astype(np.uint8)
#     # return cv2.resize(img, (HEIGHT, WIDTH), interpolation = cv2.INTER_AREA)
#     return cv2.resize(img, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)


def read_images(img_files, file_dir, size):
    X = []
    with ZipFile(file_dir, 'r') as archive:
        for img_path in img_files:
            imgfile = archive.read(f"stage_2_train_images/{img_path}.dcm")
            # x = cv2.imdecode(np.frombuffer(imgfile, np.uint8), cv2.IMREAD_COLOR)
            x = cv2.cvtColor(dcmread(BytesIO(imgfile), force=True).pixel_array, cv2.COLOR_GRAY2RGB)
            x = cv2.resize(x, size, cv2.INTER_AREA)
            X.append(x)
    return X

def process_image(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = tf.io.encode_png(image)
    return image.numpy()

def process_label(x, n_class):
    x = keras.utils.to_categorical(x, n_class, dtype='int8')
    return x.tobytes()

def make_example(encoded_image, label):

    features = Features(feature={
        'image': Feature(bytes_list=BytesList(value=[encoded_image])),
        'label': Feature(int64_list=Int64List(value=[label])),
        # 'label_v': Feature(bytes_list=BytesList(value=[v])),
        # 'label_c': Feature(bytes_list=BytesList(value=[c]))
    })

    example = Example(features=features)

    return example.SerializeToString()

def decode_image(image_data, shape):
    image = tf.io.decode_png(image_data, channels=shape[-1])
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, shape) # explicit size needed for TPU
    return image

def image_aug(image, random_mode=True):
    random_num = np.random.rand()
    if random_num < 1/4:
        data_augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.2),
            ])
        image = data_augmentation(tf.expand_dims(image, axis=0))
    elif random_num < 3/4:
        image = tfa.image.random_cutout(
            # image,
            tf.expand_dims(image, axis=0), 
            mask_size = (100, 100), 
            constant_values = 1 
        )
    return tf.squeeze(image)

def parse_example(serialized, shape, data_aug=False):
    features = {'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)
                }
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)
    image_raw = parsed_example['image']  # Get the image as raw bytes.
    image = decode_image(image_raw, shape)  # Decode the raw bytes so it becomes a tensor with type.
    # label = tf.io.decode_raw(parsed_example['label'], tf.uint8)
    # label = tf.cast(parsed_example['label'], tf.int64)
    if data_aug:
        # image = image.numpy()
        # for i in range(len(image)):
        image = image_aug(image)


    return image, tf.cast(parsed_example['label'], tf.float32)

