import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.ops import Graph
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve
from keras import layers
# from classification_models.tfkeras import Classifiers

# BATCH_SIZE = 32
# DROPOUT = 0.25
# LR = 2e-3
# OHEM_RATE = 0.5
# NUM_EPOCHS = 5
# NUM_CLASSES = 2
# IMG_SIZE = (227, 227, 3)
class Sharpen(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(Sharpen, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = np.array([[-2, -2, -2], 
                                [-2, 17, -2], 
                                [-2, -2, -2]])
        self.kernel = tf.expand_dims(self.kernel, 0)
        self.kernel = tf.expand_dims(self.kernel, 0)
        self.kernel = tf.cast(self.kernel, tf.float32)

    def call(self, input_):
        return tf.nn.conv2d(input_, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        
class gem(keras.layers.Layer):
    def __init__(self):
      super(gem, self).__init__()
      self.gm_exp = tf.Variable(3.0, dtype='float32')

    def call(self, x):
      return (tf.reduce_mean(
          tf.abs(x ** self.gm_exp), 
          axis=[1, 2], 
          keepdims=False
        ) + K.epsilon()) ** (1. / self.gm_exp)

# gm_exp = tf.Variable(3.0, dtype='float32')
# def gem():
#     lambda_layer = keras.layers.Lambda(lambda x: (tf.reduce_mean(tf.abs(x ** gm_exp),
#                                                                  axis=[1, 2], keepdims=False) + K.epsilon()) ** (
#                                                          1. / gm_exp))
#     lambda_layer.trainable_weights.extend([gm_exp])
#     return lambda_layer

def ohem_loss(ytrue, ypred, batch_size=32, ohem_rate=0.5):
    result = K.binary_crossentropy(ytrue, ypred)
    loss = tf.sort(result, direction='DESCENDING')
    ohem_loss = K.mean(loss[:int(batch_size * ohem_rate)])
    return ohem_loss



def model_build(num_train_samples, num_epochs = 10, batch_size=32, lr=2e-3, ohem_rate=0.5,
                drop_out = 0.2, img_size = (227, 227, 3), model_name='resnet50'):
    total = ((num_train_samples + batch_size - 1) // batch_size) * num_epochs
    # model_structure, _ = Classifiers.get(model_name)
    # backbone = model_structure(input_shape=img_size, weights='imagenet', include_top=False)
    # backbone = keras.applications.densenet.DenseNet121(include_top=False, input_shape=img_size)
    backbone = keras.applications.resnet50.ResNet50(include_top=False,
                              input_shape=img_size)
  
    # backbone.trainable = False
    # x = backbone.output
    pooling = keras.layers.GlobalAveragePooling2D() #gem()
    conv211 = pooling(backbone.get_layer('conv2_block1_1_relu').output)
    conv212 = pooling(backbone.get_layer('conv2_block1_2_relu').output)
    conv221 = pooling(backbone.get_layer('conv2_block2_1_relu').output)
    conv222 = pooling(backbone.get_layer('conv2_block2_2_relu').output)
    conv231 = pooling(backbone.get_layer('conv2_block3_1_relu').output)
    conv232 = pooling(backbone.get_layer('conv2_block3_2_relu').output)
    conv311 = pooling(backbone.get_layer('conv3_block1_1_relu').output)
    conv312 = pooling(backbone.get_layer('conv3_block1_2_relu').output)
    conv321 = pooling(backbone.get_layer('conv3_block2_1_relu').output)
    conv322 = pooling(backbone.get_layer('conv3_block2_2_relu').output)
    conv331 = pooling(backbone.get_layer('conv3_block3_1_relu').output)
    conv332 = pooling(backbone.get_layer('conv3_block3_2_relu').output)
    conv341 = pooling(backbone.get_layer('conv3_block4_1_relu').output)
    conv342 = pooling(backbone.get_layer('conv3_block4_2_relu').output) 
    conv451 = pooling(backbone.get_layer('conv4_block5_1_relu').output)
    conv452 = pooling(backbone.get_layer('conv4_block5_2_relu').output)
    conv461 = pooling(backbone.get_layer('conv4_block6_1_relu').output)
    conv462 = pooling(backbone.get_layer('conv4_block6_2_relu').output)
    conv511 = pooling(backbone.get_layer('conv5_block1_1_relu').output)
    conv512 = pooling(backbone.get_layer('conv5_block1_2_relu').output)
    conv521 = pooling(backbone.get_layer('conv5_block2_1_relu').output)
    conv522 = pooling(backbone.get_layer('conv5_block2_2_relu').output)
    conv531 = pooling(backbone.get_layer('conv5_block3_1_relu').output)
    conv532 = pooling(backbone.get_layer('conv5_block3_2_relu').output)

    # x = keras.layers.Concatenate(axis=-1)([conv211, conv212, conv212, conv222, conv231, conv232,
    #                     #conv311, conv312, conv321, conv322, conv331, conv332, conv342,#conv311, conv312, conv321, conv322, conv331, conv332, conv341, conv342 
    #                     #conv452, #conv451, conv452, conv461, conv462, 
    #                     conv511, conv512, conv521, conv522, conv531, conv532
    #                     ])
    
    # x = pooling(backbone.output)
    x = keras.layers.Concatenate(axis=-1)([conv222, conv311, conv312, conv321, conv322, conv331, conv332, conv342, conv452, conv532])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(drop_out)(x)
    out = keras.layers.Dense(1, activation='sigmoid', name='pneumonia')(x)

    model = keras.models.Model(backbone.input, out)
    model.compile(
        keras.optimizers.Adam(learning_rate=keras.experimental.CosineDecay(lr, total, alpha=0)),
        loss='binary_crossentropy',
        # loss = ohem_loss,
        # loss_weights={
        #     'root': 0.5,
        #     'vowel': 0.25,
        #     'consonant': 0.25
        #     }
    )
    return model


def compute_metrics(y_true, y_pred):
    roc_auc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, np.array(y_pred > 0.5).astype(int))
    recall = recall_score(y_true, np.array(y_pred > 0.5).astype(int))
    f1 = f1_score(y_true, np.array(y_pred > 0.5).astype(int))
    return roc_auc, precision, recall, f1


class IntervalEval(keras.callbacks.Callback):

    def __init__(
            self,
            model_outdir,
            valid_set,
            # len_valid_set,
            valid_labels,
            test_set,
            # len_test_set,
            test_labels):
        super(IntervalEval, self).__init__()
        self.model_outdir = model_outdir
        self.score_max = [-1] * 4
        # self.score_max_auc = -1
        self.valid_set = valid_set
        # self.len_valid_set = len_valid_set
        self.valid_labels = valid_labels
        self.test_set = test_set
        # self.len_test_set = len_test_set
        self.test_labels = test_labels
        self.best_model_f1 = None
        # self.best_model_auc = None

    def on_epoch_end(self, epoch, logs={}):
        val_pred = self.model.predict(self.valid_set, verbose=0)
        roc_auc, precision, recall, f1 = compute_metrics(self.valid_labels, val_pred)
        print(f'\nAUC: {roc_auc:.5f} Precision: {precision:.5f} Recall: {recall:.5f} F1: {f1: .5f}')
        if f1 > self.score_max[-1]:
            print(f'F1 improved from {self.score_max[-1]:.5f} to {f1:.5f}')
            self.score_max = [roc_auc, precision, recall, f1]
            self.best_model_f1 = self.model
        # if roc_auc > self.score_max_auc:
        #     print(f'AUC improved from {self.score_max_auc:.5f} to {roc_auc:.5f}')
        #     self.score_max_auc = roc_auc
        #     self.best_model_auc = self.model

    def on_train_end(self, logs=None):
        self.test_preds = self.best_model_f1.predict(self.test_set)
        self.test_score = compute_metrics(self.test_labels, self.test_preds)
        test_auc, test_pre, test_rec, test_f1 = self.test_score
        val_auc, val_pre, val_rec, val_f1 = self.score_max
        self.f1_name = f'{self.model_outdir}_val{str(val_f1)[2:7]}_test{str(test_f1)[2:7]}.h5'
        self.best_model_f1.save_weights(self.f1_name)

        print('-' * 20 + 'Model Saved!' + '-' * 20)
        print('-' * 20 + 'Val set metrics' + '-' * 20)
        print(f'AUC: {val_auc:.5f} Precision: {val_pre:.5f} Recall: {val_rec:.5f} F1: {val_f1: .5f}')
        print('-' * 20 + 'Test set metrics' + '-' * 20)
        print(f'AUC: {test_auc:.5f} Precision: {test_pre:.5f} Recall: {test_rec:.5f} F1: {test_f1: .5f}')
