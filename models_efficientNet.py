import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.ops import Graph
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve
# from classification_models.tfkeras import Classifiers

# BATCH_SIZE = 32
# DROPOUT = 0.25
# LR = 2e-3
# OHEM_RATE = 0.5
# NUM_EPOCHS = 5
# NUM_CLASSES = 2
# IMG_SIZE = (227, 227, 3)

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


def model_build(num_train_samples, num_epochs = 10, batch_size=8, lr=2e-3, ohem_rate=0.5,
                drop_out = 0.2, img_size = (227, 227, 3), model_name='EfficientNetB7'):  # original `densenet121`  Original batch size: 32
    total = ((num_train_samples + batch_size - 1) // batch_size) * num_epochs
    # model_structure, _ = Classifiers.get(model_name)
    # backbone = model_structure(input_shape=img_size, weights='imagenet', include_top=False)
    ##backbone = keras.applications.efficientnet.EfficientNetB0(include_top=False, input_shape=img_size) 
    # include_top: whether to include fully connected layer at the top of the network
    # output: a keras.Model instance

    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Dropout(DROPOUT)(x)
    # x = keras.layers.Dense(512, activation='relu')(x)
    # x = keras.layers.BatchNormalization()(x)

    # gem_out = gem()(backbone.output)
    ##pooling = keras.layers.GlobalAveragePooling2D() #gem()
    # gap1 = pooling(backbone.get_layer('pool4_relu').output)
    ## gap2 = pooling(backbone.get_layer('pool2_relu').output) # previously in densenet. Not present in efficientNet
    ## gap3 = pooling(backbone.get_layer('pool3_relu').output)
    ## x = keras.layers.Concatenate(axis=-1)([gap2, gap3])

    # x = keras.layers.BatchNormalization()(gap)
    ##x = keras.layers.Dropout(drop_out)(x)
    ##out = keras.layers.Dense(1, activation='sigmoid', name='pneumonia')(x)

    model = keras.applications.efficientnet.EfficientNetB7(include_top=False, input_shape=img_size)  # baseline model; try include top as in Luz et al.

    # freeze the pretrained weights
    #model.trainable = False

    # Rebuild top
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    #x = keras.layers.Dense(F, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)  # Luz et al
    x = keras.layers.Dropout(0.7, name="dropout1")(x)  # Luz et al
    x = keras.layers.Dense(512)(x)  # Luz et al
    x = keras.layers.BatchNormalization()(x)  # Luz et al
    #x = keras.activations.relu(x)  # add relu activation
    x = keras.layers.Dropout(0.5, name="dropout2")(x)  # Luz et al
    x = keras.layers.Dense(128)(x)  # Luz et al
    x = keras.layers.BatchNormalization()(x)  # Luz et al
    #x = keras.activations.relu(x)  # add relu activation

    x = keras.layers.Dropout(drop_out, name="top_dropout")(x) # remove bc Luz et al
    outputs = keras.layers.Dense(1, activation="sigmoid", name="pred")(x) # why the num_classes is 1 here?
    # APPISCI paper used softmax classifier
    model = keras.models.Model(model.input, outputs) # the example use layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))?
    ##model = keras.models.Model(backbone.input, out) # attach other layers after backbone
    #?Need to change image dimension?
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