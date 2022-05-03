import os
import numpy as np
os.environ['TF_CUDNN_DETERMINISTIC']='1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.ops import Graph
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve

# set tensorflow global random seed
tf.random.set_seed(1234)

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


def ohem_loss(ytrue, ypred, batch_size=32, ohem_rate=0.5):
    result = K.binary_crossentropy(ytrue, ypred)
    loss = tf.sort(result, direction='DESCENDING')
    ohem_loss = K.mean(loss[:int(batch_size * ohem_rate)])
    return ohem_loss


def model_build(num_train_samples, num_epochs = 10, batch_size=32, lr=2e-3, ohem_rate=0.5,
                drop_out = 0.2, img_size = (227, 227, 3), model_name='VGG16'):
    total = ((num_train_samples + batch_size - 1) // batch_size) * num_epochs
    backbone = keras.applications.VGG16(include_top=False, input_shape=img_size)

    # backbone.trainable = False
    # gem_out = gem()(backbone.output)
    pooling = keras.layers.GlobalAveragePooling2D() #gem()

    x = pooling(backbone.output)
    x = keras.layers.Dropout(drop_out)(x)
    out = keras.layers.Dense(1, activation='sigmoid', name='pneumonia')(x)

    model = keras.models.Model(backbone.input, out)
    model.compile(
        keras.optimizers.Adam(learning_rate=keras.experimental.CosineDecay(lr, total, alpha=0)),
        loss='binary_crossentropy',
    )
    return model
    
def model_build_finetune(num_train_samples, num_epochs = 10, batch_size=32, lr=2e-3, ohem_rate=0.5,
                drop_out = 0.2, img_size = (227, 227, 3), model_name='VGG16'):
    total = ((num_train_samples + batch_size - 1) // batch_size) * num_epochs
    backbone = keras.applications.VGG16(include_top=False, input_shape=img_size)

    # backbone.trainable = False
    # x = gem()(backbone.output)
    # pooling1 = keras.layers.GlobalAveragePooling2D()
    pooling2 = keras.layers.GlobalMaxPooling2D()
    gap3 = pooling2(backbone.get_layer('block3_conv3').output)
    gap4 = pooling2(backbone.get_layer('block4_conv3').output)
    # gap5 = pooling1(backbone.get_layer('block5_conv3').output)

    x = keras.layers.Concatenate(axis=-1)([gap3, gap4])
    # x = keras.layers.Concatenate(axis=-1)([gap3, gap4, gap5])

    # x = pooling1(backbone.output) + pooling2(backbone.output)
    # x = pooling1(backbone.output)
    x = keras.layers.Dropout(drop_out)(x)
    # x = keras.layers.Dropout(drop_out) (gap3)
    out = keras.layers.Dense(1, activation='sigmoid', name='pneumonia')(x)

    model = keras.models.Model(backbone.input, out)
    model.compile(
        keras.optimizers.Adam(learning_rate=keras.experimental.CosineDecay(lr, total, alpha=0)),
        loss='binary_crossentropy',
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
            test_labels,
            finetuning):
        super(IntervalEval, self).__init__()
        self.model_outdir = model_outdir
        self.score_max = [-1] * 4
        self.loss = []
        self.val_loss = []
        # self.score_max_auc = -1
        self.valid_set = valid_set
        # self.len_valid_set = len_valid_set
        self.valid_labels = valid_labels
        self.test_set = test_set
        # self.len_test_set = len_test_set
        self.test_labels = test_labels
        self.best_model_f1 = None
        self.finetuning = finetuning
        self.best_epoch_f1 = None
        # self.best_model_auc = None

    def on_epoch_end(self, epoch, logs={}):
        val_pred = self.model.predict(self.valid_set, verbose=0)
        roc_auc, precision, recall, f1 = compute_metrics(self.valid_labels, val_pred)
        print(f'\nAUC: {roc_auc:.5f} Precision: {precision:.5f} Recall: {recall:.5f} F1: {f1: .5f}')
        if f1 > self.score_max[-1]:
            print(f'F1 improved from {self.score_max[-1]:.5f} to {f1:.5f}')
            self.score_max = [roc_auc, precision, recall, f1]
            self.best_model_f1 = self.model
            self.best_epoch_f1 = epoch
        # if roc_auc > self.score_max_auc:
        #     print(f'AUC improved from {self.score_max_auc:.5f} to {roc_auc:.5f}')
        #     self.score_max_auc = roc_auc
        #     self.best_model_auc = self.model
        self.loss.append(logs["loss"])
        self.val_loss.append(logs["val_loss"])

    def on_train_end(self, logs=None):
        self.test_preds = self.best_model_f1.predict(self.test_set)
        self.test_score = compute_metrics(self.test_labels, self.test_preds)
        test_auc, test_pre, test_rec, test_f1 = self.test_score
        val_auc, val_pre, val_rec, val_f1 = self.score_max
        if self.finetuning == True:
            self.f1_name = f'{self.model_outdir}_val{str(val_f1)[2:7]}_test{str(test_f1)[2:7]}_FineTuningWoLastLayerPreTuning.h5'
        else:
            self.f1_name = f'{self.model_outdir}_val{str(val_f1)[2:7]}_test{str(test_f1)[2:7]}_LastLayerTuning.h5'
        self.best_model_f1.save_weights(self.f1_name)

        print('-' * 20 + 'Model Saved!' + '-' * 20)
        print('-' * 20 + 'Val set metrics' + '-' * 20)
        print(f'AUC: {val_auc:.5f} Precision: {val_pre:.5f} Recall: {val_rec:.5f} F1: {val_f1: .5f}')
        print('-' * 20 + 'Test set metrics' + '-' * 20)
        print(f'AUC: {test_auc:.5f} Precision: {test_pre:.5f} Recall: {test_rec:.5f} F1: {test_f1: .5f}')
