# Pneumonia detection on chest X-ray images using ensemble method

Final Project for OMSCS CS7643 Deep Learning 

## Objective

Using the RSNA pneumonia chest X-ray images for pneumonia classification, we ensemble various popular computer vision 
(CV) model architectures to incorporate more recent model architecture (e.g., EfficientNet) and data perturbation to
further improve model external validity.


## Environment Setup

- Local
    - `git clone https://github.gatech.edu/zwang3313/CS7643_Project.git`
    - `git checkout jason`
    - `cd cs7643_project`
    - `conda env create -f environment.yaml`
    - In case the model not included in `keras.applications`, need to install classification models Zoo - Keras (and
      TensorFlow Keras) \
      `pip install git+https://github.com/qubvel/classification_models.git`
    - Download Kaggle datafiles \
      https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
    - Run data pipeline scripts \
      `python tfrecords_writing.py`
- Colab
    - One stop tutorial to download data, set up Github on Colab and create tfrecords \
      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17YMKmI_K5QmQTib8wj_dQNNRr9c8yr9O)

## Data pipelines

**Tfrecord_writing.py**

* Reformat and resize images
* Write image files into batch (average size of 32 with 1:1 case-control ratio)
* Write into `tfrecords`
* Parameters

```
python tfrecords_writing.py --help

optional arguments:
  -h, --help            show this help message and exit
  --width WIDTH, -wt WIDTH
                        Resized image width (default is 227)
  --height HEIGHT, -ht HEIGHT
                        Resized image height (default is 227)
  --data_dir DATA_DIR, -dd DATA_DIR
                        Zip file path (default is rsna-pneumonia-detection-challenge.zip)
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        Batch size in each tfrecord file (default is 32)
  --num_shards NUM_SHARDS, -ns NUM_SHARDS
                        Number of shards (default is 200, must be less than 945)
```

* Execution

```
python tfrecords_writing.py
```

* Output
  * There are total 26,684 (6,012 cases and 20,672 controls) in original dataset
  * For this project, we randomly pick (3212 cases and 3200 controls) to form 200 tfrecords with average batch size of 32
  * 200 `tfrecord` (specified by `num_shards`) files saved in `./tfrecords`

## Model training
See `Training_[Model Name].ipynb` notebook for each model training implementation details and outputs.

Or open notebook in colab (This is for Densenet121)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SyVfCMpq0cJDoEylfMXY5uKJha0oniUi?usp=sharing)

Please note that Training_vgg16.ipynb is only tested in local Windows machine, for running it in Colab or Linux, it may need some adaptations.

## Data augmentation
See `image_aug` function in `utils_[Model Name].py` for data augmentation strategies for each model type.

## Model architecture, training strategies and callbacks
See `models_[Model Name].py` for data augmentation strategies for each model type.

## Project report
https://www.overleaf.com/project/62333ea55f10390439d64511

