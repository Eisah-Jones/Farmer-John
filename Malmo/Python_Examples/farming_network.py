import os
import math
import json
import random
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from keras.metrics import top_k_categorical_accuracy

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

class FarmNetwork:
    def __init__(self, model_path = 'default_checkpoint'):
        with open(model_path + "/model.json", 'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights('models/farming/model.h5')
        self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics = [top_2_accuracy, 'categorical_accuracy'])
