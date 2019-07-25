import sys, os, time, copy

import numpy as np

import sklearn.base
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from keras import optimizers
import torch
from einops import rearrange
import time
import pdb
from scipy.ndimage import morphology
import traceback

from ServerModelsAbstract import BackendModel
from ServerModelsOverlapClustering import OverlapClustering
from ServerModelsNIPS import KerasDenseFineTune

from server import ROOT_DIR

AUGMENT_MODEL = MLPClassifier(
    hidden_layer_sizes=(),
    activation='relu',
    alpha=0.001,
    solver='lbfgs',
    tol=0.1,
    verbose=False,
    validation_fraction=0.0,
    n_iter_no_change=10
)


class OverlapClusteringVoting(BackendModel):

    def __init__(self, model_fn, gpuid, superres=False, verbose=False):
        self.clustering_server_model = OverlapClustering(gpuid=gpuid)
        self.fine_tuning_server_model = KerasDenseFineTune(model_fn, gpuid, superres=superres, verbose=verbose)
        
    def run(self, naip_data, extent, on_tile=False):
        ''' Expects naip_data to have shape (height, width, channels) and have values in the [0, 255] range.
        '''
        self.clustering_server_model.run(naip_data, extent, on_tile=on_tile)
        self.fine_tuning_server_model.run(naip_data, extent, on_tile=on_tile)

        
        
    def retrain(self, **kwargs):
        # Commit any training samples we have received to the training set
        success_1, message_1 = self.clustering_server_model.retrain(**kwargs)
        success_2, message_2 = self.fine_tuning_server_model.retrain(**kwargs)

        return (success_1 and success_2), "Clustering: %s;\n Fine-tuning: %s" % (message_1, message_2)

    
    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
        self.clustering_server_model.add_sample(tdst_row, bdst_row, tdst_col, bdst_col, class_idx)
        self.fine_tuning_server_model.add_sample(tdst_row, bdst_row, tdst_col, bdst_col, class_idx)


    def undo(self):
        success_1, message_1 = self.clustering_server_model.undo()
        success_2, message_2 = self.fine_tuning_server_model.undo()
        
        return (success_1 and success_2), "Clustering: %s;\n Fine-tuning: %s" % (message_1, message_2)

    
    def reset(self):
        self.clustering_server_model.reset()
        self.fine_tuning_server_model.reset()
