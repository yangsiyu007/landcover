import sys, os, time, copy

import numpy as np
import random

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

from web_tool import ROOT_DIR
from training.pytorch.utils import save_visualize
from web_tool.utils import represents_int
from clustering import OverlapCluster


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


class OverlapClustering(BackendModel):

    def __init__(self, gpuid=0):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

        feature_layer_idx = None
        
        self.naip_data = None
        self.previous_extent = None
        
        self.correction_labels = None
        self.correction_locations = None
        self.cluster_assignments = None

        self.down_weight_padding = 40

        self.batch_x = []
        self.batch_y = []
        self.num_corrected_pixels = 0
        
        self.num_output_channels = 4
        self.device = torch.device('cuda:0')

        self.overlap = OverlapCluster()
        
        
    def run(self, naip_data, extent, on_tile=False, collapse_clusters=True):
        ''' Expects naip_data to have shape (height, width, channels) and have values in the [0, 255] range.
        '''
        # If we click somewhere else before retraining we need to commit the current set of training samples
        if self.correction_labels is not None and not on_tile:
            self.process_correction_labels()

        naip_data = naip_data / 255.0
        
        path = '/mnt/blobfuse/pred-output/overlap-clustering/debug/2'

        inputs_to_save = torch.tensor([rearrange(naip_data, 'h w c -> c h w')])
        save_visualize.save_visualize(inputs_to_save,
                                      torch.zeros(inputs_to_save.shape),
                                      torch.zeros(inputs_to_save.shape).argmax(dim=-1),
                                      '%s/%d' % (path, 1))
        
        height = naip_data.shape[0]
        width = naip_data.shape[1]
        
        if extent == self.previous_extent:
            #try:
            output = self.run_updated_model_on_tile(naip_data)
            yield output
            #except:
            #    traceback.print_stack()
            #    pdb.set_trace()
        else:
            for output in self.run_model_on_tile(naip_data, collapse_clusters=collapse_clusters):
                self.previous_extent = extent
                # Reset the state of our retraining mechanism
                if not on_tile:
                    self.correction_labels = np.zeros((height, width, self.num_output_channels), dtype=np.float32)
                    self.naip_data = naip_data.copy()
        
                yield output

    # Currently not used.
    def run_model_on_batch(self, batch_data, batch_size=32, predict_central_pixel_only=False):
        ''' Expects batch_data to have shape (none, 240, 240, 4) and have values in the [0, 255] range.
        '''
        print('in run_model_on_batch')
        
        output = output / 255.0
        output = self.model.predict(batch_data, batch_size=batch_size, verbose=0)
        output = output[:,:,:,1:]

        if predict_central_pixel_only:
            output = output[:,120,120,:]
        
        return output

    def retrain(self, number_of_steps=5, last_k_layers=3, learning_rate=0.01, batch_size=32, **kwargs):
        # Commit any training samples we have received to the training set
        self.process_correction_labels()
        success = True
        message = 'Corrections received.'
        
        return success, message

    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):        
        self.correction_labels[tdst_row:bdst_row+1, tdst_col:bdst_col+1, class_idx] = 1.0

    def process_correction_labels(self):
        '''Store labels from current patch.
        '''

        # TODO: actually implement this for overlap clustering.
        # Right now, just throwing these corrections away.
        
        height = self.naip_data.shape[0]
        width = self.naip_data.shape[1]

        self.correction_locations = self.correction_labels.sum(axis=-1)
        # 1 in each location with a label, 0 elsewhere


    def undo(self):
        return False, "Not implemented yet"

    def reset(self):

        # self.model = copy.deepcopy(self.old_model)
        self.batch_x = []
        self.batch_y = []
        self.naip_data = None
        self.correction_labels = None
        
    def run_model_on_tile(self, naip_tile, batch_size=32, collapse_clusters=True):
        ''' Expects naip_tile to have shape (height, width, channels) and have values in the [0, 1] range.
        '''
        print('in run_model_on_tile')
        height = naip_tile.shape[0]
        width = naip_tile.shape[1]
        img_for_clustering = rearrange(naip_tile, 'h w c -> c h w')
        
        for (p, mean, var, prior) in overlap.run_clustering(img_for_clustering, n_classes=8, radius=25, n_iter=10, stride=8, warmup_steps=2, warmup_radius=200, radius_steps=([200]*2)):
            # p: (clusters, height, width)
            self.cluster_assignments = p
            
            if collapse_clusters:
                label_img = np.array([
                    p[0, :, :] + p[1, :, :],
                    p[2, :, :] + p[3, :, :],
                    p[4, :, :] + p[5, :, :],
                    p[6, :, :] + p[7, :, :]
                ])
            else:
                label_img = p
        
            output = rearrange(label_img, 'c h w -> h w c')

            yield output

    def run_updated_model_on_tile(self, naip_tile, batch_size=32):
        ''' Expects naip_tile to have shape (height, width, channels) and have values in the [0, 1] range.
        '''
        print('in run_updated_model_on_tile')
        height = naip_tile.shape[0]
        width = naip_tile.shape[1]
        
        soft_assignments = self.cluster_assignments
        # (cluster_ids height width)
        num_clusters = self.cluster_assignments.shape[0]
        
        hard_assignments = soft_assignments.argmax(axis=0)
        # (height width)

        correction_labels = self.correction_labels.argmax(axis=-1)
        
        cluster_lookups = []
        
        for i in range(num_clusters):
            cluster_locations = (hard_assignments == i) * 1  # 1 everywhere it is cluster i, 0 elsewhere
            corrections_in_cluster = self.correction_locations * cluster_locations
            corrections_in_cluster = 1 - corrections_in_cluster # invert for distance_transform_edt function
            # self.correction_locations: (height width)
            distances, closest_label_indices = morphology.distance_transform_edt(corrections_in_cluster, return_indices=True, return_distances=True)
            cluster_lookups.append(closest_label_indices)
                
        output_one_hot = np.zeros((height, width, 4))
        for y in range(height):
            for x in range(width):
                cluster_id = hard_assignments[y, x]
                nearest_label_idx = cluster_lookups[cluster_id][:, y, x]
                predicted_label = correction_labels[tuple(nearest_label_idx)]
                output_one_hot[y, x, predicted_label] = 1.0
                
        return output_one_hot

    
