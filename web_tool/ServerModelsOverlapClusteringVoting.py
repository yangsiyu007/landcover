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
from web_tool.ServerModelsOverlapClustering import OverlapClustering
from ServerModelsNIPS import KerasDenseFineTune

from training.pytorch.utils import save_visualize
from web_tool import ROOT_DIR
from web_tool.utils import represents_int

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
        # use for clusters-only view:
        #output_clustering_soft = self.clustering_server_model.run(naip_data, extent, on_tile=on_tile, collapse_clusters=True)
        #return output_clustering_soft

        # NN only view:
        output_neural_net_soft = self.fine_tuning_server_model.run(naip_data, extent, on_tile=on_tile)
        # (height width label)
        # return output_neural_net_soft

        # Actual algorithm

        output_clusterings_soft = self.clustering_server_model.run(naip_data, extent, on_tile=on_tile, collapse_clusters=False)
        # (height width cluster)

        # pdb.set_trace()

        outputs = []
        
        for output_clustering_soft in output_clusterings_soft:        
            height_cluster, width_cluster, num_clusters = output_clustering_soft.shape
            height_nn, width_nn, num_labels = output_neural_net_soft.shape

            assert height_cluster == height_nn
            assert width_cluster == width_nn

            output_clustering_argmax = output_clustering_soft.argmax(axis=-1)
            # (h w)
            output_clustering_one_hot = np.zeros((height_cluster, width_cluster, num_clusters))
            for i in range(output_clustering_argmax.shape[0]):
                for j in range(output_clustering_argmax.shape[1]):
                    cluster_id = output_clustering_argmax[i, j]
                    output_clustering_one_hot[i, j, cluster_id] = 1.0

            output_clustering_one_hot += 0.000001
                    
            output = np.zeros((height_cluster, width_cluster, num_labels))

            vote_radius = 25
            stride = 1
            diameter = 2 * vote_radius + 1
            
            c = torch.tensor(rearrange(output_clustering_one_hot, 'h w c -> () c h w')) # , dtype=torch.float
            l = torch.tensor(rearrange(output_neural_net_soft, 'h w l -> () l h w')) # , dtype=torch.float
            
            normalizations = torch.nn.functional.avg_pool2d(c, diameter, stride, padding=vote_radius, count_include_pad=False)
            # (1 c h w)            
            
            joint_labels_clusters = rearrange(
                torch.nn.functional.avg_pool2d(
                    torch.tensor(rearrange(
                        np.einsum('...c,...l->...cl', output_clustering_one_hot, output_neural_net_soft),
                        'h w c l -> () (c l) h w')
                    ),
                    diameter,
                    stride,
                    padding=vote_radius,
                    count_include_pad=False
                ),
                '() (c l) h w -> l c h w',
                c=num_clusters,
                l=num_labels
            )

            prob_label_given_cluster = joint_labels_clusters / normalizations
            # (l c h w) / (1 c h w) --> (l c h w)
            
            
            prob_label_given_point = np.einsum('lchw,hwc->hwl', prob_label_given_cluster, output_clustering_one_hot)
            # (height, width, num_labels)
            
            output = prob_label_given_point
            outputs.append(rearrange(output, 'h w l -> l h w'))

            # Put this in when ready to send a stream of outputs to the client
            # yield output


        base_path = '/mnt/blobfuse/pred-output/overlap-clustering'
        previous_saves = [int(subdir) for subdir in os.listdir(base_path) if represents_int(subdir)]
        if len(previous_saves) > 0:
            last_save = max(previous_saves)
        else:
            last_save = 0
        path = '%s/%d-final-predictions' % (base_path, last_save)
        
        inputs = torch.tensor([rearrange(naip_data, 'h w c -> c h w')])
        outputs = torch.tensor(outputs)
        
        save_visualize.save_visualize(inputs,
                                      torch.tensor(np.pad(outputs, ((0,0), (1,0), (0,0), (0,0)), mode='constant')), # add an extra entry for 0 class, with 0 probability assigned
                                      rearrange(output_neural_net_soft.argmax(axis=-1) + 1, 'h w -> () h w'),
                                      path, rand_colors=False)
        # save_visualize.save_visualize(inputs, rearrange(output_neural_net_soft, 'h w c -> c h w'), None, path, rand_colors=False)
            
        '''for i in range(num_clusters):
            cluster_mask = (output_clustering_hard == i) * 1.0
            # (h w)
            cluster_mask_labels = np.repeat(rearrange(cluster_mask, 'h w -> h w ()'), 4, axis=-1)
            # (h w num_labels)
            
            masked_neural_net_soft = output_neural_net_soft * cluster_mask_labels
            # (h w c) w/ 0s
            neural_net_votes = masked_neural_net_soft.sum(axis=0).sum(axis=0)
            # (num_labels)
            neural_net_votes = rearrange(neural_net_votes, 'l -> l () ()')
            # (num_labels 1 1)
            
            # assert sum(neural_net_votes) == np.count_nonzero(cluster_mask)
            # (Pdb) np.count_nonzero(cluster_mask)
            # 3504
            # (Pdb) sum(neural_net_votes)
            # 3503.8304080405505
            # Not quite equal... concern here?
            
            cluster_decisions = np.repeat(
                rearrange(cluster_mask, 'h w -> () h w'),
                num_labels,
                axis=0
            )
            # (c h w)
            
            cluster_decisions *= neural_net_votes
            # (num_labels h w)
            cluster_decisions = rearrange(cluster_decisions, 'l h w -> h w l')
            
            output += cluster_decisions'''

        
        return output
    
        
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
