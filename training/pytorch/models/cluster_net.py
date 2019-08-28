import torch
import torch.nn as nn
import json
import os

from einops import rearrange

from training.pytorch.models.unet import Unet
from training.clustering.overlap_clustering import run_clustering
import utils

from training.pytorch.utils.save_visualize import crop_to_smallest_dimensions

import pdb

# from web_tool.ServerModelsOverlapClustering import OverlapClustering


class ClusterNet(nn.Module):

    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model
        self.output_clustering_soft = None
        
    def forward(self, x, hard_clustering=False):
        # x: (batch, height, width, channels), range [0, 1]
        output_neural_net_soft = self.model.forward(x).double()

        if not self.output_clustering_soft:
            x_new = rearrange(torch.tensor(x, dtype=torch.double).cuda(), '() h w c -> h w c')
            output_clusterings_soft = run_clustering(x_new, n_classes=8, radius=25, n_iter=10, stride=8, warmup_steps=2, warmup_radius=200, radius_steps=([25]*1))
            # Iterate to final clustering
            for output_clustering_soft in output_clusterings_soft: pass
            self.output_clustering_soft = output_clustering_soft


        cluster_assignments, mean, var, prior = self.output_clustering_soft
            
        num_clusters, height_cluster, width_cluster = cluster_assignments.shape
        _, num_labels, height_nn, width_nn = output_neural_net_soft.shape

        # <HACK> / wrong -- see neural net cropping/offset logic for more precise calculations
        offset_width = width_cluster % 2
        offset_height = height_cluster % 2        
        cluster_assignments_in_window = crop_to_smallest_dimensions(cluster_assignments[:,
                                                                                        :height_cluster-offset_height,
                                                                                        :width_cluster-offset_width],
                                                                    output_neural_net_soft[0],
                                                                    (1, 2))
        # </HACK>
        num_clusters, height_cluster, width_cluster = cluster_assignments_in_window.shape
        assert height_cluster == height_nn
        assert width_cluster == width_nn
        
        
        if hard_clustering:
            output_clustering = soft_to_hard(cluster_assignments_in_window)
        else:
            output_clustering = cluster_assignments_in_window
            
        output_clustering += 0.000001
        
        # output = torch.zeros((height_cluster, width_cluster, num_labels)).to(x.device)
        
        vote_radius = 25
        stride = 1
        diameter = 2 * vote_radius + 1

        pdb.set_trace()
        
        c = rearrange(output_clustering, 'c h w -> () c h w')
        l = rearrange(output_neural_net_soft, '() l h w -> () h w l')
        
        normalizations = torch.nn.functional.avg_pool2d(c, diameter, stride, padding=vote_radius, count_include_pad=False)
        # (1 c h w)
        
        joint_labels_clusters = rearrange(
            torch.nn.functional.avg_pool2d(
                rearrange(
                    torch.einsum('chw, blhw -> bclhw', output_clustering, output_neural_net_soft),
                    'b c l h w -> b (c l) h w'),
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
        
        
        prob_label_given_point = torch.einsum('lchw,chw->lhw', prob_label_given_cluster, output_clustering).to(x.device)
        # (num_labels, height, width)
        
        output = prob_label_given_point

        
        return output
