import numpy as np
import rasterio
import torch
from matplotlib import pyplot as pt
import time
import pdb
import os
from web_tool.utils import represents_int

device = torch.device('cuda:0')


def local_avg(data, radius, stride=1):
    w,h = data.shape[-2:]
    mean = torch.nn.functional.avg_pool2d(data.reshape(-1,1,w,h), 2*radius+1, stride=stride, padding=radius, count_include_pad=False)
    if stride>1: mean = torch.nn.functional.interpolate(mean, size=(w,h), mode='bilinear')
    return mean.view(data.shape)


def local_moments(data, q, radius, stride=1, var_min=0.0001, mq_min=0.000001):
    mq = local_avg(q, radius, stride)
    mq.clamp(min=mq_min)
    weighted = torch.einsum('zij,cij->czij', data, q) #class,channel,x,y
    weighted_sq = torch.einsum('zij,cij->czij', data**2, q)
    mean = local_avg(weighted, radius, stride) / mq.unsqueeze(1)
    var = local_avg(weighted_sq, radius, stride) / mq.unsqueeze(1) - mean**2
    var = var.clamp(min=var_min)
    return mean, var


def lp_gaussian(data, mean, var, radius, stride=1):
    #means: c,ch,x,y
    #data: ch,x,y
    #out: c,x,y
    m0 = -local_avg(1 / var, radius, stride)
    m1 = local_avg(2 * mean / var, radius, stride)
    m2 = -local_avg(mean**2 / var, radius, stride)
    L = local_avg(torch.log(var), radius, stride)
    return (m0*data**2 + m1*data + m2 - 2 * L).sum(1) / 2


def prob_gaussian(data, prior, mean, var, radius, stride=1):
    lp = lp_gaussian(data, mean, var, radius, stride)
    p = lp.softmax(0) * prior
    p /= p.sum(0)
    p+=0.001
    p /= p.sum(0)
    return p


def em(data, p, radius, stride=1):
    prior = local_avg(p, radius, stride)
    mean, var = local_moments(data, p, radius, stride)
    p_new = prob_gaussian(data, prior, mean, var, radius, stride)
    return p_new, mean, var, prior


def run_clustering(image, n_classes, radius, n_iter, stride, warmup_steps, warmup_radius, radius_steps):
    t = time.time()
    
    data = torch.tensor(image).to(device)
    p = torch.rand((n_classes,) + image.shape[1:], dtype=torch.double).to(device)
    p /= p.sum(0)
    
    n_iter = len(radius_steps)
    
    outputs = []
    for i in range(n_iter):
        p, mean, var, prior = em(data, p, radius_steps[i], stride)# stride if i<n_iter-1 else 1)
        
        p_ = p.cpu().numpy()
        mean_ = mean.cpu().numpy()
        var_ = var.cpu().numpy()
        prior_ = prior.cpu().numpy()
        
        # output_clusters_hard = p_.argmax(axis=0)
        # output_clusters_img = save_visualize.classes_to_rgb(output_clusters_hard, color_map)
        outputs.append(p_)
        yield p_, mean_, var_, prior_
        
    '''base_path = '/mnt/blobfuse/pred-output/overlap-clustering'
    previous_saves = [int(subdir) for subdir in os.listdir(base_path) if represents_int(subdir)]
    if len(previous_saves) > 0:
        last_save = max(previous_saves)
    else:
        last_save = 0
    path = '%s/%d' % (base_path, last_save + 1)

    inputs = torch.tensor([image])
    outputs = torch.tensor(outputs)
        
    # save_visualize.save_batch(outputs, path, 'output_clustering')
    # save_visualize.save_batch(inputs_visualize, path, 'input_clustering')
    save_visualize.save_visualize(inputs, outputs, None, path, rand_colors=True)'''
    
    p_ = p.cpu().numpy()
    mean_ = mean.cpu().numpy()
    var_ = var.cpu().numpy()
    prior_ = prior.cpu().numpy()
    
    print('total time: ', time.time() - t)
    
    return p_, mean_, var_, prior_
    
