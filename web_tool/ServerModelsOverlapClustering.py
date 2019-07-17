import sys, os, time, copy

import numpy as np

import sklearn.base
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from keras import optimizers
import torch
from einops import rearrange
import time

from ServerModelsAbstract import BackendModel

from web_tool.frontend_server import ROOT_DIR

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
        self.correction_labels = None

        self.down_weight_padding = 40

        self.batch_x = []
        self.batch_y = []
        self.num_corrected_pixels = 0

        self.num_output_channels = 4
        self.device = torch.device('cuda:0')
        
    def run(self, naip_data, extent, on_tile=False):
        ''' Expects naip_data to have shape (height, width, channels) and have values in the [0, 255] range.
        '''
        # If we click somewhere else before retraining we need to commit the current set of training samples
        if self.correction_labels is not None and not on_tile:
            self.process_correction_labels()

        naip_data = naip_data / 255.0
        height = naip_data.shape[0]
        width = naip_data.shape[1]
        output = self.run_model_on_tile(naip_data)

        # Reset the state of our retraining mechanism
        if not on_tile:
            self.correction_labels = np.zeros((height, width, self.num_output_channels), dtype=np.float32)
            self.naip_data = naip_data.copy()
        
        return output

    def run_model_on_batch(self, batch_data, batch_size=32, predict_central_pixel_only=False):
        ''' Expects batch_data to have shape (none, 240, 240, 4) and have values in the [0, 255] range.
        '''
        output = output / 255.0
        output = self.model.predict(batch_data, batch_size=batch_size, verbose=0)
        output = output[:,:,:,1:]

        if predict_central_pixel_only:
            output = output[:,120,120,:]
        
        return output

    def retrain(self, number_of_steps=5, last_k_layers=3, learning_rate=0.01, batch_size=32, **kwargs):
        # Commit any training samples we have received to the training set
        self.process_correction_labels()

        # Reset the model to the initial state
        num_layers = len(self.model.layers)
        for i in range(num_layers):
            if self.model.layers[i].trainable:
                self.model.layers[i].set_weights(self.old_model.layers[i].get_weights())
            self.model.layers[i].trainable = False

        for i in range(num_layers-last_k_layers, num_layers):
            self.model.layers[i].trainable = True
        self.model.compile(optimizers.Adam(lr=learning_rate, amsgrad=True), "categorical_crossentropy")

        if len(self.batch_x) > 0:

            x_train = np.array(self.batch_x)
            y_train = np.array(self.batch_y)
            y_train_labels = y_train.argmax(axis=3)

            # Perform retraining
            history = []
            for i in range(number_of_steps):
                idxs = np.arange(x_train.shape[0])
                np.random.shuffle(idxs)
                x_train = x_train[idxs]
                y_train = y_train[idxs]
                
                training_losses = []
                for j in range(0, x_train.shape[0], batch_size):
                    batch_x = x_train[j:j+batch_size]
                    batch_y = y_train[j:j+batch_size]

                    actual_batch_size = batch_x.shape[0]

                    training_loss = self.model.train_on_batch(batch_x, batch_y)
                    training_losses.append(training_loss)
                history.append(np.mean(training_losses))
            beginning_loss = history[0]
            end_loss = history[-1]
            
            # Evaluate training accuracy - surrogate for how well we are able to fit our supplemental training set
            y_pred = self.model.predict(x_train)        
            y_pred_labels = y_pred.argmax(axis=3)
            mask = y_train_labels != 0
            acc = np.sum(y_train_labels[mask] == y_pred_labels[mask]) / np.sum(mask)
            
            # The front end expects some return message
            success = True
            message = "Re-trained model with %d samples<br>Starting loss:%f<br>Ending loss:%f<br>Training acc: %f." % (
                x_train.shape[0],
                beginning_loss, end_loss,
                acc
            )
        else:
            success = False
            message = "Need to add labels before you can retrain"

        return success, message

    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):        
        self.correction_labels[tdst_row:bdst_row+1, tdst_col:bdst_col+1, class_idx+1] = 1.0

    def process_correction_labels(self):
        '''Store labels from previous patch that have not yet been trained with.
        Initialize a new, empty set of correction labels.'''

        # TODO: actually implement this for overlap clustering.
        
        height = self.naip_data.shape[0]
        width = self.naip_data.shape[1]

        self.correction_labels = None

    def undo(self):
        return False, "Not implemented yet"

    def reset(self):

        # self.model = copy.deepcopy(self.old_model)
        self.batch_x = []
        self.batch_y = []
        self.naip_data = None
        self.correction_labels = None
        
    def run_model_on_tile(self, naip_tile, batch_size=32):
        ''' Expects naip_tile to have shape (height, width, channels) and have values in the [0, 1] range.
        '''
        height = naip_tile.shape[0]
        width = naip_tile.shape[1]
        img_for_clustering = rearrange(naip_tile, 'h w c -> c h w')
        
        p, mean, var, prior = self.run_clustering(img_for_clustering, n_classes=8, radius=25, n_iter=10, stride=8, warmup_steps=2, warmup_radius=200)
        # p: (clusters, height, width)

        label_img = np.array([
            p[0, :, :] + p[1, :, :],
            p[2, :, :] + p[3, :, :],
            p[4, :, :] + p[5, :, :],
            p[6, :, :] + p[7, :, :]
        ])
        
        output = rearrange(label_img, 'c h w -> h w c')

        return output
    
    ########### Overlap Clustering Code ##############

    def local_avg(self, data, radius, stride=1):
            w,h = data.shape[-2:]
            mean = torch.nn.functional.avg_pool2d(data.reshape(-1,1,w,h), 2*radius+1, stride=stride, padding=radius, count_include_pad=False)
            if stride>1: mean = torch.nn.functional.interpolate(mean, size=(w,h), mode='bilinear')
            return mean.view(data.shape)

    def local_moments(self, data, q, radius, stride=1, var_min=0.0001, mq_min=0.000001):
        mq = self.local_avg(q, radius, stride)
        mq.clamp(min=mq_min)
        weighted = torch.einsum('zij,cij->czij', data, q) #class,channel,x,y
        weighted_sq = torch.einsum('zij,cij->czij', data**2, q)
        mean = self.local_avg(weighted, radius, stride) / mq.unsqueeze(1)
        var = self.local_avg(weighted_sq, radius, stride) / mq.unsqueeze(1) - mean**2
        var = var.clamp(min=var_min)
        return mean, var

    def lp_gaussian(self, data, mean, var, radius, stride=1):
        #means: c,ch,x,y
        #data: ch,x,y
        #out: c,x,y
        m0 = -self.local_avg(1 / var, radius, stride)
        m1 = self.local_avg(2 * mean / var, radius, stride)
        m2 = -self.local_avg(mean**2 / var, radius, stride)
        L = self.local_avg(torch.log(var), radius, stride)
        return (m0*data**2 + m1*data + m2 - 2 * L).sum(1) / 2
    
    def prob_gaussian(self, data, prior, mean, var, radius, stride=1):
        lp = self.lp_gaussian(data, mean, var, radius, stride)
        p = lp.softmax(0) * prior
        p /= p.sum(0)
        p+=0.001
        p /= p.sum(0)
        return p

    def em(self, data, p, radius, stride=1):
        prior = self.local_avg(p, radius, stride)
        mean, var = self.local_moments(data, p, radius, stride)
        p_new = self.prob_gaussian(data, prior, mean, var, radius, stride)
        return p_new, mean, var, prior

    def run_clustering(self, image, n_classes, radius, n_iter, stride, warmup_steps, warmup_radius):
        t = time.time()
        
        data = torch.tensor(image).to(self.device)
        p = torch.rand((n_classes,) + image.shape[1:], dtype=torch.double).to(self.device)
        p /= p.sum(0)
        
        for i in range(n_iter):
            p, mean, var, prior = self.em(data, p, warmup_radius if i<warmup_steps else radius, stride)# stride if i<n_iter-1 else 1)
            
        p_ = p.cpu().numpy()
        mean_ = mean.cpu().numpy()
        var_ = var.cpu().numpy()
        prior_ = prior.cpu().numpy()
        
        print('total time: ', time.time() - t)
        
        return p_, mean_, var_, prior_
    
