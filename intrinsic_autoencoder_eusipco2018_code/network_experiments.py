#!/usr/bin/python3
#
# Evaluates the network, computes badpix performance and depth maps for
# all of the light fields in the training set.
#
# not sure yet how well it generalizes to light fields it has never seen
# before.
#

from queue import Queue
import time
import code
import os
import numpy as np
import scipy.misc

import pycuda.autoinit
from pycuda import gpuarray;

# plotting
import matplotlib
import matplotlib.pyplot as plt

# todo: get rid of this (or empty files)
nlabels = 141
#nlabels = 350
#nlabels = 512

import lf_cnn_data_v3 as data
data = data.read_lf_ap_data( '/data/cnns/training_data/', 'synthetic_patches_v3.hdf5', nlabels=nlabels )
data = data.train

# timing and multithreading
import _thread

# python tools for our lf database
import file_io

# light field GPU tools
import lf_tools

# evaluator thread
from cnn_compute_disp_map import cnn_compute_disp_map


# I/O queues for multithreading
inputs = Queue( 15*15 )
outputs = Queue( 15*15 )

#model_id = 'drsoap_v8b_141_all_noise'
#model_id = 'drsoap_v8b_anf3'
model_id = 'drsoap_v8b_anf4'
#model_id = 'drsoap_v8b_141_pretrain'
#model_id = 'drsoap_v8b_141_all_noise'
model_path = './networks/' + model_id + '/model.ckpt'

#model_path = './networks/drsoap_v8b/model.ckpt'
#model_path = './networks/drsoap_v8b_anf2/model.ckpt'
#model_path = './networks/drsoap_v8b_anf_retrain/model.ckpt'
#model_path = './networks/drsoap_v8b_141_all_noise/model.ckpt'
model_path = '/data/cnns/trained_models/170708/drsoap_v8b_anf3/model.ckpt'

from evaluator_v8 import evaluator_thread

_thread.start_new_thread( evaluator_thread,
                          ( model_path, inputs,  outputs ))


# wait a bit to not skew timing results with initialization
time.sleep(1)


# create random patches of a certain structure and send them to network for
# evaluation

# target disparity is always zero

# batch size
bs_y = 32
bs_x = 32

# result (maybe later randomly initialize)
dmap = np.zeros( [bs_y, bs_x], np.float32 )


# compute angular patch stack. Careful, block sizes hardcoded to our scenario.
N = nlabels + 8
T = 9
S = 9
aps = np.zeros( [ bs_x*bs_y, N, T, S ], np.float32 )
cvs = np.zeros( [ bs_x*bs_y, 31,31 ], np.float32 )

# init angular patches (focal stack symmetry)
for i in range( bs_x*bs_y ):
  #center_patch = np.random.rand( T,S )
  #aps[ i, 74, :,: ] = center_patch
  for j in range( 74 ):
    patch = np.random.rand( T,S ) - 0.5
    aps[ i, j, :,: ] = patch
    aps[ i, N-j-1, :,: ] = np.flip( np.flip( patch, 0 ), 1 )


    
# fill rest of batch with dummy data, but encode patch location
batch = ( aps,
          cvs,
          0, 0 )

# debug break
#cv = cvs[ 0,:,: ]
#code.interact( local=locals() )

# push into evaluation queue
inputs.put( batch );

# in theory, we should be able to pass this if we have no results yet.
# in practice, strage things happen - no idea why
# (patch location seems to be wrong in patches received after main loop,
# or something)
#
result = outputs.get()
L = data.fractional_one_hot_to_labels( result[0] )
disp = data.labels_to_disparity( L )
print( np.mean( abs(disp) ))

# kill worker thread
inputs.put( () )
code.interact( local=locals() )

