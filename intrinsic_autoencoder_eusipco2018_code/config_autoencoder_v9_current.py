# define the configuration (hyperparameters) for the residual autoencoder
# for this type of network.


# NETWORK MODEL NAME
network_model = 'current_full'

# CURRENT TRAINING DATASET
data_path = '/home/aa/Python_projects/Data_train/'
# data_path = "H:\\trainData\\"
training_data = [
    # data_path+'lf_patch_intrinsic_test.hdf5',
  data_path+'lf_patch_intrinsic_100.hdf5',
  data_path+'lf_patch_intrinsic_200.hdf5',
  data_path+'lf_patch_intrinsic_lytro.hdf5',
  data_path+'lf_patch_intrinsic_300.hdf5',
  data_path+'lf_patch_intrinsic2_diffuse_96.hdf5',
  data_path+'lf_patch_intrinsic_400.hdf5',
  data_path+'lf_patch_intrinsic_hci.hdf5',
  data_path+'lf_patch_intrinsic_500.hdf5',
  data_path+'lf_patch_intrinsic3_diffuse_96.hdf5',
  data_path+'lf_patch_intrinsic_600.hdf5',
  data_path+'lf_patch_intrinsic_stanford.hdf5',
  data_path+'lf_patch_intrinsic_700.hdf5',
  data_path+'lf_benchmark_intrinsic.hdf5',
  data_path+'lf_patch_intrinsic_lytro_flowers.hdf5',
  data_path+'lf_patch_intrinsic1_96.hdf5',
  data_path+'lf_patch_intrinsic2_96.hdf5',
  data_path+'lf_patch_intrinsic3_96.hdf5',
  data_path+'lf_patch_intrinsic4_96.hdf5',
  data_path+'lf_patch_intrinsic5_96.hdf5',
  # data_path+'lf_patch_intrinsic_light.hdf5',
]

# NETWORK LAYOUT HYPERPARAMETERS

# general config params
config = {
    # flag whether we want to train for RGB (might require more
    # changes in other files, can't remember right now)
    'rgb'                  : True,

    # maximum layer which will be initialized (deprecated)
    'max_layer'            : 100,

    # this will log every tensor being allocated,
    # very spammy, but useful for debugging
    'log_device_placement' : False,
}


# encoder for 48 x 48 patch, 9 views, RGB
D = 9
H = 96 # 48
W = 96 # 48
if config['rgb']:
    C = 3
else:
    C = 1

# Number of features in the layers
L = 16 # 16
L0 = 24 # 24
L1 = 2*L
L2 = 4*L
L3 = 6*L
L4 = 8*L
L5 = 10*L
L6 = 12*L
L7 = 14*L
L8 = 16*L
L9 = 18*L
L10 = 20*L
L11 = 22*L

import numpy as np

# Encoder stack for downwards convolution
layers = dict()

# TODO: we would like each decoder to be configured individually
# something for later.

# layers[ 'autoencoder' ] = [
#     { 'conv'   : [ 3,3,3, C, L0 ],
#       'stride' : [ 1,1,2, 2, 1 ],
#       'L_middle': 8
#     },
#     { 'conv'   : [ 3,3,3, L0, L1 ],
#       'stride' : [ 1,1, 2,2, 1 ],
#       'L_middle': np.int32(L0 + (L1 - L0)/2)
#     },
#     # resolution now 9 x 24 x 24
#     { 'conv'   : [ 3,3,3, L1, L2 ],
#       'stride' : [ 1,1,2, 2, 1 ],
#       'L_middle': np.int32(L1 + (L2 - L1)/2)
#     },
#     { 'conv'   : [ 3,3,3, L2, L3],
#       'stride' : [ 1,2, 1,1, 1 ],
#       'L_middle': np.int32(L2 + (L3 - L2)/2)
#     },
#     # resolution now 5 x 24 x 24
#     { 'conv'   : [ 3,3,3, L3, L4 ],
#       'stride' : [ 1,1, 2,2, 1 ],
#       'L_middle': np.int32(L3 + (L4 - L3)/2)
#     },
#     { 'conv'   : [ 3,3,3, L4, L5 ],
#       'stride' : [ 1,2,2, 2, 1 ],
#       'L_middle': np.int32(L4 + (L5 - L4)/2)
#     },
#     # resolution now 5 x 12 x 12
# ]

layers[ 'autoencoder' ] = [
    { 'conv'   : [ 3,3,3, C, L0 ],
      'stride' : [ 1,1,2, 2, 1 ],
      'L_middle': 8
    },
    { 'conv'   : [ 3,3,3, L0, L1 ],
      'stride' : [ 1,1, 2,2, 1 ],
      'L_middle': np.int32(L0 + (L1 - L0)/2)
    },
    # resolution now 9 x 24 x 24
    { 'conv'   : [ 3,3,3, L1, L2 ],
      'stride' : [ 1,1,1, 1, 1 ],
      'L_middle': np.int32(L1 + (L2 - L1)/2)
    },
    { 'conv'   : [ 3,3,3, L2, L3],
      'stride' : [ 1,2, 1,1, 1 ],
      'L_middle': np.int32(L2 + (L3 - L2)/2)
    },
    # resolution now 5 x 24 x 24
    { 'conv'   : [ 3,3,3, L3, L4 ],
      'stride' : [ 1,1, 1,1, 1 ],
      'L_middle': np.int32(L3 + (L4 - L3)/2)
    },
    { 'conv'   : [ 3,3,3, L4, L5 ],
      'stride' : [ 1,1,2, 2, 1 ],
      'L_middle': np.int32(L4 + (L5 - L4)/2)
    },
    # resolution now 5 x 12 x 12
    { 'conv'   : [ 3,3,3, L5, L6 ],
      'stride' : [ 1,1, 1,1, 1 ],
      'L_middle': np.int32(L5 + (L6 - L5)/2)
    },
    { 'conv'   : [ 3,3,3, L6, L7 ],
      'stride' : [ 1,1,2, 2, 1 ],
      'L_middle': np.int32(L6 + (L7 - L6)/2)
    },
    # resolution now 5 x 6 x 6
    { 'conv'   : [ 3,3,3, L7, L8 ],
      'stride' : [ 1,1, 1,1, 1 ],
      'L_middle': np.int32(L7 + (L8 - L7)/2)
    },
    { 'conv'   : [ 3,3,3, L8, L9 ],
      'stride' : [ 1,2, 1,1, 1 ],
      'L_middle': np.int32(L8 + (L9 - L8)/2)
    },
    # resolution now 3 x 6 x 6
    {'conv': [3, 3, 3, L9, L10],
     'stride': [1, 1, 1, 1, 1],
     'L_middle': np.int32(L9 + (L10 - L9)/2)
     },
    {'conv': [3, 3, 3, L10, L11],
     'stride': [1, 1, 2, 2, 1],
     'L_middle': np.int32(L10 + (L11 - L10) / 2)
     },
    # resolution now 3 x 3 x 3
]

# chain of dense layers to form small bottleneck (can be empty)
#layers[ 'autoencoder_nodes' ] = [1024, 512]
layers[ 'autoencoder_nodes' ] = []
#layers[ '2D_decoder_nodes' ] = [2500,2000]
# 3D layer chain sucks, try upconvolution for labels as well
layers[ '2D_decoder_nodes' ] = []
layers[ 'preferred_gpu' ] = 0


#
# 3D DECODERS
#
# Generates one default pathway for the EPI stacks
# In the training data, there must be corresponding
# data streams 'decoder_name_v' and 'decoder_name_h',
# which are used for loss computation.
#
decoders_3D = [
    { 'id':      'stacks',
      'channels': C,
      'preferred_gpu' : 0,
      'loss_fn':  'L2',
      'train':    True,
      'weight':   1.0,
    },
]

linked_decoders_3D = [
    { 'id':      'albedo',
      'channels': C,
      'preferred_gpu' : 1,
      'loss_fn':  'L2',
      'train':    True,
      'weight':   1.0,
    },
    {'id': 'sh',
     'channels': C,
     'preferred_gpu': 2,
     'loss_fn': 'L2',
     'train': True,
     'weight': 1.0,
     },
    { 'id':      'specular',
      'channels': C,
      'preferred_gpu' : 3,
      'loss_fn':  'L2',
      'train':    True,
      'weight':   1.0,
    },
]

connect_layers = [ 11,12] #[5,6]
pinhole_connections = True
# pinhole_ratio = 0.25
#
# 2D DECODERS
#
# Each one generates a 2D upsampling pathway next to the
# two normal autoencoder pipes.
#
# Careful, takes memory. Remove some from training if limited.
#
decoders_2D = [
    {'id': 'depth_regression',
     'source': 'depth',
     'channels': 1,
     'preferred_gpu': 0,
     'loss_fn': 'L2',
     'train': True,
     'weight': 1.0,
     },
]

# MINIMIZERS
minimizers = [
    # {'id': 'AE',
    #  'losses_3D': ['stacks'],
    #  'optimizer': 'Adam',
    #  'preferred_gpu': 0,
    #  'step_size': 1e-5,
    #  },
    { 'id' : 'AE',
      'losses_3D' : [ 'stacks', 'intrinsic_sum'],
      'optimizer' : 'Adam',
      'preferred_gpu' : 0,
      'step_size' : 1*1e-5,
    },
    {'id': 'ASS',
     'losses_3D': ['albedo', 'sh', 'specular'],
     'optimizer': 'Adam',
     'preferred_gpu': 2,
     'step_size': 1*1e-5,
     },

    # disparity
    {'id': 'D',
     'losses_2D': ['depth_regression'],
     'optimizer': 'Adam',
     'preferred_gpu': 3,
     'step_size': 1*1e-5,
     },
]
lambda_loss = 0.95

# TRAINING HYPERPARAMETERS
training = dict()

# subsets to split training data into
# by default, only 'training' will be used for training, but the results
# on mini-batches on 'validation' will also be logged to check model performance.
# note, split will be performed based upon a random shuffle with filename hash
# as seed, thus, should be always the same for the same file.
#
training[ 'subsets' ] = {
  'validation'   : 0.05,
  'training'     : 0.95,
}


# number of samples per mini-batch
# reduce if memory is on the low side,
# but if it's too small, training becomes ridicuously slow
training[ 'samples_per_batch' ] = 10

# log interval (every # mini-batches per dataset)
training[ 'log_interval' ] = 5

# save interval (every # iterations over all datasets)
training[ 'save_interval' ] = 50 #15

# noise to be added on each input patch
# (NOT on the decoding result)
training[ 'noise_sigma' ] = 0.0

# decay parameter for batch normalization
# should be larger for larger datasets
# TODO: is there a guideline for this?
training[ 'batch_norm_decay' ]  = 0.9
# flag whether BN center param should be used
training[ 'batch_norm_center' ] = False
# flag whether BN scale param should be used
training[ 'batch_norm_scale' ]  = False
# flag whether BN should be zero debiased (extra param)
training[ 'batch_norm_zero_debias' ]  = False

eval_res = {
    'h_mask': 80,
    'w_mask': 80,
    'm': 4,
    'min_mask': 0.1,
    'result_folder': "./results/",
}