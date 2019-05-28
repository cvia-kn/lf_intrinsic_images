#!/usr/bin/python3
#
# read a bunch of source light fields and write out
# training data for our autoencoder in useful chunks
#
# pre-preparation is necessary as the training data
# will be fed to the trainer in random order, and keeping
# several light fields in memory is impractical.
#
# WARNING: store data on an SSD drive, otherwise randomly
# assembing a bunch of patches for training will
# take ages.
#
# (c) Bastian Goldluecke, Uni Konstanz
# bastian.goldluecke@uni.kn
# License: Creative Commons CC BY-SA 4.0
#

from queue import Queue
import time
import code
import os
import sys
import h5py

import numpy as np


# data config
import config_data_format as cdf

# python tools for our lf database
import file_io
# additional light field tools
import lf_tools
# data config



# OUTPUT CONFIGURATION

# patch size. patches of this size will be extracted and stored
# must remain fixed, hard-coded in NN
px = 48
py = 48

# number of views in H/V/ direction
# input data must match this.
nviews = 9

# block step size. this is only 16, as we keep only the center 16x16 block
# of each decoded patch (reason: reconstruction quality will probably strongly
# degrade towards the boundaries).
# 
# TODO: test whether the block step can be decreased during decoding for speedup.
#
sx = 16
sy = 16

# output file to write to
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!! careful: overwrite mode !!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# previous training data will be erased.

training_data_dir      = '/data/cnns/training_data/'
training_data_filename = 'lf_patch_autoencoder_depth.hdf5'
file = h5py.File( training_data_dir + training_data_filename, 'w' )



# INPUT DATA
#
# configure source datasets
# all must have depth available !
#
data_folders_base =  ( ( "training", "boxes" ),
                       ( "training", "dino" ),
                       ( "training", "cotton" ),
                       ( "training", "sideboard" ),
                       ( "stratified", "pyramids" ),
                       ( "stratified", "dots" ),        
                       ( "stratified", "stripes" ),
                       ( "stratified", "backgammon" )
                     )

data_folders_add =  ( ( "additional", "boardgames" ),
                      ( "additional", "kitchen" ),
                      ( "additional", "medieval2" ),
                      ( "additional", "museum" ),
                      ( "additional", "pens" ),
                      ( "additional", "pillows" ),
                      ( "additional", "platonic" ),
                      ( "additional", "rosemary" ),
                      ( "additional", "table" ),
                      ( "additional", "tomb" ),
                      ( "additional", "town" ),
                      ( "additional", "vinyl" )
                     )


#
#data_folders = ( ( "training", "boxes" ), )
data_folders = data_folders_base + data_folders_add



# EPI patches, nviews x patch size x patch size x channels
# horizontal and vertical direction (to get crosshair)
dset_v = file.create_dataset( 'stacks_v', ( 9, py,px, 3, 1 ),
                              chunks = ( 9, py,px, 3, 1 ),
                              maxshape = ( 9, py,px, 3, None ) )

dset_h = file.create_dataset( 'stacks_h', ( 9, py,px, 3, 1 ),
                              chunks = ( 9, py,px, 3, 1 ),
                              maxshape = ( 9, py,px, 3, None ) )

# dataset for corresponding depth patches
dset_depth = file.create_dataset( 'depth', ( 48,48, 1 ),
                                  chunks = ( 48,48, 1 ),
                                  maxshape = ( 48,48, None ) )

# dataset for correcponsing center view patch (to train joint upsampling)
# ideally, would want to reconstruct full 4D LF patch, but probably too memory-intensive
# keep for future work
dset_cv = file.create_dataset( 'cv', ( 48,48, 3, 1 ),
                               chunks = ( 48,48, 3, 1 ),
                               maxshape = ( 48,48, 3, None ) )


#
# loop over all datasets, write out each dataset in patches
# to feed to autoencoder in random order
#
index = 0
for lf_name in data_folders:

  data_folder = "/data/lfa/" + lf_name[0] + "/" + lf_name[1] + "/"
  LF = file_io.read_lightfield( data_folder )
  LF = LF.astype( np.float32 ) / 255.0

  disp = file_io.read_disparity( data_folder )
  disp_gt = np.array( disp[0] )
  disp_gt = np.flip( disp_gt,0 )

  cv = lf_tools.cv( LF )

  # maybe we need those, probably not.
  param_dict = file_io.read_parameters(data_folder)

  # write out one individual light field
  # block count
  cx = np.int32( ( LF.shape[3] - px) / sx ) + 1
  cy = np.int32( ( LF.shape[2] - py) / sy ) + 1

  for by in np.arange( 0, cy ):
    sys.stdout.write( '.' )
    sys.stdout.flush()

    for bx in np.arange( 0, cx ):

      # extract data
      patch = cdf.get_patch( LF, cv, disp_gt, by, bx )

      # write to respective HDF5 datasets
      dset_v.resize( index+1, 4 )
      dset_v[ :,:,:,:, index ] = patch[ 'stack_v' ]

      dset_h.resize( index+1, 4 )
      dset_h[ :,:,:,:, index ] = patch[ 'stack_h' ]

      dset_depth.resize( index+1, 2 )
      dset_depth[ :,:, index ] = patch[ 'depth' ]

      dset_cv.resize( index+1, 3 )
      dset_cv[ :,:,:, index ] = patch[ 'cv' ]

      # next patch
      index = index + 1

  # next dataset
  print(' done.')

