#
# Push a light field through decoder/encoder modules of the autoencoder
#

from queue import Queue
import code
import numpy as np

import scipy
import scipy.signal

# timing and multithreading
import _thread
import time
from timeit import default_timer as timer

# light field GPU tools
import lf_tools
import libs.tf_tools as tft

# data config
import config_data_format as cdf



def add_result_to_cv( data, result, cv, bs_x, bs_y ):

  """ note: numpy arrays are passed by reference ... I think
  """

  print( 'x', end='', flush=True )
  by = result[1]['py']
  sv = result[0][ 'cv' ]

  # cv data is in the center of the result stack
  # lazy, hardcoded the current fixed size
  for bx in range( sv.shape[0] ):
    px = bs_x * bx + 16
    py = bs_y * by + 16
    if len( sv.shape ) == 4:
      cv[ py:py+16, px:px+16, : ] = sv[ bx, 16:32,16:32,: ]
    else:
      cv[ py:py+16, px:px+16, : ] = sv[ bx, 4, 16:32,16:32,: ]


def encode_decode_lightfield( data, LF, inputs, outputs, decoder_path='stacks', disp=None ):

  # light field size
  H = LF.shape[2]
  W = LF.shape[3]
  dc = cdf.data_config

  # patch step sizes
  bs_y = dc['SY']
  bs_x = dc['SX']
  # patch height/width
  ps_y = dc['H']
  ps_x = dc['W']
  ps_v = dc['D']

  # patches per row/column
  by = np.int16( (H-ps_y) / bs_y ) + 1
  bx = np.int16( (W-ps_x) / bs_x ) + 1

  # one complete row per batch
  cv = np.zeros( [ H, W, 3 ], np.float32 )
  lf_cv = lf_tools.cv( LF )

  print( 'starting LF encoding/decoding [', end='', flush=True )
  start = timer()

  results_received = 0
  for py in range(by):
    print( '.', end='', flush=True )

    stacks_h      = np.zeros( [bx, ps_v, ps_y, ps_x, 3], np.float32 )
    stacks_v      = np.zeros( [bx, ps_v, ps_y, ps_x, 3], np.float32 )
    depth_one_hot = np.zeros( [bx, ps_y, ps_x, dc['L'] ], np.float32 )

    for px in range(bx):
      # get single patch
      patch = cdf.get_patch( LF, cv, disp, py, px )
      #patch = cdf.get_patch( LF, cv, disp, 10, 10 )      

      stacks_v[ px, :,:,:,: ] = patch[ 'stack_v' ]
      stacks_h[ px, :,:,:,: ] = patch[ 'stack_h' ]
      depth_one_hot[ px, :,:,: ] = patch[ 'depth_one_hot' ]
      
    # push complete batch to encoder/decoder pipeline
    batch = dict()
    batch[ 'stacks_h' ] = stacks_h
    batch[ 'stacks_v' ] = stacks_v
    batch[ 'depth_one_hot' ] = depth_one_hot
    batch[ 'depth' ] = np.zeros( [bx, ps_y, ps_x, 1], np.float32 )
    batch[ 'py' ] = py
    batch[ 'decoder_path' ] = decoder_path
    
    inputs.put( batch )

    #
    if not outputs.empty():
      result = outputs.get()
      add_result_to_cv( data, result, cv, bs_x, bs_y )
      results_received += 1
      outputs.task_done()


  # catch remaining results
  while results_received < by:
    result = outputs.get()
    add_result_to_cv( data, result, cv, bs_x, bs_y )
    results_received += 1
    outputs.task_done()


  # elapsed time since start of dmap computation
  end = timer()
  total_time = end - start
  print( '] done, total time %g seconds.' % total_time )
  
        
  # evaluate result
  mse = 0.0
  
  # compute stats and return result
  print( 'total time ', end - start )
  print( 'MSE          : ', mse )
  
  #code.interact( local=locals() )
  return ( cv, total_time, mse )

