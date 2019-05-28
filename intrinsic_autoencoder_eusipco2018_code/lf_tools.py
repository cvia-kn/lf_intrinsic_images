#
# A bunch of useful helper functions to work with
# the light field data.
#
# (c) Bastian Goldluecke, Uni Konstanz
# bastian.goldluecke@uni.kn
# License: Creative Commons CC BY-SA 4.0
#

import numpy as np
import matplotlib
import matplotlib.pyplot as plt



# returns two epipolar plane image stacks (horizontal/vertical),
# block size (xs,ys), block location (x,y), both in pixels.
def epi_stacks( LF, y, x, ys, xs ):

  T = np.int32( LF.shape[0] )
  cv_v = np.int32( (T - 1) / 2 )
  S = np.int32( LF.shape[1] )
  cv_h = np.int32( (S - 1) / 2 )
  stack_h = LF[ cv_v, :, y:y+ys, x:x+xs, : ]
  stack_v = LF[ :, cv_h, y:y+ys, x:x+xs, : ]
  return (stack_v, stack_h)



# returns center view
def cv( LF ):

  T = np.int32( LF.shape[0] )
  cv_v = np.int32( (T - 1) / 2 )
  S = np.int32( LF.shape[1] )
  cv_h = np.int32( (S - 1) / 2 )
  return LF[ cv_v, cv_h, :,:,: ]


# show an image (with a bunch of checks)
def show( img, cmap='gray' ):

  if len( img.shape ) == 2:
    img = img[ :,:,np.newaxis ]
    
  if img.shape[2]==1:
    img = img[ :,:,0]
    #img = np.clip( img, 0.0, 1.0 )
    imgplot = plt.imshow( img, interpolation='none', cmap=cmap )
  else:
    imgplot = plt.imshow( img, interpolation='none' )

  #code.interact(local=locals())
  plt.show( block=False )


# show a slicing through an array along given dimension indices
def show_slice( array, dims ):

  ext = [ slice(None) ] * len(array)
  for i in range(0,len(dims)):
    ext[ dims[i] ] = slice( 0,-1 )

  np.allclose( array[ext], subset )
  show( subset )



# visualize an element of a batch for training/test
def show_batch( batch, n ):
  ctr = 4

  # vertical stack
  plt.subplot(2, 2, 1)
  plt.imshow( batch[ 'stacks_v' ][ n, :,:, 24,: ] )
  
  # horizontal stack
  plt.subplot(2, 2, 2)
  plt.imshow( batch[ 'stacks_h' ][ n, :,:, 24,: ] )

  # vertical stack center
  plt.subplot(2, 2, 3)
  plt.imshow( batch[ 'stacks_v' ][ n, ctr, :,:,: ] )

  # horizontal stack center
  plt.subplot(2, 2, 4)
  plt.imshow( batch[ 'stacks_h' ][ n, ctr, :,:,: ] )  
    
  plt.show()


def augment_data_intrinsic(input, idx):
  size = input['stacks_v'][idx].shape
  a = np.tile(np.random.randn(1,1,1,3) / 8 + 1, (size[0], size[1], size[2],1))
  b = np.tile(np.random.randn(1,1,1,3) / 8, (size[0], size[1], size[2],1))

  albedo_v = input['albedo_v'][idx]
  albedo_h = input['albedo_h'][idx]
  albedo_v = augment_albedo(a, b, albedo_v)
  albedo_h = augment_albedo(a, b, albedo_h)

  input['albedo_v'][idx] = albedo_v
  input['albedo_h'][idx] = albedo_h

  d = np.tile(np.abs(np.random.randn(1, 1, 1,3) / 8 + 1), (size[0], size[1], size[2],1))
  c = np.abs(np.random.randn(1) / 4 + 1)
  sh_v = input['sh_v'][idx]
  sh_h = input['sh_h'][idx]

  (sh_v,alpha_v) = augment_sh(d, sh_v)
  (sh_h,alpha_h) = augment_sh(d, sh_h)

  input['sh_v'][idx] = sh_v
  input['sh_h'][idx] = sh_h

  specular_v = input['specular_v'][idx]
  specular_h = input['specular_h'][idx]

  specular_v = c * alpha_v * np.multiply(d, specular_v)
  specular_h = c * alpha_h* np.multiply(d, specular_h)

  input['specular_v'][idx] = specular_v
  input['specular_h'][idx] = specular_h

  input['stacks_v'][idx] = np.multiply( input['albedo_v'][idx],input['sh_v'][idx]) + input['specular_v'][idx]
  input['stacks_h'][idx] = np.multiply(input['albedo_h'][idx],input['sh_h'][idx]) + input['specular_h'][idx]

  return(input)

def augment_albedo(a,b,albedo):
  # albedo = np.multiply(pow(-1,sign+1),(sign - albedo))
  out = np.multiply( a, albedo) + b
  out = out - np.minimum(0, np.amin(out))
  out = np.divide(out, np.maximum(np.amax(out), 1))
  return(out)


def augment_sh(d,sh):
  clip_max = 2.0
  sh = np.multiply(d, sh)
  sh = sh - np.minimum(0, np.amin(sh))
  max_v = np.amax(sh)
  if max_v > 2:
   sh_old = sh
   sh = np.multiply(np.divide(sh, max_v), clip_max)
   # find scale constant
   tmp = sh_old
   tmp[sh_old == 0] = 1
   alpha = np.divide(sh, tmp)
   alpha[sh_old == 0] = 1
   alpha[np.isnan(alpha)] = 1
   alpha[np.isinf(alpha)] = 1
   del sh_old
  else:
   alpha = 1
  return (sh,alpha)

def augment_data(input, idx):
  size = input['stacks_v'][idx].shape
  a = np.tile(np.random.randn(1,1,1,3) / 8 + 1, (size[0], size[1], size[2],1))
  b = np.tile(np.random.randn(1,1,1,3) / 8, (size[0], size[1], size[2],1))

  stacks_v = input['stacks_v'][idx]
  stacks_h = input['stacks_h'][idx]
  stacks_v = augment_albedo(a, b, stacks_v)
  stacks_h = augment_albedo(a, b, stacks_h)

  input['stacks_v'][idx] = stacks_v
  input['stacks_h'][idx] = stacks_h

  return(input)