#!/usr/bin/python3
import h5py
import code

f = h5py.File( '/data/cnns/training_data/training_140_full_cv.hdf5', 'r' )
d0 = f[ '/data' ]
d1 = f[ '/cv' ]

ap0 = d0[ 0,:,:,0 ]
cv = d1[ :,:,0 ]

code.interact( local=locals() )

