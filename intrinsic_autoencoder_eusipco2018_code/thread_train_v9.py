#
# Thread for training the autoencoder network.
# Idea is that we can stream data in parallel, which is a bottleneck otherwise
#
import code
import os
import datetime
import sys

import numpy as np

import tensorflow as tf
import libs.tf_tools as tft
import config_data_format as cdf


def trainer_thread( model_path, hp, inputs ):

  # start tensorflow session
  session_config = tf.ConfigProto( allow_soft_placement=True,
                                   log_device_placement=hp.config[ 'log_device_placement' ] )
  sess = tf.InteractiveSession( config=session_config )

  # import network
  from   cnn_autoencoder_v9 import create_cnn
  cnn = create_cnn( hp )

  # add optimizers (will be saved with the network)
  cnn.add_training_ops()
  #cnn.add_additional_training_ops( step_size=1e-4 )
  #cnn.initialize_uninitialized( sess )

  # start session
  print( '  initialising TF session' )
  sess.run(tf.global_variables_initializer())
  print( '  ... done' )

  # save object
  print( '  checking for model ' + model_path )
  if os.path.exists( model_path + 'model.ckpt.index' ):
    print( '  restoring model ' + model_path )
    tft.optimistic_restore( sess,  model_path + 'model.ckpt' )
    print( '  ... done.' )
  else:
    print( '  ... not found.' )
  # redefine layout: add remaining layers not yet in save file
  #cnn.create_variables( layout.config )
  #cnn.create_autoencoder_layers( layout.config )
  # add optimizers (will be saved with the network)
  #cnn.add_training_ops()

  # new saver object with complete network
  saver = tf.train.Saver()


  # statistics
  accuracy_average = 0.0
  difference_average = 0.0
  loss_average = 0.0
  count = 0.0
  print( 'lf cnn trainer waiting for inputs' )

  id_list = []
  for id in cnn.minimizers:
    id_list.append(id)

  terminated = 0;
  while not terminated:

    batch = inputs.get()
    if batch == ():
            terminated = 1
    else:

      niter      = batch[ 'niter' ]
      ep         = batch[ 'epoch' ]
      nsamples   = batch[ 'stacks_v' ].shape[0]
      if 'depth' in batch and not 'depth_one_hot' in batch:
        depth = batch[ 'depth' ]
        depth = np.minimum( cdf.data_config['dmax'], depth )
        depth = np.maximum( cdf.data_config['dmin'], depth )

        ds = depth.shape
        L = cdf.get_labelspace()
        depth_labels  = L.value_to_labels( depth ).reshape( [-1] )
        depth_one_hot = L.labels_to_fractional_one_hot( depth_labels )
        batch[ 'depth_one_hot' ] = depth_one_hot.reshape( [ ds[0], ds[1], ds[2], L._nlabels ] )


      # default params for network input
      net_in = cnn.prepare_net_input( batch )

      # evaluate current network performance on mini-batch
      if batch[ 'logging' ]:

        print()
        sys.stdout.write( '  dataset(%d:%s) ep(%d) batch(%d) : ' %(batch[ 'nfeed' ], batch[ 'feed_id' ], ep, niter) )

        #loss_average = (count * loss_average + loss) / (count + 1.0)
        count = count + 1.0
        fields=[ '%s' %( datetime.datetime.now() ), batch[ 'feed_id' ], batch[ 'nfeed' ], niter, ep ]

        # compute loss for decoder pipelines
        #net_in[ cnn.input_features ] = features
        for id in cnn.decoders_3D:
          if id + '_v' in batch or 'computational' in cnn.decoders_3D[id]:
            ( loss ) = sess.run( cnn.decoders_3D[id]['loss'], feed_dict=net_in )
            sys.stdout.write( '  %s %g   ' %(id, loss) )
            if not 'computational' in cnn.decoders_3D[id]:
              (loss_cv) = sess.run(cnn.decoders_3D[id]['loss_cv'], feed_dict=net_in)
              sys.stdout.write('  %s %g   ' % (id + '_cv', loss_cv))
            fields.append( id )
            fields.append( loss )
          else:
            fields.append( '' )
            fields.append( '' )

        for id in cnn.decoders_2D:
          source = cnn.decoders_2D[id]['source']
          if source in batch:
            ( loss ) = sess.run( cnn.decoders_2D[id]['loss'], feed_dict=net_in )
            sys.stdout.write( '  %s %g   ' %(id, loss) )
            fields.append( id )
            fields.append( loss )
          else:
            fields.append( '' )
            fields.append( '' )

        import csv
        with open( model_path + batch[ 'logfile' ], 'a+') as f:
          writer = csv.writer(f)
          writer.writerow(fields)

        print( '' )
        #code.interact( local=locals() )


      if batch[ 'niter' ] % hp.training[ 'save_interval' ] == 0 and niter != 0 and batch[ 'nfeed' ] == 0 and batch[ 'training' ]:
        # epochs now take too long, save every few 100 steps
        # Save the variables to disk.
        save_path = saver.save(sess, model_path + 'model.ckpt' )
        # save graph
        tf.train.write_graph(sess.graph_def, model_path, 'graph.pb', as_text=False)
        print( 'NEXT EPOCH' )
        print("  model saved in file: %s" % save_path)
        # statistics
        #print("  past epoch average loss %g"%(loss_average))
        count = 0.0


      # run training step
      if batch[ 'training' ]:
        net_in[ cnn.phase ] = True
        #code.interact( local=locals() )
        sys.stdout.write( '.' ) #T%i ' % int(count) )
        # random_val = np.random.rand()
        # if random_val < 0.3:
        #   id = id_list[0]
        # elif random_val < 0.65:
        #   id = id_list[1]
        # else:
        #   id = id_list[2]
        for id in cnn.minimizers:
        # check if all channels required for minimizer are present in batch
          ok = True
          for r in cnn.minimizers[id][ 'requires' ]:
            if not (r in batch):
              ok = False

          if ok:
            sys.stdout.write( cnn.minimizers[id][ 'id' ] + ' ' )
            sess.run( cnn.minimizers[id][ 'train_step' ],
                      feed_dict = net_in )
        sys.stdout.flush()

    inputs.task_done()
