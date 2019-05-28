
from queue import Queue
import time
import numpy as np
import h5py

# plotting
import matplotlib.pyplot as plt

# timing and multithreading
import _thread

# python tools for our lf database
import file_io

# light field GPU tools
import lf_tools

# evaluator thread
from encode_decode_lightfield_v9_anna_interp import encode_decode_lightfield
from encode_decode_lightfield_v9_anna_interp import scale_back
from thread_evaluate_v9 import evaluator_thread

# configuration
import config_autoencoder_v9_current as hp

# Model path setup
model_id = hp.network_model
model_path = './networks/' + model_id + '/model.ckpt'
result_folder =hp.eval_res['result_folder']

# I/O queues for multithreading
inputs = Queue( 15*15 )
outputs = Queue( 15*15 )

data_folders = (
# selected
(  "intrinsic_images", "lytro", "not seen", "lf_test_lytro_koala_lf.mat" ),
(  "intrinsic_images", "lytro", "not seen", "lf_test_lytro_IMG_2693_eslf.png.mat" ),
( "intrinsic_images", "cycles", "not seen", "lf_test_cycles_antonius_lf.mat" ),
( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsicFRMYJ3bYIKVICq" ),
( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic0AruXjjpWdmTOz" ),
( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsiciq16JtRgF7yzKp" ),
( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic100.blend86nN1trYYIMOER" ),
( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic100.blendf3CHROYUXGPVUe" ),
( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic100.blenduBHaulQPh8GbRG" ),
( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic300.blendP2yXl0SXi1unpl" ),
( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic300.blendSopgsvSpyWzK66" ),
( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic400.blendHOcZpNoWCRMw6E" ),

# lytro
# (  "intrinsic_images", "lytro", "not seen", "lf_test_lytro_koala_lf.mat" ),
# (  "intrinsic_images", "lytro", "not seen", "lf_test_lytro_owl_str_lf.mat" ),
# (  "intrinsic_images", "lytro", "not seen", "lf_test_lytro_IMG_2693_eslf.png.mat" ),
# (  "intrinsic_images", "lytro", "not seen", "lf_test_lytro_IMG_3467_eslf.png.mat" ),
# (  "intrinsic_images", "lytro", "seen", "lf_test_lytro_cat3_lf.mat" ),
# (  "intrinsic_images", "lytro", "seen", "lf_test_lytro_hedgehog3_lf.mat.mat" ),

# cycles
# ( "intrinsic_images", "cycles", "not seen", "lf_test_cycles_antonius_lf.mat" ),
# ( "intrinsic_images", "cycles", "not seen", "lf_test_cycles_monkey_lf.mat" ),

# stanford
# ( "intrinsic_images", "stanford", "not seen", "lf_test_stanford_Amethyst_lf.mat" ),

# benchmark
# ( "intrinsic_images", "benchmark", "seen", "lf_benchmark_cotton" ),
# ( "intrinsic_images", "benchmark", "seen", "lf_benchmark_dino" ),
# ( "intrinsic_images", "benchmark", "seen", "lf_benchmark_boxes" ),
# ( "intrinsic_images", "benchmark", "seen", "lf_benchmark_sideboard" ),
# ( "intrinsic_images", "benchmark", "not seen", "lf_benchmark_bedroom" ),
# ( "intrinsic_images", "benchmark", "not seen", "lf_benchmark_bicycle" ),
# ( "intrinsic_images", "benchmark", "not seen", "lf_benchmark_herbs" ),
# ( "intrinsic_images", "benchmark", "not seen", "lf_benchmark_origami" ),

# not seen intrinsic
    # evaluations eusipco2018
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic100.blendKBLL1tSli7mHxh" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic100.blendM9ZZH8rjZns6yQ" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic300.blendaKuyID9x4GTDFR" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsicFRMYJ3bYIKVICq" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsicGumNhefYrATJLh" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsicF9oJj8EUagULX3" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsicDcO5nAshBnldAx" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsiccTRQYxjW6XXw5J" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsicBrZmxtWCIkYTFU" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic6CgoBrTon07emN" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic0AruXjjpWdmTOz" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsiciq16JtRgF7yzKp" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsicz1DefSIynpJhqi" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic100.blend86nN1trYYIMOER" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic100.blendCmtuymbYklRbyL" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic100.blendf3CHROYUXGPVUe" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic100.blendlsr6kp0b3Cta9A" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic100.blenduBHaulQPh8GbRG" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic200.blendSzHXNP9PlVqf8l" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic200.blenduAz8hASsoPtC4D" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic200.blendURNqxu6iw3YrK7" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic200.blendVFuLrwgQ4xsFjd" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic200.blendvxVtzQzVNiL8hE" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic300.blendLIT4JswXiCPiju" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic300.blendOQ15e2k55A6rg1" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic300.blendP2yXl0SXi1unpl" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic300.blendSopgsvSpyWzK66" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic300.blendTNbJETVq6yR8YH" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic400.blendhGXFcYBAyGIeCP" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic400.blendHOcZpNoWCRMw6E" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic400.blendIBOHlC4Guz3A9p" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic400.blendinTo7G8Id3b0oG" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic400.blendKI9ryWVCbSHIBJ" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic500.blendKaiWUfyhlTj5N4" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic500.blendKtuh6fThSZRbfp" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic500.blendM9y0obvNCGGgUn" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic500.blendOVdpSEIXXMI9ZU" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic500.blendpHdF6iapZ36nuL" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic600.blendbbXPSeN89NiZTN" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic600.blendlDPZUFyusbcT4V" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic600.blendLLrATo1v4Xqhol" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic600.blendLWdza5Solz8g9c" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic600.blendMVlmAI59gr5BK3" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic700.blend3rY7JoUS6C6vvF" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic700.blend3wC1ITuAXzpz85" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic700.blend5mgzUR23n5LEnD" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic700.blend6u53KNkepvxGjz" ),
# ( "intrinsic_images", "intrinsic", "not seen", "lf_test_intrinsic700.blend8gXF0AKCnsFX5x" ),

# seen intrinsic
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic100.blend0g6iC59hXrtwmz" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic100.blend1cRRKPIB6BNRlf" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic100.blendZLulPHPb9Wtfia" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic200.blend0c2SBkdE5ZBhSA" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic200.blend1unayNnCqbLegR" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic200.blend3iL3AFE6c2iGCH" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic300.blend4fDF8ZUIxaSrkC" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic300.blend6TFwnh9NZ3o9BP" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic300.blend831c0pm9cWMBnx" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic400.blend5voYa72JGjEP6B" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic400.blend6x5Q53kvPhWXkK" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic400.blend7ahgJfmGU6aw7x" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic500.blend2GiCRTPk4IaRdb" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic500.blend2qY0StvhXOF9Wl" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic500.blend4mgidWW4PeCIZN" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic600.blend6zfwscl9Tm9hWZ" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic600.blend9aA1uoQUuXUZOg" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic600.blend9KdRuCmXqvpMCd" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic700.blendbvO8vxzgfXL47A" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic700.blendCvEwzAZ6g14YSZ" ),
# ( "intrinsic_images", "intrinsic", "seen", "lf_test_intrinsic700.blenddrbmMacPzPvzba" ),

)


# evaluator thread
_thread.start_new_thread( evaluator_thread,
                          ( model_path, hp, inputs,  outputs ))

# wait a bit to not skew timing results with initialization
time.sleep(20)

# loop over all datasets and collect errors
results = []
for lf_name in data_folders:
    file = h5py.File(result_folder + lf_name[3] + '.hdf5', 'w')
    if lf_name[1] == 'intrinsic':
    # stored directly in hdf5
        data_file = "./testData/" + lf_name[0] + "/" + lf_name[1] + "/" +  lf_name[2] + "/"  +  lf_name[3] + ".hdf5"
        hdf_file = h5py.File( data_file, 'r+')
        # hard-coded size, just for testing
        LF = hdf_file[ 'LF' ]
        LF_albedo_gt = hdf_file['LF_albedo']
        LF_sh_gt = hdf_file['LF_sh']
        LF_specular_gt = hdf_file['LF_specular']

        cv_gt = lf_tools.cv( LF )
        albedo_gt = LF_albedo_gt[4,4,:,:,:]
        sh_gt = LF_sh_gt[4, 4, :, :, :]
        specular_gt = LF_specular_gt[4, 4, :, :, :]

        disp_gt = hdf_file['LF_disp']

        dmin = np.min(disp_gt)
        dmax = np.max(disp_gt)
    elif lf_name[1] == 'benchmark':
        data_file = "./testData/" + lf_name[0] + "/" + lf_name[1] + "/" + lf_name[2] + "/"  +  lf_name[3] + ".hdf5"
        hdf_file = h5py.File(data_file, 'r')
        # hard-coded size, just for testing
        LF = hdf_file['LF']
        cv_gt = lf_tools.cv(LF)

        if lf_name[2] == "seen":
            disp_gt = hdf_file['LF_disp']

            dmin = np.min(disp_gt)
            dmax = np.max(disp_gt)
        else:
            dmin = -3.5
            dmax = 3.5
            disp_gt = np.zeros((cv_gt.shape[0], cv_gt.shape[1]), dtype=np.float32)
        albedo_gt = np.zeros((cv_gt.shape[0], cv_gt.shape[1], cv_gt.shape[2]), dtype=np.float32)
        sh_gt = np.zeros((cv_gt.shape[0], cv_gt.shape[1], cv_gt.shape[2]), dtype=np.float32)
        specular_gt = np.zeros((cv_gt.shape[0], cv_gt.shape[1], cv_gt.shape[2]), dtype=np.float32)

    else:
        data_file = "./testData/" + lf_name[0] + "/" + lf_name[1] + "/" + lf_name[2] + "/"  +  lf_name[3] + ".hdf5"
        hdf_file = h5py.File(data_file, 'r+')
        # hard-coded size, just for testing
        LF = hdf_file['LF']
        cv_gt = lf_tools.cv(LF)
        disp_gt = np.zeros((cv_gt.shape[0],cv_gt.shape[1]), dtype = np.float32)
        dmin = -3.5
        dmax = 3.5
        albedo_gt = np.zeros((cv_gt.shape[0], cv_gt.shape[1], cv_gt.shape[2]), dtype=np.float32)
        sh_gt = np.zeros((cv_gt.shape[0], cv_gt.shape[1], cv_gt.shape[2]), dtype=np.float32)
        specular_gt = np.zeros((cv_gt.shape[0], cv_gt.shape[1], cv_gt.shape[2]), dtype=np.float32)

    data = []

    result_cv = encode_decode_lightfield(data, LF,
                                             inputs, outputs,
                                             decoder_path='stacks',
                                             disp=disp_gt)
    mask = result_cv[3]
    cv_out = result_cv[0]
    LF_cv = result_cv[4]

    cv_out = scale_back(cv_out, mask)
    LF_cv = scale_back(LF_cv, mask)

    test_decomposition = True

    if test_decomposition:
        result_albedo = encode_decode_lightfield( data, LF,
                                                   inputs, outputs,
                                                   decoder_path='albedo',
                                                   disp=disp_gt )
        result_sh = encode_decode_lightfield( data, LF,
                                                   inputs, outputs,
                                                   decoder_path='sh',
                                                   disp=disp_gt )
        result_specular = encode_decode_lightfield( data, LF,
                                                    inputs, outputs,
                                                    decoder_path='specular',
                                                    disp=disp_gt )
        mask = result_albedo[3]
        albedo_out = result_albedo[0]
        LF_albedo = result_albedo[4]

        albedo_out = scale_back(albedo_out, mask)
        LF_albedo = scale_back(LF_albedo, mask)

        sh_out = result_sh[0]
        LF_sh = result_sh[4]

        sh_out = scale_back(sh_out, mask)
        LF_sh = scale_back(LF_sh, mask)

        specular_out = result_specular[0]
        LF_specular = result_specular[4]

        specular_out = scale_back(specular_out, mask)
        LF_specular = scale_back(LF_specular, mask)

        cmin =   0.0
        cmax =  1.0
        albedo = np.maximum( cmin, np.minimum( cmax, albedo_out ))
        sh = np.maximum(cmin, np.minimum(cmax, sh_out))
        specular = np.maximum( cmin, np.minimum( cmax, specular_out ))
        both = np.maximum( cmin, np.minimum( cmax, np.multiply(albedo,sh) + specular ))

        result_disp_regression = encode_decode_lightfield(data, LF,
                                                          inputs, outputs,
                                                          decoder_path='depth_regression',
                                                          disp=disp_gt)
        disp_regression_out = result_disp_regression[0]
        disp_regression_out = scale_back(disp_regression_out, mask)

        disp_regression = np.maximum(dmin, np.minimum(dmax, disp_regression_out[:, :, 0]))



        # vertical stack center
        plt.subplot(4, 3, 1)
        plt.imshow( np.clip(albedo,0,1) )
        plt.subplot(4, 3, 2)
        plt.imshow( np.clip(sh, 0,1) )
        plt.subplot(4,3, 3)
        plt.imshow( np.clip(specular, 0,1) )

        plt.subplot(4,3, 4)
        plt.imshow( np.clip(albedo_gt, 0,1) )
        plt.subplot(4, 3, 5)
        plt.imshow( np.clip(sh_gt, 0,1) )
        plt.subplot(4, 3, 6)
        plt.imshow( np.clip(specular_gt, 0,1) )

        plt.subplot(4, 3, 7)
        plt.imshow( np.clip(both,0,1) )
        plt.subplot(4, 3, 8)
        plt.imshow( np.clip(cv_out, 0,1) )
        plt.subplot(4, 3, 9)
        plt.imshow( np.clip(cv_gt, 0,1) )
        plt.subplot(4, 3, 10)
        plt.imshow( disp_gt )
        plt.subplot(4, 3, 11)
        plt.imshow( disp_regression )

        plt.show( block=False )

        dset_LF_out = file.create_dataset('LF_out', data=LF_cv)
        dset_LF_albedo = file.create_dataset('LF_albedo', data=LF_albedo)
        dset_LF_sh = file.create_dataset('LF_sh', data=LF_sh)
        dset_LF_specular = file.create_dataset('LF_specular', data=LF_specular)
        dset_disp = file.create_dataset('disp', data=disp_regression)

inputs.put( () )
#
# #
# code.interact( local=locals() )
