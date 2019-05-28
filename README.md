# Intrinsic Light Field Decomposition and Disparity Estimation with Deep Encoder-Decoder Network

We present an encoder-decoder deep neural network
that solves non-Lambertian intrinsic light field decomposition,
where we recover all three intrinsic components: albedo, shading,
and specularity. We learn a sparse set of features from 3D
epipolar volumes and use them in separate decoder pathways to
reconstruct intrinsic light fields. While being trained on synthetic
data generated with Blender, our model still generalizes to real
world examples captured with a Lytro Illum plenoptic camera.
The proposed method outperforms state-of-the-art approaches
for single images and achieves competitive accuracy with recent
modeling methods for light fields.
![teaser_git](https://user-images.githubusercontent.com/41570345/58459365-836c4b00-812b-11e9-82f5-3341725a9229.png)

## Project description
Our project consist of 2 steps:

1. Divide input light fields into 3D patches and create network inputs with [DataRead_code](https://github.com/cvia-kn/lf_autoencoder_cvpr2018_code/tree/master/DataRead_code) project
2. Train and evaluate the network with [lf_autoencoder_cvpr2018_code](https://github.com/cvia-kn/lf_autoencoder_cvpr2018_code/tree/master/lf_autoencoder_cvpr2018_code) project

### Prerequisites
1. Python 3.5
2. Tensorflow with GPU support

### 1. Creating the data
Depends on the type of data use separate scripts to create inputs (.hdf5 data container) for the network: 
* synthetic **create_training_data_intrinsic.py**
* real-world use separete script **create_training_data_lytro_intrinsic.py**
```
px = 96 # patch size
py = 96 
nviews = 9 # number of views
sx = 32 # block step size
sy = 32

training_data_dir = "./trainData/"
training_data_filename = 'lf_patch_autoencoder1.hdf5'
file = h5py.File( training_data_dir + training_data_filename, 'w' )

data_source = "./CNN_data/1"
```
Synthetic data that can be used for training, for more training data please contact our research group:
* [Container 1](http://data.lightfield-analysis.net/CNN_data/1.zip)
* [Container 2](http://data.lightfield-analysis.net/CNN_data/2.zip)
* [Container 3](http://data.lightfield-analysis.net/CNN_data/3.zip)
* [Container 4](http://data.lightfield-analysis.net/CNN_data/4.zip)
* [Container 5](http://data.lightfield-analysis.net/CNN_data/5.zip)
* [Container 6](http://data.lightfield-analysis.net/CNN_data/diffuse.zip) Here we create Lambertian light fields where we assign 0 to specular component.
* [Test data](http://data.lightfield-analysis.net/CNN_data/test_intrinsic.zip)

### 2. Run the network
To train the network you need to specify all training options in the **config_autoencoder_v9_final.py**
Also, you need to specify patch size and minimum and maximum disparity values in the **config_data_format.py**
In **cnn_autoencoder.py** you need to specify coordinates that are taken into account when the loss is computed.
For example, if the input patch size is 48x48, and we select *loss_min_coord_3D= 0*, *loss_max_coord_3D = 40*,
then the last 8 pixels will be omitted while computing loss.

To use the trained model, please download the model [current_v9.zip](http://data.lightfield-analysis.net/current_v9.zip) and extract the archive to **./networks/** folder.
We provide some test examples [diffuse_specular.zip](http://data.lightfield-analysis.net/diffuse_specular.zip) that shoulb be extracted to the **./test_data/** folder.

To evaluate on all test examples used in the paper, create the test data with the [DataRead_code](https://github.com/cvia-kn/lf_autoencoder_cvpr2018_code/tree/master/DataRead_code) project.

### References
* [Our webpage](https://www.cvia.uni-konstanz.de/)
* [Paper](http://publications.lightfield-analysis.net/AJSG18_cvpr.pdf)
* [Supplementary material](http://publications.lightfield-analysis.net/AJSG18_cvpr_supplemental.pdf)

