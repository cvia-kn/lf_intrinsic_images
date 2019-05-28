import os

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

#model_dir = "/data/cnns/trained_models/170529/combined/"
#model_dir = "./networks/combined/"
#model_dir = "./networks/drsoap_v3/"

model_dir = "./networks/current_v8_DS_2/"
checkpoint_path = os.path.join(model_dir, "model.ckpt")

# List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=0 )

# List contents of v0 tensor.
# Example output: tensor_name:  v0 [[[[  9.27958265e-02   7.40226209e-02   4.52989563e-02   3.15700471e-02
#print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='v0')

# List contents of v1 tensor.
#print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='v1')
