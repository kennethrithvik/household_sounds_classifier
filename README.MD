
## Pretrained weights in Keras h5py:

* [Model](https://drive.google.com/open?id=1mhqXZ8CANgHyepum7N4yrjiyIg6qaMe6) with the top fully connected layers

* [Model](https://drive.google.com/open?id=16JrWEedwaZFVZYvn1woPKCuWx85Ghzkp) without the top fully connected layers

--------------------------------------------------------------
- Clone this repo
	

- install dependencies

- Change vggish_params.py::


	SOUND_FILES = '/mnt/409602D09602C67E/Files/Downloads/blind_dl/Dataset'
    # weight path
    WEIGHTS_PATH = 'vggish_audioset_weights_without_fc2.h5'
    WEIGHTS_PATH_TOP = 'vggish_audioset_weights.h5'


- Run it::


	python evaluation.py


That's it!!