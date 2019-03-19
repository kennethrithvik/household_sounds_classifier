from __future__ import division

import glob
import numpy as np
import pandas as pd
from numpy.random import seed, randint
from scipy.io import wavfile
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
import linecache

from keras.models import Model
from keras.layers import GlobalAveragePooling2D

import sys
sys.path.append("../")
sys.path.append("./")
#sys.path.append("./blare_base_classifier/keras_vggish")
#sys.path.append("./blare_base_classifier/models")
from vggish import VGGish
from preprocess_sound import preprocess_sound
import vggish_params as params




sound_files = params.SOUND_FILES

# In[]
def loading_data(files_names, labels, sound_extractor):
    files_names = np.array(files_names)
    sample_num = len(files_names)
    seg_num = 1
    seg_len = 5  # 5s
    data = np.zeros((seg_num * sample_num, 496, 64, 1))
    label = np.zeros((seg_num * sample_num,))

    for i in range(len(files_names)):
        print(i,files_names[i])
        sound_file = sound_files + '/' + files_names[i]
        sr, wav_data = wavfile.read(sound_file)

        length = sr * seg_len  # 5s segment
        range_high = len(wav_data) - length
        if range_high <=0:
            continue
        seed(1)  # for consistency and replication
        random_start = randint(range_high, size=seg_num)

        for j in range(seg_num):
            cur_wav = wav_data[random_start[j]:random_start[j] + length]
            cur_wav = cur_wav / 32768.0
            cur_spectro = preprocess_sound(cur_wav, sr)
            cur_spectro = np.expand_dims(cur_spectro, 3)
            data[i * seg_num + j, :, :, :] = cur_spectro
            label[i * seg_num + j] = labels[i]

    data = sound_extractor.predict(data)

    return data, label


# In[]
if __name__ == '__main__':

    sound_model = VGGish(include_top=False, load_weights=True)

    x = sound_model.get_layer(name="conv4/conv4_2").output
    output_layer = GlobalAveragePooling2D()(x)
    sound_extractor = Model(input=sound_model.input, output=output_layer)

    labels = glob.glob(sound_files + '/*')
    labels = [x.split('/')[-1] for x in labels]
    data_files = []
    for class_name in labels:
        wav_files = glob.glob(sound_files + '/' + class_name + '/*.wav')
        wav_files = [[class_name + '/' + x.split('/')[-1], class_name] for x in wav_files]
        data_files.extend(wav_files)
    data_files = pd.DataFrame(data_files, columns=["file_path", "label"])
    data_files=data_files[1:400]
    le = LabelEncoder()
    le_fit = le.fit(data_files['label'])
    label_encoded = le_fit.transform(data_files['label'])

    X_train, X_test, y_train, y_test = \
        train_test_split(data_files["file_path"], label_encoded, test_size=0.3, random_state=13)

    # load training data
    print("loading training data...")
    training_data, training_label = loading_data(X_train,y_train, sound_extractor)

    # load testing data
    print("loading testing data...")
    testing_data, testing_label = loading_data(X_test,y_test, sound_extractor)

    clf = svm.LinearSVC(C=0.01, dual=False)
    clf.fit(training_data, training_label.ravel())

    preds = clf.predict(testing_data)
    print(classification_report(le.inverse_transform(testing_label.astype('int')),
                                le.inverse_transform(preds.astype('int'))))
    print(confusion_matrix(le.inverse_transform(testing_label.astype('int')),
                                le.inverse_transform(preds.astype('int'))))


