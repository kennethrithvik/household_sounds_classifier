from __future__ import division

import glob
import numpy as np
import pandas as pd
from numpy.random import seed, randint
from scipy.io import wavfile
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import sys

sys.path.append("../")
sys.path.append("./")
# sys.path.append("./blare_base_classifier/keras_vggish")
# sys.path.append("./blare_base_classifier/models")

from vggish import VGGish
from preprocess_sound import preprocess_sound
import vggish_params as params

sound_files = params.SOUND_FILES


# In[]
def loading_data(files_names, labels, sound_extractor):
    files_names = np.array(files_names)
    sample_num = len(files_names)
    seg_len = 5  # 5s
    seg_num = 1
    data = np.empty((1, 96, 64, 1), float)  # np.zeros((seg_num * sample_num, 496, 64, 1))
    label = np.empty((1), int)  # np.zeros((seg_num * sample_num,))

    for i in range(len(files_names)):
        print(i, files_names[i])
        sound_file = sound_files + '/' + files_names[i]
        sr, wav_data = wavfile.read(sound_file)

        # length = sr * seg_len  # 5s segment
        # range_high = len(wav_data) - length
        # if range_high <= 0:
        #     continue
        # seed(1)  # for consistency and replication
        # random_start = randint(range_high, size=seg_num)
        length_seconds = len(wav_data) // sr
        for j in range(length_seconds):
            cur_data = np.zeros((1, 96, 64, 1))
            cur_label = np.zeros((1,))
            cur_wav = wav_data[(j * sr):((j + 1) * sr)]  # wav_data[random_start[j]:random_start[j] + length]
            cur_wav = cur_wav / 32768.0
            cur_spectro = preprocess_sound(cur_wav, sr)
            cur_spectro = np.expand_dims(cur_spectro, 3)
            cur_data = cur_spectro
            cur_label[0] = labels[i]

            data = np.append(data, cur_data, axis=0)
            label = np.append(label, cur_label, axis=0)

    data = sound_extractor.predict(data)

    return data, label


# In[]
if __name__ == '__main__':

    sound_model = VGGish(include_top=True, load_weights=True)

    labels = glob.glob(sound_files + '/*')
    labels = [x.split('/')[-1] for x in labels]
    data_files = []
    for class_name in labels:
        wav_files = glob.glob(sound_files + '/' + class_name + '/*.wav')
        wav_files = [[class_name + '/' + x.split('/')[-1], class_name] for x in wav_files]
        data_files.extend(wav_files)
    data_files = pd.DataFrame(data_files, columns=["file_path", "label"])
    #data_files = data_files[1:400]
    le = LabelEncoder()
    le_fit = le.fit(data_files['label'])
    label_encoded = le_fit.transform(data_files['label'])

    X_train, X_test, y_train, y_test = \
        train_test_split(data_files["file_path"], label_encoded, test_size=0.3, random_state=13)

    # load training data
    print("loading training data...")
    training_data, training_label = loading_data(X_train, y_train, sound_model)
    print(training_data.shape)
    print(training_data)
    # load testing data
    print("loading testing data...")
    testing_data, testing_label = loading_data(X_test, y_test, sound_model)

    clf = svm.SVC(kernel='rbf', gamma=1e-3)
    clf.fit(training_data, training_label.ravel())

    preds = clf.predict(testing_data)
    print(classification_report(le.inverse_transform(testing_label.astype('int')),
                                le.inverse_transform(preds.astype('int'))))
    print(confusion_matrix(le.inverse_transform(testing_label.astype('int')),
                           le.inverse_transform(preds.astype('int'))))

'''
    features, feature_labels = loading_data(data_files["file_path"], label_encoded, sound_model)

    np.save("features", features)
    np.save("labels", feature_labels)
    
'''
