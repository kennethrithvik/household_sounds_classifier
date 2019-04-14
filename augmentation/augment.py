import argparse
parser = argparse.ArgumentParser()
parser.add_argument("path", help="Path where your file is")
parser.add_argument("folder", help="Folder name")
args = parser.parse_args()

path = args.path
folder = args.folder

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import random
import itertools
import librosa
import os
import glob
from itertools import combinations 
import random
from pydub import AudioSegment




def load_audio_file(file_path):
    input_length = 100000
    data = librosa.core.load(file_path)[0] #, sr=16000
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data



def create_folder(path_):
    try:
        os.mkdir(path_)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)
        
        
        
def plot_time_series(data):
    fig = plt.figure(figsize=(14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.show()
    
# stretching the sound
def stretch(data, rate=1):
    input_length = 100000
    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms



def remove_silence(sound):
    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())
    duration = len(sound)    
    trimmed_sound = sound[start_trim:duration-end_trim]

    return(trimmed_sound)


def main():
    
    #make 4 folders
    #white_path = path+folder+'_wnoise'
    shift_path = path+folder+'_shift'
    stretch_low_path = path+folder+'_stretch_low'
    stretch_high_path = path+folder+'_stretch_high'
    mix_path = path+folder+'_mix'
    path_wav = path+folder +'/*.wav'
    #create_folder(white_path)
    print(shift_path)
    create_folder(shift_path)
    create_folder(stretch_low_path)
    create_folder(mix_path)
    create_folder(stretch_high_path)



    all_files = glob.glob(path_wav)
    for filename in all_files:
        #print(filename)
        file_name=os.path.basename(filename)
        data = load_audio_file(filename)

        # Adding white noise 

        #wn = np.random.randn(len(data))
        #data_wn = data + 0.005*wn
        #plot_time_series(data_wn)
        # We limited the amplitude of the noise so we can still hear the sound even with the noise, 
        #which is the objective
        #ipd.Audio(data_wn, rate=16000)
        #file_path  = white_path + '/' +file_name.replace('.wav','_wn.wav')
        #print(file_path)
        #librosa.output.write_wav(file_path, data_wn, 16000)

        # Shifting sound

        data_roll = np.roll(data, 1600)
        #plot_time_series(data_roll)
        #ipd.Audio(data_roll, rate=16000)
        file_path  = shift_path + '/' +file_name.replace('.wav','_shift.wav')
        #print(file_path)
        librosa.output.write_wav(file_path, data_roll, 16000)

        # Streatching 

        data_stretch_low =stretch(data, 0.8)
        #print("This makes the sound deeper but we can still hear 'off' ")
        #plot_time_series(data_stretch)
        #ipd.Audio(data_stretch, rate=16000)
        file_path  = stretch_low_path + '/' +file_name.replace('.wav','_stretch_low.wav')
        librosa.output.write_wav(file_path, data_stretch_low, 16000)

        data_stretch_high =stretch(data, 1.2)
        #print("Higher frequencies  ")
        #plot_time_series(data_stretch)
        #ipd.Audio(data_stretch, rate=16000)
        file_path  = stretch_high_path + '/' +file_name.replace('.wav','_stretch_high.wav')
        librosa.output.write_wav(file_path, data_stretch_high, 16000)
            
            
        # Mix two sounds 


    comb = list(itertools.permutations(all_files, 2))
    for a,b in comb:

        sound1 = AudioSegment.from_file(a, format="wav")
        sound2 = AudioSegment.from_file(b, format="wav")

        sound1 = remove_silence(sound1)
        sound2 = remove_silence(sound2)

        played_togther = sound2.overlay(sound1)

        file_name_1=os.path.basename(a)
        file_name_2=os.path.basename(b)
        file_path  = mix_path + '/' +file_name_2+'_'+file_name_1.replace('.wav','mix.wav')
        played_togther.export(file_path, format="wav")


if __name__ == "__main__":
    main()