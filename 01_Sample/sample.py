import zipfile
import os
import librosa
import pandas as pd
import shutil
import sys

# Mood Miners Emotion Detection Project
# Description: This file contains the code to load the data from the zip file

def process_audio_from_zip(zip_path):
    """
    This function takes in the path to a zip file containing audio files.
    It then processes the audio data using librosa and creates and returns a dataframe.
    """
    # set extract path to name of zip
    print(zip_path)
    extract_path = zip_path.split('.')[0]
    # create the directory if it doesn't exist
    if not os.path.exists(extract_path):
        print(extract_path)
        print('Extracting zip file ' + zip_path.split('/')[1] + ' to ' + zip_path.split('.')[0])
        
        os.makedirs(extract_path)

        # extract the contents of the zip file to the directory
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    # first, simple audio features used by librosa, excluding the raw data
    # features: tempo, spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate
    #           chroma_stft, mfcc, rmse

    # second, inherent emotion features
    # creating the feature dictionary to return later as a df
    feature_dict = {'actor': [], 'tempo': [], 'y':[], 'sr':[], 'onset_env':[], 'spectral_centroid': [], 'spectral_bandwidth':[], 'spectral_rolloff':[], 'zero_crossing_rate':[], 'chroma_stft':[],
                    'mfcc':[], 'rmse':[], 'modality':[], 'vocal_channel':[], 'emotion':[], 'emotional_intensity':[], 'statement':[],'repetition':[]
    }
    # for each file in the directory ill be inserting the data into the feature dictionary
    for actor_dir in os.listdir(extract_path):
        if not actor_dir.startswith('Actor'):
            continue
        print('Processing the actor directory: ' + actor_dir)
        for wav_file in os.listdir(extract_path + '/' + actor_dir):
            if not wav_file.endswith('.wav'):
                continue
            ### Process Librosa Features ###
            # load the audio file
            y, sr = librosa.load(extract_path + '/' + actor_dir + '/' + wav_file)
            # calculate the tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            # calculate the spectral centroid
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            # calculate the spectral bandwidth
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            # calculate the spectral rolloff
            spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            # calculate the zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            # calculate the chroma stft
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            # calculate the mfcc
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            # calculate the rmse
            rmse = librosa.feature.rms(y=y)
            # calculate onset strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            # add all the features to the dictionary
            # appending the features to the dictionary list with the following keys [tempo, spec_cent, spec_bw, spec_rolloff, zcr, chroma_stft, mfcc, rmse]
            feature_dict['actor'].append(actor_dir)
            feature_dict['tempo'].append(tempo)
            feature_dict['spectral_centroid'].append(spec_cent)
            feature_dict['spectral_bandwidth'].append(spec_bw)
            feature_dict['spectral_rolloff'].append(spec_rolloff)
            feature_dict['zero_crossing_rate'].append(zcr)
            feature_dict['chroma_stft'].append(chroma_stft)
            feature_dict['mfcc'].append(mfcc)
            feature_dict['rmse'].append(rmse)
            feature_dict['onset_env'].append(onset_env)
            feature_dict['y'].append(y)
            feature_dict['sr'].append(sr)

            ### Process Inherent Emotion Features ###
            identifiers_only = wav_file.split('.')[0].split('-')
            # Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
            feature_dict['modality'].append(identifiers_only[0])
            # Vocal channel (01 = speech, 02 = song).
            feature_dict['vocal_channel'].append(identifiers_only[1])
            # Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
            feature_dict['emotion'].append(identifiers_only[2])
            # Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
            feature_dict['emotional_intensity'].append(identifiers_only[3])
            # Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
            feature_dict['statement'].append(identifiers_only[4])
            # Repetition (01 = 1st repetition, 02 = 2nd repetition).
            feature_dict['repetition'].append(identifiers_only[5])

        print("Finished processing the actor directory: " + actor_dir)
    print("Finished processing all the audio files in the zip file " + zip_path.split('/')[1])
    
    # deleting the root directory after processing
    print("Deleting the root directory " + extract_path)
    shutil.rmtree(extract_path)
    
    actor_audio_df = pd.DataFrame(feature_dict)
    return actor_audio_df

# Applying the function to all the zip files in the ./emotiona_speech directory
actors_meta_df = pd.DataFrame()
zip_path = 'emotiona_speech'
for actors_zip in os.listdir(zip_path):
    if not actors_zip.endswith('.zip'):
        continue
    print('Processing the zip file: ' + actors_zip.split('.')[0])
    actor_audio_df = process_audio_from_zip(zip_path + '/' + actors_zip)
    if actors_meta_df.empty:
        actors_meta_df = actor_audio_df
    else:
        actors_meta_df = pd.concat([actors_meta_df, actor_audio_df])
print('Finished processing all the zip files in the directory')
actors_meta_df.to_csv('actors_meta_df.csv', index=False)
print('Saved the dataframe to actors_meta_df.csv')
