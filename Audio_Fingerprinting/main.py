import numpy as np
import pandas as pd
import soundfile
from typing import List, Tuple
from typing import Dict, List, Tuple
from itertools import groupby
from Input_processor import*
from fingerprint import*

sample_rate = 44100

def save_to_wav(fname, aud, sr, j, input=False):
    # clip before saving
    for i in list(range(aud.shape[0])):
        if (input):
            soundfile.write(f'{fname}/item_input_{j}.wav', aud[i], samplerate=sr, format='wav')
        else:
            soundfile.write(f'{fname}/item_{j}.wav', aud[i], samplerate=sr, format='wav')

def output_matched_input(song_id, seconds, offset_seconds, files):

    data, sr = load_audio(files[song_id], sr=44100, offset=offset_seconds, duration=seconds,time_base='sec')
    save_to_wav('./result', data, 44100, song_id)
    return

def display_result(result, seconds, no_of_songs_to_display = 5):

    files = find_files(file_dir='./data')
    for i in range(no_of_songs_to_display):
        name = result[i]['song_name']
        start = result[i]['offset_seconds']
        end = start + seconds
        print(f'song name : {name}, matched from {start} secs to {end} secs')
        output_matched_input(result[i]['song_id'], seconds, start, files)
    return

def test_file(input_signal,df_fp, df_st, seconds = 5):
    # using a random chunk of song for testing even whole song can be tested
    # will try all the options in the future
    inddex = np.random.randint(0, len(input_signal) - seconds * 44100)
    test_input = input_signal[inddex:inddex + seconds * 44100]
    save_to_wav('./result', test_input.reshape(1,len(test_input),1), 44100, 0, input=True)
    # finding hashes
    test_hashes = fingerprint(test_input)
    test_hashes = [test_hashes[i][0:2] for i in range(len(test_hashes))]
    hashes = set()
    hashes |= set(test_hashes)
    test_hashes = hashes

    # finding the matches and
    match_results, dedup_hashes = return_matches(df_fp, test_hashes)
    # print(dedup_hashes)
    result = align_matches(df_fp, df_st, match_results, dedup_hashes, len(test_hashes))

    return display_result(result, seconds, 3)

def main():
    print(f'Welcome to Audio detector!')
    print(f'Now loading the necessary files')
    df_fp = pd.read_csv('./hashes.csv')
    print(f'Retrieving the song list!')
    df_st = pd.read_csv('./songs.csv')
    print('Done!')

    # file_path = input("Enter the address to file directory : ")

    file_path = './test'
    seconds = 24

    input_signal,_ = input_single_file(file_path)

    test_file(input_signal, df_fp, df_st, seconds)

    return 0

if __name__ == '__main__':
    main()
