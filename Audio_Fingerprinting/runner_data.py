
import pandas as pd
import tqdm
from Input_processor import*
from fingerprint import*
from pydub import *


def main():
    # 1 Take input and create database
    # 2 Now test using various files present in database (using different offsets)
    # 3 Edit the input and test in different script

    while(True):
        files_dir = input("Enter the address to file directory : ")
        num_files = 30
        input_all, input_names = input_processor(files_dir=files_dir,num_files=num_files)

        if(input_all != None and input_names != None):
            break

    # Processing hashes and making a dataframe for storage
    fingerprints = []
    hashes = []
    song_table = []

    for inputt in tqdm.tqdm(input_all):
        hashh = fingerprint(inputt)
        hashes.append(hashh)

    for i in tqdm.tqdm(range(len(hashes))):
        song_table.append((i, input_names[i]))
        for j in range(len(hashes[i])):
            print((hashes[i][j][0], i, hashes[i][j][1]))
            fingerprints.append((hashes[i][j][0], i, hashes[i][j][1]))

    df_fp = pd.DataFrame(fingerprints, columns=["hash", "song_id", "offset(sec)"])
    df_st = pd.DataFrame(song_table,columns=["song_id", "song_name"])
    df_fp = df_fp.drop_duplicates()
    df_st = df_st.drop_duplicates()
    df_fp.to_csv('./hashes.csv')
    df_st.to_csv('./songs.csv')


if __name__ == "__main__":
    main()