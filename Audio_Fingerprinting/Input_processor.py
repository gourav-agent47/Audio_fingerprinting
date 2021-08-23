import os
import librosa,librosa.display
import numpy as np
import av
import tqdm
import torch as t
from scipy.io import wavfile
from pydub import AudioSegment




def audio_preprocess(x, aug_blend = True):
    # Extra layer in case we want to experiment with different preprocessing
    # For two channel, blend randomly into mono (standard is .5 left, .5 right)

  # x: NTC
    x = x.float()
    if x.shape[-1]==2:
        if aug_blend:
            mix=t.rand((x.shape[0],1), device=x.device) #np.random.rand()
        else:
            mix = 0.5
        x=(mix*x[:,:,0]+(1-mix)*x[:,:,1])
    elif x.shape[-1]==1:
        x=x[:,:,0]
    else:
        assert False, f'Expected channels 1 or 2. Got unknown {x.shape[-1]} channels'

    # x: NT -> NTC
    x = x.unsqueeze(2)
    return x

def get_duration_sec(file, cache=False):
    try:
        with open(file + '.dur', 'r') as f:
            duration = float(f.readline().strip('\n'))
        return duration
    except:
        container = av.open(file)
        audio = container.streams.get(audio=0)[0]
        duration = audio.duration * float(audio.time_base)
        if cache:
            with open(file + '.dur', 'w') as f:
                f.write(str(duration) + '\n')
        return duration

def load_audio(file, sr, offset, duration, resample=True, approx=False, time_base='samples', check_duration=True):
    if time_base == 'sec':
        offset = offset * sr
        duration = duration * sr
    # Loads at target sr, stereo channels, seeks from offset, and stops after duration
    container = av.open(file)
    audio = container.streams.get(audio=0)[0] # Only first audio stream
    audio_duration = audio.duration * float(audio.time_base)
    if approx:
        if offset + duration > audio_duration*sr:
            # Move back one window. Cap at audio_duration
            offset = np.min(audio_duration*sr - duration, offset - duration)
    else:
        if check_duration:
            assert offset + duration <= audio_duration*sr, f'End {offset + duration} beyond duration {audio_duration*sr}'
    if resample:
        resampler = av.AudioResampler(format='fltp',layout='stereo', rate=sr)
    else:
        assert sr == audio.sample_rate
    offset = int(offset / sr / float(audio.time_base)) #int(offset / float(audio.time_base)) # Use units of time_base for seeking
    duration = int(duration) #duration = int(duration * sr) # Use units of time_out ie 1/sr for returning
    sig = np.zeros((2, duration), dtype=np.float32)
    container.seek(offset, stream=audio)
    total_read = 0
    for frame in container.decode(audio=0): # Only first audio stream
        if resample:
            frame.pts = None
            frame = resampler.resample(frame)
        frame = frame.to_ndarray() # Convert to floats and not int16
        read = frame.shape[-1]
        if total_read + read > duration:
            read = duration - total_read
        sig[:, total_read:total_read + read] = frame[:, :read]
        total_read += read
        if total_read == duration:
            break
    assert total_read <= duration, f'Expected {duration} frames, got {total_read}'
    return sig, sr

def input_processor(files_dir = '/content/drive/My Drive/fingerprint_folder/WAV', num_files  = 30):
    print("Processing Input!")
    files = librosa.util.find_files(files_dir, ['mp3'])
    
    
    print(f'Found {len(files)} files!')

    if(len(files) == 0):
        print("Please enter a valid directory!")
        return None,None

    files = files[:num_files]

    input_all = []
    input_names = []

    for file in tqdm.tqdm(files):
       
     
        name = os.path.basename(file) # name of file
        input_names.append(name[:len(name) - 4]) 
        
        
        #mp3 to wav conversion
        
        src = f"{files_dir}\{name}"
        dst = f"waves\{name[:len(name) - 4]}.wav" 
        sound = AudioSegment.from_mp3(src)
        sound.export(dst, format="wav")
        rate, song_array = wavfile.read(dst) # reading wav from waves folder
        song_array = song_array.reshape(1, song_array.shape[0], song_array.shape[1])
        input_ = audio_preprocess(t.from_numpy(song_array))
        input_ = input_.reshape(input_.shape[1])
        input_ = input_.numpy()
        input_all.append(input_)
    return input_all,input_names

def input_single_file(file_path):
    input_sig = []
    input_name = []

    files = librosa.util.find_files(file_path, ['wav'])
    if(len(files) > 1):
        return

    file = files[0]
    # saving the name of the file
    name = os.path.basename(file)
    input_name.append(name[:len(name) - 4])  # subtracting the extension
    # main reading of the wave file
    rate, song_array = wavfile.read(file)  # In future add mp3 to wav converter here maybe (Also look about the tradeoff of conversion)
    # preprocessing

    song_array = song_array.reshape(1, song_array.shape[0], song_array.shape[1])
    # input_ = (song_array[:,:,0] + song_array[:,:,1])/2
    # print(input_.shape)
    # input_ = audio_preprocess(t.from_numpy(song_array))
    input_ = song_array[:, :, 0]
    input_ = input_.reshape(input_.shape[1])
    input_sig.append(input_)

    return input_sig[0],input_name[0]

def find_files(file_dir = './data'):
    files = librosa.util.find_files(file_dir, ['wav'])
    # print(f'Found {len(files)} files!')

    if (len(files) == 0):
        print(f'Please enter a valid directory!')
        return None, None

    return files