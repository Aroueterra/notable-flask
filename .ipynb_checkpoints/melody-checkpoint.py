import os
import io
from io import BytesIO
import time
import traceback
import logging
import ctc_utils
import numpy as np
import simpleaudio as sa
from pathlib import Path
from midi.player import *
from scipy.io.wavfile import write as WAV
import tensorflow.python.util.deprecation as deprecation
def generate_WAV(all_predictions, merged):
    logging.basicConfig(filename='logs/melody.log', level=logging.DEBUG)
    SEMANTIC = ''
    playlist = []
    track = 0
    export = 0
    single = merged
    directory = 'data\\melody\\'
    del_directory = '\\data\\melody'
    mypath = Path().absolute()
    delete_str = str(mypath) + del_directory
    remove_dir = os.listdir(delete_str)
    for item in remove_dir:
        if (item.endswith(".wav")) or (item.endswith(".txt")):
            os.remove(os.path.join(delete_str, item))
    all_predictions = [x for x in all_predictions if x.strip()]
    logging.info("SYMBOL: printing all predictions")   
    text_files = []
    song_files = []
    fullsong_file = []
    temp_file=BytesIO()
    try:
        for SEMANTIC in all_predictions:
            if SEMANTIC:
                # gets the audio file
                audio = get_sinewave_audio(SEMANTIC)
                # horizontally stacks the freqs    
                audio =  np.hstack(audio)
                # normalizes the freqs
                audio *= 32767 / np.max(np.abs(audio))
                #converts it to 16 bits
                audio = audio.astype(np.int16)
                playlist.append(audio)
                text_files.append(str.encode(SEMANTIC))
                export+=1        
    except Exception as e:
        logging.error(traceback.format_exc())   
        logging.error("AUDIO: could not generate sinewave audio")
    try:
        if(playlist):
            if(single == "true"):
                for song in playlist:
                    output_file = directory + 'staff' + str(track) + '.wav'
                    WAV(output_file, 44100, song)
                    track+=1
            else:
                fullsong_bytes = BytesIO()
                full_song = None
                for song in playlist:
                    song_bytes = BytesIO()
                    WAV(song_bytes, 44100, song)
                    song_files.append(song_bytes)
                    track+=1
                    if (full_song) is None:
                        full_song = song
                    else:
                        full_song = np.concatenate((full_song, song))
                WAV(fullsong_bytes, 44100, full_song)
                fullsong_file.append(fullsong_bytes)
    except Exception as e:
        logging.error(traceback.format_exc())   
        logging.error("AUDIO: could not generate sinewave audio")
    return text_files, fullsong_file, song_files











