import os
import io
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
logging.basicConfig(filename='logs/melody.log', level=logging.DEBUG)
def generateWAV(all_predictions, merged):
    logging.basicConfig(filename='logs/ml.log', level=logging.DEBUG)
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
    logging.info(all_predictions)   
    all_txt = ''.join(map(str, all_predictions))
    with open(directory + 'all_predictions'+'.txt', 'w') as file:
        file.write(all_txt)
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
                with open(directory + 'predictions'+ str(export) +'.txt', 'w') as file:
                    file.write(SEMANTIC)
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
                output_file = directory + "full_song" + '.wav'
                full_song = None
                for song in playlist:
                    small_file = directory + 'staff' + str(track) + '.wav'
                    WAV(small_file, 44100, song)
                    track+=1
                    if (full_song) is None:
                        full_song = song
                    else:
                        full_song = np.concatenate((full_song, song))

                WAV(output_file, 44100, full_song)
    except Exception as e:
        logging.error(traceback.format_exc())   
        logging.error("AUDIO: could not generate sinewave audio")











