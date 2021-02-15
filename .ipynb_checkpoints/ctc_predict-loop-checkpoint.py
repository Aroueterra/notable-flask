from segmenter import Slice
import os
import io
import cv2
import time
import logging
import argparse
import ctc_utils
import numpy as np
from segmenter.slicer import Slice
import silence_tensorflow.auto
import tensorflow as tf
import simpleaudio as sa
from pathlib import Path
from midi.player import *
from audio_to_midi.main import main as MIDI
import tensorflow.compat.v1 as tf_v1
from scipy.io.wavfile import write as WAV
import tensorflow.python.util.deprecation as deprecation
start_time = time.time()
tf_v1.compat.v1.disable_eager_execution()

def Predict():
    parser = argparse.ArgumentParser(description='Decode a music score image with a trained model (CTC).')
    parser.add_argument('-sheet',  dest='sheet', type=str, required=True, help='Path to the whole sheet.')
    parser.add_argument('-image',  dest='image', type=str, required=True, help='Path to the prediction input.')
    parser.add_argument('-model', dest='model', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('-vocabulary', dest='voc_file', type=str, required=True, help='Path to the vocabulary file.')
    parser.add_argument('-type',  dest='type', type=str, nargs='?', help='Path to the output type.')
    parser.add_argument('-seq',  dest='seq', type=str, nargs='?', help='Singles or sequential.')
    args = parser.parse_args()

    '''
    Launch Slice from slicer, perform binarization and divide the sheets

    '''
    slices = Slice(args.sheet)
    print('Slices made? ' + str(slices))

    voc_file = "vocabulary_semantic.txt"
    model = "semantic_model/semantic_model.meta"
    
    tf_v1.reset_default_graph()
    sess = tf_v1.InteractiveSession()

    # Read the dictionary
    dict_file = open(voc_file,'r')
    dict_list = dict_file.read().splitlines()
    int2word = dict()
    for word in dict_list:
        word_idx = len(int2word)
        int2word[word_idx] = word
    dict_file.close()

    # Restore weights
    saver = tf_v1.train.import_meta_graph(args.model)
    saver.restore(sess,args.model[:-5])
    graph = tf_v1.get_default_graph()

    input = graph.get_tensor_by_name("model_input:0")
    seq_len = graph.get_tensor_by_name("seq_lengths:0")
    rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
    height_tensor = graph.get_tensor_by_name("input_height:0")
    width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
    logits = graph.get_tensor_by_name("fully_connected/BiasAdd:0")

    # Constants that are saved inside the model itself
    WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

    decoded, _ = tf_v1.nn.ctc_greedy_decoder(logits, seq_len)

    #TODO: FIX THIS MESS
    path = Path(__file__).parent.absolute()
    sheet_path, sheet_name = os.path.split(args.sheet)
    mypath = Path().absolute()
    file_path = str(mypath) + '\\'
    file_forward = Path(args.image)
    absolute_path = Path(file_path + args.image)
    absolute_str = str(absolute_path)
    file_name = file_forward.name.split('.')[-2]
    file_ext = str(absolute_path).split('.')[1]
    counter = 1
    all_predictions=[]

    print("Input of slices? "+str(absolute_path))
    print("=================================PREDICT=================================")
    while absolute_path.exists():
        print("    ++Adding one more song to playlist " + str(time.time() - start_time))
        file_name = absolute_str.split('.')[-2]
        image = cv2.imread(str(absolute_path),0)
        image = ctc_utils.resize(image, 128)
        image = ctc_utils.normalize(image)
        image = np.asarray(image).reshape(1,image.shape[0],-1,1)
        seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]
        prediction = model.predict(image,seq_lengths)
        str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
        parsed_predictions = ''
        for w in str_predictions[0]:
            parsed_predictions += int2word[w] + '\n' 
        absolute_path = Path(file_name[:-1] + str(counter) + '.' + file_ext)
        counter+=1
        all_predictions.append(parsed_predictions)
    
if __name__ == '__main__':
    SEMANTIC = ''
    playlist = []
    track = 0
    export = 0
    directory=''
    if (args.type == "clean"):
        directory = 'Data\\clean\\'
    elif(args.type == "raw"):
        directory = 'Data\\raw\\'
    else:
        directory = 'Data\\perfect\\'
    all_txt = ''.join(map(str, all_predictions))
    with open(directory + 'all_predictions'+'.txt', 'w') as file:
        file.write(all_txt)       
    for SEMANTIC in all_predictions:
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
        
    if(playlist):
        if(args.seq == "false"):
            for song in playlist:
                output_file = directory + 'staff' + str(track) + '.wav'
                WAV(output_file, 44100, song)
                print("created wav file " + str(time.time() - start_time))
                track+=1
        else:
            output_file = directory + sheet_name[:-4] + '.wav'
            full_song = None
            for song in playlist:
                if (full_song) is None:
                    full_song = song
                else:
                    full_song = np.concatenate((full_song, song))
                    
            WAV(output_file, 44100, full_song)
            #MIDI(output_file)
            print("Generated full song")
            
    print("FULL PROCESS COMPLETED in: " + str(time.time() - start_time) )