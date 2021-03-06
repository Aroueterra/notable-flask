#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import silence_tensorflow.auto
import time
import traceback
import logging
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from segmenter.slicer import Slice
import config
from apputil import elements
import cv2
import ctc_utils
import os
from pathlib import Path
from config import logger2

tf_v1.compat.v1.disable_eager_execution()
class ML:
    model = ''
    HEIGHT = 128
    def __init__(self, model, vocabulary,input_dir,slice_dir,classification,seq):
        self.model=model
        self.vocabulary=vocabulary
        self.input_dir=input_dir
        self.slice_dir=slice_dir
        self.classification=classification
        self.seq=seq
        
    def setup(self):
        # Read the dictionary
        dict_file = open(self.vocabulary,'r')
        dict_list = dict_file.read().splitlines()
        self.int2word = dict()
        for word in dict_list:
            word_idx = len(self.int2word)
            self.int2word[word_idx] = word
        dict_file.close()
        tf_v1.reset_default_graph()
        self.session = tf_v1.InteractiveSession()
        saver = tf_v1.train.import_meta_graph(self.model)
        saver.restore(self.session,self.model[:-5])
        graph = tf_v1.get_default_graph()
        self.input = graph.get_tensor_by_name("model_input:0")
        self.seq_len = graph.get_tensor_by_name("seq_lengths:0")
        self.rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self.height_tensor = graph.get_tensor_by_name("input_height:0")
        self.width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
        self.logits = graph.get_tensor_by_name("fully_connected/BiasAdd:0")
        self.decoded, _ = tf_v1.nn.ctc_greedy_decoder(self.logits, self.seq_len)
        self.WIDTH_REDUCTION = 16
        return self.session
    
    def predict(self, cv_img):
        start_time = time.time()
        segmented_staves = Slice(cv_img)  
        logger2.info("MODEL: sliced segments: " + str(time.time() - start_time))    
        file_name = segmented_staves[0].name.split('.')[-2]
        file_ext = str(segmented_staves[0]).split('.')[1]
        counter = 1
        all_predictions=[]
        current_file = segmented_staves[0]
        while current_file.exists():
            logger2.info("MODEL: song++ to playlist " + str(time.time() - start_time))    
            file_name = str(current_file).split('.')[-2]
            image = cv2.imread(str(current_file),0)
            image = ctc_utils.resize(image, 128)
            image = ctc_utils.normalize(image)
            image = np.asarray(image).reshape(1,image.shape[0],-1,1)
            seq_lengths = [ image.shape[2] / self.WIDTH_REDUCTION ]
            prediction = self.session.run(self.decoded,
                              feed_dict={
                                  self.input: image,
                                  self.seq_len: seq_lengths,
                                  self.rnn_keep_prob: 1.0,
                              })
            str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
            
            parsed_predictions = ''
            for w in str_predictions[0]:
                parsed_predictions += self.int2word[w] + '\n' 
            if counter < len(segmented_staves):
                current_file = segmented_staves[counter]
            else:
                valid_path, valid_name = os.path.split(segmented_staves[counter-1])
                current_file = Path(valid_path + '\\invalid.png' )
            counter+=1
            all_predictions.append(parsed_predictions)
            logger2.info("MODEL: work completed " + str(time.time() - start_time))    
        return all_predictions
    
        
        
        

