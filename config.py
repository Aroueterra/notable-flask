import logging
logging.basicConfig(filename='logs/ml.log', level=logging.DEBUG)
logger2 = logging.getLogger('ML_MODEL')

voc_file = "vocabulary_semantic.txt"
model = "semantic_model/semantic_model.meta"
input_dir = "input/sheet.png"
slice_dir = "segmenter/output/slice0.PNG"
classification = "raw"
seq = "true"
