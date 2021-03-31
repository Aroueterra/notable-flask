import logging
logging.basicConfig(filename='logs/app.log', level=logging.DEBUG)
logger2 = logging.getLogger('app')

voc_file = "vocabulary_semantic.txt"
model = "semantic_model/semantic_model.meta"
input_dir = "input/sheet.png"
slice_dir = "segmenter/output/slice0.PNG"
classification = "raw"
seq = "true"

