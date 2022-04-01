from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils

LABEL_PATH = 'labels.txt'
MODEL_PATH = 'yolov4-tiny-416.tflite'
SAVE_TO_PATH = 'model.tflite'

writer = object_detector.MetadataWriter.create_for_inference(
  writer_utils.load_file(MODEL_PATH), input_norm_mean=[0],
  input_norm_std=[255], label_file_paths=[LABEL_PATH]
)

writer_utils.save_file(writer.populate(), SAVE_TO_PATH)