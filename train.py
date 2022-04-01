import numpy as np
import os
from tensorflow_examples.lite.model_maker.core import export_format

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

spec = model_spec.get('efficientdet_lite0')

label_map = {
  1: 'fork',
  2: 'knife',
  3: 'mug',
  4: 'scissor',
  5: 'soy_sauce',
  6: 'spoon',
  7: 'vinegar'
}

train_data = object_detector.DataLoader.from_pascal_voc(
  './train', './train', label_map=label_map
)

test_data = object_detector.DataLoader.from_pascal_voc(
  './test', './test', label_map=label_map
)

validation_data = object_detector.DataLoader.from_pascal_voc(
  './validate', './validate', label_map=label_map
) 

model = object_detector.create(train_data, model_spec=spec, batch_size=8, epochs=250, train_whole_model=True, validation_data=validation_data)

model.evaluate(test_data)

model.export(export_dir='.', export_format=[ExportFormat.SAVED_MODEL, ExportFormat.TFLITE, ExportFormat.LABEL])

config = QuantizationConfig.for_float16()

model.export(export_dir='.', tflite_filename='model_fp16.tflite', quantization_config=config)