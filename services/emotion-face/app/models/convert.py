import tensorflow as tf
import tf2onnx
import onnx
from tensorflow import keras
from keras.models import load_model

model = load_model('model.h5')

onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, 'efficientnet_b0_face_v2.onnx')