import tensorflow as tf

class_names = ['down' 'go' 'left' 'no' 'right' 'stop' 'up' 'yes']

loaded_model = tf.keras.models.load_model("./cnn_model")