import tensorflow as tf
from record_audio import *
from preprocess import *
# import wave

class_names = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

loaded_model = tf.keras.models.load_model("./cnn_model")

# wav_path = "./output_by_marcel.wav"
# with wave.open(wav_path, "rb") as obj:
#     print("Number of Channels: {}".format(obj.getnchannels()))
#     print("Number of Sample width: {}".format(obj.getsampwidth()))
#     print("Number of Frame Rate: {}".format(obj.getframerate()))
#     print("Number of Frames: {}".format(obj.getframerate()))
#     print("Number of Parameters: {}".format(obj.getparams()))
#     print("Audio Time: {} s ".format(obj.getnframes() // obj.getframerate()))

def predict():
    # record the audio using mic
    audio = record_sound()
    terminate()
    # preprocess the audio into spectrogram
    spectrogram = preprocess_audio_buffer(audio)
    # predict the model
    prediction = loaded_model(spectrogram)
    # find the index with the highest value
    label = class_names[np.argmax(prediction[0])]
    # print the label
    # print(np.argmax(prediction, axis=1))
    print("Prediction Result: {}".format(label))
    
if __name__=="__main__":
    predict()
