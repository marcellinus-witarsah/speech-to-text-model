import wave
import tensorflow as tf

def preprocess_audio_buffer(arr_audio):
    # normalize the array
    waveform = arr_audio / 32768
    # convert to tensor
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
    # convert to spectrogram
    spectrogram = get_spectrogram(waveform)
    # make into form pf batch
    spectrogram = tf.expand_dims(spectrogram, axis=0)
    
    return spectrogram



def get_spectrogram(waveform):
   # zero padding if the waveform is less than 16.000 samples
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32
    )
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram