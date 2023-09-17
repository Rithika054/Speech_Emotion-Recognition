import streamlit as st
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load the pre-trained Keras model
model = keras.models.load_model('model1.h5')

# Define emotion labels
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Streamlit app
st.title('Emotion Detection App')

# Upload an audio file
audio_file = st.file_uploader('Upload an audio file (wav/mp3)', type=['wav', 'mp3'])

if audio_file is not None:
    # Load the audio file
    audio_data, sample_rate = librosa.load(audio_file, duration=3, offset=0.5)

    # Display waveform and spectrogram
#     st.subheader('Waveform')
#     plt.figure(figsize=(10, 4))
#     plt.title('Waveform', size=20)
#     librosa.display.waveshow(audio_data, sr=sample_rate)
#    # st.pyplot()

#     st.subheader('Spectrogram')
#     plt.figure(figsize=(11, 4))
#     plt.title('Spectrogram', size=20)
    # x = librosa.stft(audio_data)
    # xdb = librosa.amplitude_to_db(abs(x))
    # librosa.display.specshow(xdb, sr=sample_rate, x_axis='time', y_axis='hz')
    # #plt.colorbar()
    #st.pyplot()

    # Extract MFCC features
    mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
    
    # Display MFCC features
    # st.subheader('MFCC Features')
    # st.write(mfcc)

    # Make predictions
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)
    predictions = model.predict(mfcc)

    # Convert predictions to emotion label
    predicted_emotion_index = np.argmax(predictions)
    predicted_emotion = class_labels[predicted_emotion_index]

    st.subheader('Predicted Emotion')
    st.write(predicted_emotion)

    # Play audio associated with predicted emotion (update paths)
    emotion_audio_paths = {
        'angry': r'C:\Users\Student\Downloads\1694906967268zro0n8ro-voicemaker.in-speech.wav',
        'disgust': r'C:\Users\Student\Downloads\169490701390043aw652-voicemaker.in-speech.wav',
        'fear': r'C:\Users\Student\Downloads\1694907047742vf3u8e2g-voicemaker.in-speech.wav',
        'happy': r'C:\Users\Student\Downloads\16949071028834tu79w3u-voicemaker.in-speech.wav',
        'neutral': r'C:\Users\Student\Downloads\1694907087575q4m3js8a-voicemaker.in-speech.wav',
        'sad': r'C:\Users\Student\Downloads\16949071028834tu79w3u-voicemaker.in-speech.wav',
        'surprise': r'C:\Users\Student\Downloads\1694907117641abedhhfz-voicemaker.in-speech.wav',
    }

    emotion_audio_path = emotion_audio_paths.get(predicted_emotion)

    if emotion_audio_path is not None:
        st.audio(emotion_audio_path, format='audio/wav', start_time=0)
    else:
        st.warning("Audio file not found for the predicted emotion.")
