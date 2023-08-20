import streamlit as st
import numpy as np
import tensorflow as tf
import librosa

st.title("Gender Detection from Audio")
st.write("## Upload any piece of voice and this model will predict its gender")
uploader = st.file_uploader("Upload a voice recording")
model = tf.keras.models.load_model('model/kaggle/working/model')
if uploader is not None:
    with open("audio.mp3","wb") as f:
        f.write(uploader.getvalue())
    wf,sr = librosa.load('audio.mp3')
    mfcc_wf = librosa.feature.mfcc(y=wf, sr=sr)
    b = tf.keras.utils.pad_sequences(mfcc_wf, padding='post', maxlen=200)
    pred = model.predict(np.array([b]))[0]
    if pred <= 0.65:
        st.write(f"## Female with accuracy of {float((1-pred[0])*100).__round__(2)}%")
    else:
        st.write(f"## Male with accuracy of {float((pred[0])*100).__round__(2)}%")


