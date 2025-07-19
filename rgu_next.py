import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load Model
model = tf.keras.models.load_model('gru.h5')

# Load the tokenizer
with open('optimizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

def predict_next_word(model, tokenizer, text, max_seq_len):
    # Convert text to sequence of tokens
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Handle empty token list
    if len(token_list) == 0:
        return None
    
    # Truncate if longer than max_seq_len
    if len(token_list) >= max_seq_len:
        token_list = token_list[-(max_seq_len-1):]
    
    # Pad the sequence to the required length
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    
    # Make prediction
    prediction = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(prediction, axis=1)[0]
    
    # Find the word corresponding to the predicted index
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    
    return None

st.title('Next Word Predictor')
input_text = st.text_input('Enter the text to predict next word')

if st.button('Predict'):
    if input_text.strip():  # Check if input is not empty
        max_seq_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_seq_len)
        
        if next_word:
            st.write(f"Next word is: {next_word}")
        else:
            st.write("Could not predict the next word. Please try with different input.")
    else:
        st.write("Please enter some text to predict the next word.")