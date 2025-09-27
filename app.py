import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading
import os

# Check if running on Streamlit Cloud
ON_STREAMLIT_CLOUD = "STREAMLIT_SERVER" in os.environ

# Import voice modules only if running locally
if not ON_STREAMLIT_CLOUD:
    import speech_recognition as sr
    import pyttsx3
    engine = pyttsx3.init()

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load and preprocess chatbot data
with open('chat_data.txt', 'r', encoding='utf8') as file:
    raw_text = file.read().lower()

sent_tokens = nltk.sent_tokenize(raw_text)
word_tokens = nltk.word_tokenize(raw_text)

# Chatbot response function
def chatbot_response(user_input):
    sent_tokens.append(user_input)
    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words='english')
    tfidf = vectorizer.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    sent_tokens.pop()  # remove user input after finding response

    if req_tfidf == 0:
        return "I am sorry, I didn't understand that."
    else:
        return sent_tokens[idx]

# Speech to text function (local only)
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio")
            return ""
        except sr.RequestError:
            st.error("Could not request results from Google Speech Recognition service")
            return ""

# Speak text function (local only)
def speak_text(text):
    if ON_STREAMLIT_CLOUD:
        return  # Skip speaking on cloud
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech).start()

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Streamlit App
st.title("Speech-Enabled Chatbot with Voice Response")

input_mode = st.radio("Choose input mode:", ["Text", "Speech"])

user_input = ""
if input_mode == "Text":
    user_input = st.text_input("Type your message:")
else:
    if ON_STREAMLIT_CLOUD:
        st.warning("Speech input is disabled on Streamlit Cloud. Use Text input.")
    else:
        if st.button("Speak"):
            user_input = speech_to_text()

# Generate response and update history
if user_input:
    response = chatbot_response(user_input.lower())
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))
    
    # Speak the chatbot response
    speak_text(response)

# Display conversation history
st.subheader("Conversation")
for sender, message in st.session_state.history:
    if sender == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")
