import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading
import os

# Detect if running on Streamlit Cloud
ON_STREAMLIT_CLOUD = "STREAMLIT_SERVER" in os.environ

# Try importing voice modules; disable if unavailable
VOICE_AVAILABLE = False
if not ON_STREAMLIT_CLOUD:
    try:
        import speech_recognition as sr
        import pyttsx3
        engine = pyttsx3.init()
        VOICE_AVAILABLE = True
    except ImportError:
        VOICE_AVAILABLE = False

# NLTK data directory inside project
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download required NLTK data
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)

# Add to NLTK data path
nltk.data.path.append(nltk_data_dir)

# Load chatbot data
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
    sent_tokens.pop()
    if req_tfidf == 0:
        return "I am sorry, I didn't understand that."
    return sent_tokens[idx]

# Speech-to-text function (local only)
def speech_to_text():
    if not VOICE_AVAILABLE:
        st.warning("Voice input not available.")
        return ""
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
    if not VOICE_AVAILABLE:
        return
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
    if not VOICE_AVAILABLE:
        st.warning("Speech input is disabled in this environment. Use Text input.")
    else:
        if st.button("Speak"):
            user_input = speech_to_text()

# Generate response
if user_input:
    response = chatbot_response(user_input.lower())
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))
    speak_text(response)

# Display conversation history
st.subheader("Conversation")
for sender, message in st.session_state.history:
    if sender == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")
