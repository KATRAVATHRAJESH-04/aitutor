
import streamlit as st
import bcrypt
import json
import os
import google.generativeai as genai
from deep_translator import GoogleTranslator
from gtts import gTTS
import speech_recognition as sr
from PIL import Image
import io
import tempfile
from langdetect import detect
import pygame
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from google.generativeai import types
from reportlab.pdfgen import canvas
import re
import time

# ---------------- CONFIG ------------------
genai.configure(api_key="AIzaSyDMvmDTQN7qvTEJ3L1xVFguy7Q_d_C-4nA")  # Replace with your actual API key

vision_model = genai.GenerativeModel("gemini-1.5-flash")
text_model = genai.GenerativeModel("gemini-1.5-flash")

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Marathi": "mr"
}
SUBJECTS = ["Mathematics", "Physics", "Chemistry"]
USER_DATA_FILE = "users.json"

# CSS Styles
css = """
<style>
/* General Styles */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f4f7f6;
    color: #333;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

.st-emotion-cache-z5fcl4 { /* Main content area */
    padding: 2rem;
    flex-grow: 1;
    max-width: 1200px;
    margin: 0 auto;
    width: 100%;
}

h1 {
    color: #2c3e50;
    font-size: 2.5rem;
    margin-bottom: 1rem;
    text-align: center;
}

h2 {
    color: #34495e;
    font-size: 2rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

h3 {
    color: #555;
    font-size: 1.5rem;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
}

subheader { /* Target Streamlit's subheader */
    display: block;
    color: #34495e;
    font-size: 2rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

/* Sidebar Styles */
.st-emotion-cache-164nlkn { /* Target Streamlit's sidebar */
    background-color: #2c3e50;
    color: #fff;
    padding: 2rem;
    width: 300px;
    height: 100vh;
    position: fixed;
    left: 0;
    top: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
}

.st-emotion-cache-164nlkn h1,
.st-emotion-cache-164nlkn h2,
.st-emotion-cache-164nlkn h3,
.st-emotion-cache-164nlkn markdown {
    color: #fff;
}

.st-emotion-cache-164nlkn a {
    color: #81ecec;
    text-decoration: none;
}

.st-emotion-cache-164nlkn a:hover {
    text-decoration: underline;
}

.st-emotion-cache-164nlkn hr {
    border-top: 1px solid #34495e;
    margin: 1.5rem 0;
}

.st-emotion-cache-164nlkn .st-selectbox > label,
.st-emotion-cache-164nlkn .st-radio > label,
.st-emotion-cache-164nlkn .st-toggle > label {
    color: #fff;
    margin-bottom: 0.5rem;
    display: block;
}

.st-emotion-cache-164nlkn .st-selectbox div[role="button"],
.st-emotion-cache-164nlkn .st-radio div[role="radiogroup"] > label div:first-child,
.st-emotion-cache-164nlkn .st-toggle div[role="checkbox"] {
    background-color: #34495e;
    border-radius: 5px;
    padding: 0.5rem 1rem;
    color: #fff;
    border: none;
}

.st-emotion-cache-164nlkn .st-selectbox div[role="button"]:hover,
.st-emotion-cache-164nlkn .st-radio div[role="radiogroup"] > label:hover div:first-child,
.st-emotion-cache-164nlkn .st-toggle div[role="checkbox"]:hover {
    background-color: #4a6572;
}

.st-emotion-cache-164nlkn .st-progress-bar > div > div {
    background-color: #81ecec;
}

.st-emotion-cache-164nlkn .streamlit-expanderHeader {
    color: #fff;
}

.st-emotion-cache-164nlkn .streamlit-expanderContent {
    color: #ddd;
}

/* Main Content Area Styles */
.st-emotion-cache-z5fcl4 .st-subheader {
    color: #34495e;
    border-bottom: 2px solid #ecf0f1;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

.st-emotion-cache-z5fcl4 .st-text-input > label,
.st-emotion-cache-z5fcl4 .st-text-area > label {
    color: #555;
    margin-bottom: 0.5rem;
    display: block;
}

.st-emotion-cache-z5fcl4 .st-text-input > div > input,
.st-emotion-cache-z5fcl4 .st-text-area > div > textarea {
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 0.7rem;
    width: 100%;
    box-sizing: border-box;
    font-size: 1rem;
}

.st-emotion-cache-z5fcl4 .st-radio div[role="radiogroup"] > label div:first-child {
    background-color: #fff;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 0.5rem 1rem;
    color: #333;
    margin-bottom: 0.3rem;
}

.st-emotion-cache-z5fcl4 .st-radio div[role="radiogroup"] > label:hover div:first-child {
    border-color: #3498db;
}

.st-emotion-cache-z5fcl4 .st-button > button {
    background-color: #3498db;
    color: #fff;
    border: none;
    border-radius: 5px;
    padding: 0.7rem 1.5rem;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.3s ease;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

.st-emotion-cache-z5fcl4 .st-button > button:hover {
    background-color: #2980b9;
}

.st-emotion-cache-z5fcl4 .st-success {
    background-color: #d4edda;
    color: #155724;
    padding: 0.8rem;
    border: 1px solid #c3e6cb;
    border-radius: 5px;
    margin-bottom: 1rem;
}

.st-emotion-cache-z5fcl4 .st-warning {
    background-color: #fff3cd;
    color: #85640a;
    padding: 0.8rem;
    border: 1px solid #ffeeba;
    border-radius: 5px;
    margin-bottom: 1rem;
}

.st-emotion-cache-z5fcl4 .st-error {
    background-color: #f8d7da;
    color: #721c24;
    padding: 0.8rem;
    border: 1px solid #f5c6cb;
    border-radius: 5px;
    margin-bottom: 1rem;
}

.st-emotion-cache-z5fcl4 .st-markdown {
    line-height: 1.6;
    margin-bottom: 1rem;
}

.st-emotion-cache-z5fcl4 .st-file-uploader > div > div {
    border: 2px dashed #ddd;
    border-radius: 5px;
    padding: 2rem;
    text-align: center;
    color: #777;
}

.st-emotion-cache-z5fcl4 .st-file-uploader > div > div:hover {
    border-color: #3498db;
    cursor: pointer;
}

.st-emotion-cache-z5fcl4 .streamlit-expander {
    border: 1px solid #ddd;
    border-radius: 5px;
    margin-bottom: 1rem;
}

.st-emotion-cache-z5fcl4 .streamlit-expanderHeader {
    background-color: #f9f9f9;
    padding: 0.8rem 1rem;
    cursor: pointer;
    border-bottom: 1px solid #eee;
    font-weight: bold;
    color: #444;
}

.st-emotion-cache-z5fcl4 .streamlit-expanderContent {
    padding: 1rem;
    color: #666;
}

/* Quiz Styles */
.st-emotion-cache-z5fcl4 .st-subheader:nth-child(1) { /* Style the Quiz subheader */
    color: #e67e22;
    border-bottom-color: #d35400;
}

.st-emotion-cache-z5fcl4 .st-radio div[role="radiogroup"] > label div:first-child {
    background-color: #fff;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 0.7rem 1.2rem;
    color: #333;
    margin-bottom: 0.5rem;
    transition: border-color 0.3s ease, background-color 0.3s ease;
}

.st-emotion-cache-z5fcl4 .st-radio div[role="radiogroup"] > label:hover div:first-child {
    border-color: #e67e22;
    background-color: #fef6e9;
}

.st-emotion-cache-z5fcl4 .st-markdown strong {
    color: #27ae60; /* For "Correct!" */
}

.st-emotion-cache-z5fcl4 .st-markdown em {
    color: #c0392b; /* For "Incorrect." */
    font-style: normal;
}

/* Profile Section Styles (within sidebar) */
.st-emotion-cache-164nlkn .profile-section {
    margin-top: 2rem;
    padding: 1rem;
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 5px;
}

.st-emotion-cache-164nlkn .profile-section h3 {
    color: #fff;
    margin-top: 0;
    margin-bottom: 0.5rem;
}

.st-emotion-cache-164nlkn .profile-section p {
    color: #ddd;
    margin-bottom: 0.3rem;
}

.st-emotion-cache-164nlkn .profile-section .badges {
    margin-top: 0.8rem;
}

.st-emotion-cache-164nlkn .profile-section .badge {
    display: inline-block;
    background-color: #4a6572;
    color: #fff;
    padding: 0.3rem 0.6rem;
    border-radius: 3px;
    font-size: 0.9rem;
    margin-right: 0.3rem;
    margin-bottom: 0.3rem;
}

.st-emotion-cache-164nlkn .profile-section .progress-chart {
    margin-top: 1rem;
}

/* Chat History Styles (within sidebar) */
.st-emotion-cache-164nlkn .chat-history-section {
    margin-top: 2rem;
}

.st-emotion-cache-164nlkn .chat-history-section h3 {
    color: #fff;
    margin-bottom: 0.5rem;
}

.st-emotion-cache-164nlkn .chat-history-item {
    margin-bottom: 0.8rem;
    padding: 0.5rem;
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 5px;
    font-size: 0.95rem;
    color: #eee;
}

.st-emotion-cache-164nlkn .chat-history-item strong {
    color: #81ecec;
}

/* Responsive Design */
@media (max-width: 768px) {
    body {
        flex-direction: column;
    }

    .st-emotion-cache-164nlkn { /* Sidebar */
        position: static;
        width: 100%;
        height: auto;
        margin-bottom: 2rem;
        overflow-y: visible;
    }

    .st-emotion-cache-z5fcl4 { /* Main content */
        padding: 1rem;
    }
}

/* Animations (Optional) */
.st-emotion-cache-z5fcl4 .st-button > button {
    transition: background-color 0.3s ease, transform 0.2s ease-in-out;
}

.st-emotion-cache-z5fcl4 .st-button > button:hover {
    background-color: #2980b9;
    transform: scale(1.05);
}

.st-emotion-cache-z5fcl4 .st-radio div[role="radiogroup"] > label div:first-child {
    transition: border-color 0.3s ease, background-color 0.3s ease, box-shadow 0.2s ease-in-out;
}

.st-emotion-cache-z5fcl4 .st-radio div[role="radiogroup"] > label:hover div:first-child {
    border-color: #e67e22;
    background-color: #fef6e9;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}
</style>
"""

# ------------- Helpers -------------
def load_users():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            return json.load(file)
    return {}

@st.cache_resource
def load_local_model():
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    return tokenizer, model

def generate_quiz_questions(topic, language="English"):
    prompt = (
        f"You are an AI tutor. Generate 5 multiple choice questions (MCQs) on the topic '{topic}'. "
        f"Each question should have 4 options like:\n"
        f"A. Option A\nB. Option B\nC. Option C\nD. Option D\n"
        f"After that, include the correct answer letter and a short explanation.\n"
        f"All questions must follow this format exactly:\n\n"
        f"Q: <question text>\n"
        f"A. <option A>\n"
        f"B. <option B>\n"
        f"C. <option C>\n"
        f"D. <option D>\n"
        f"Answer: <A/B/C/D>\n"
        f"Explanation: <one-line explanation>\n"
    )

    try:
        response = text_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"❌ Gemini API error: {e}")
        return ""
def parse_quiz_text(text):
    quiz = []
    questions = re.split(r'\n(?=Q:)', text)
    for q in questions:
        lines = q.strip().split('\n')
        if len(lines) >= 6:
            question = lines[0][3:].strip()

            options = {}
            for line in lines[1:5]:
                match = re.match(r"([ABCD])[).]?\s*(.+)", line)
                if match:
                    key = match.group(1)
                    value = match.group(2)
                    options[key] = value

            answer_match = re.search(r'Answer:\s*([ABCD])', q)
            explanation_match = re.search(r'Explanation:\s*(.+)', q)

            if answer_match and explanation_match and len(options) == 4:
                quiz.append({
                    'question': question,
                    'options': options,
                    'answer': answer_match.group(1),
                    'explanation': explanation_match.group(1)
                })

    return quiz


def start_quiz_timer(seconds):
    if 'quiz_start_time' not in st.session_state:
        st.session_state['quiz_start_time'] = time.time()
    elapsed_time = int(time.time() - st.session_state['quiz_start_time'])
    remaining_time = max(0, seconds - elapsed_time)
    return remaining_time

def run_quiz():
    st.subheader("🧪 Take a Quiz")
    if 'quiz_questions' not in st.session_state:
        st.session_state['quiz_questions'] = {}
    if 'quiz_answers' not in st.session_state:
        st.session_state['quiz_answers'] = {}
    if 'quiz_score' not in st.session_state:
        st.session_state['quiz_score'] = 0
    if 'question_index' not in st.session_state:
        st.session_state['question_index'] = 0
    if 'quiz_started' not in st.session_state:
        st.session_state['quiz_started'] = False
    if 'quiz_topic' not in st.session_state:
        st.session_state['quiz_topic'] = None
    if 'quiz_timer_duration' not in st.session_state:
        st.session_state['quiz_timer_duration'] = 60  # Default 60 seconds
    if 'auto_submit' not in st.session_state:
        st.session_state['auto_submit'] = False

    st.markdown("### ✏ Enter Your Quiz Topic")
    selected_topic = st.text_input("Topic (e.g., Fractions, Newton's Laws, Chemical Reactions)")

    if selected_topic and not st.session_state['quiz_started']:
        if st.button("Start Quiz"):
            with st.spinner("Generating quiz questions..."):
                quiz_text = generate_quiz_questions(selected_topic, LANGUAGES[st.session_state.get('language_name', 'English')])
                st.session_state['quiz_questions'] = parse_quiz_text(quiz_text)
                st.session_state['quiz_started'] = True
                st.session_state['question_index'] = 0
                st.session_state['quiz_answers'] = {}
                st.session_state['quiz_score'] = 0
                st.session_state['quiz_topic'] = selected_topic
                st.session_state['quiz_start_time'] = time.time()
                st.rerun()

    if st.session_state['quiz_started'] and st.session_state['quiz_questions']:
        if st.session_state['question_index'] < len(st.session_state['quiz_questions']):
            current_question_data = st.session_state['quiz_questions'][st.session_state['question_index']]
            st.subheader(f"Question {st.session_state['question_index'] + 1}: {current_question_data['question']}")

            options = current_question_data['options']  # dict like {'A': '30', 'B': '40' ...}

            # Create choices with full label
            choices = [f"{key}. {value}" for key, value in options.items()]

            # Get previously selected answer if available
            prev_answer = st.session_state['quiz_answers'].get(st.session_state['question_index'])
            if prev_answer:
                try:
                    default_index = choices.index(f"{prev_answer}. {options.get(prev_answer)}")
                except:
                    default_index = 0
            else:
                default_index = 0

            # Display radio with full text
            selected = st.radio("Select your answer", choices, index=default_index,
                                key=f"q_{st.session_state['question_index']}")

            # Extract only the letter 'A', 'B', etc.
            selected_letter = selected.split('.')[0]
            st.session_state['quiz_answers'][st.session_state['question_index']] = selected_letter

            # Navigation buttons
            cols = st.columns(2)
            if cols[0].button("Previous Question", disabled=st.session_state['question_index'] == 0,
                              key=f"prev_q_{st.session_state['question_index']}"):
                st.session_state['question_index'] -= 1
                st.rerun()
            if cols[1].button("Next Question", disabled=st.session_state['question_index'] == len(
                    st.session_state['quiz_questions']) - 1, key=f"next_q_{st.session_state['question_index']}"):
                st.session_state['question_index'] += 1
                st.rerun()

            if st.button("Submit Quiz", key="submit_quiz_btn"):
                st.session_state['question_index'] = len(st.session_state['quiz_questions'])
                st.rerun()

        else:
            st.subheader("Quiz Finished!")
            correct_count = 0
            for i, q_data in enumerate(st.session_state['quiz_questions']):
                user_ans = st.session_state['quiz_answers'].get(i)
                correct_ans = q_data['answer']
                if user_ans == correct_ans:
                    correct_count += 1
                    st.markdown(f"✅ Question {i + 1}: Correct! (Answer: {correct_ans})")
                else:
                    st.markdown(f"❌ Question {i + 1}: Incorrect. Your answer: {user_ans}, Correct answer: {correct_ans}")
                    with st.expander("Explanation"):
                        st.info(q_data['explanation'])

            score = correct_count / len(st.session_state['quiz_questions']) * 100
            st.success(f"Your final score: {score:.2f}% ({correct_count} out of {len(st.session_state['quiz_questions'])}")

            if score == 100:
                st.balloons()
                st.session_state["score"] += 20
                st.session_state["badges"].append(f"🏆 {st.session_state['quiz_topic']} Expert")
            elif score >= 75:
                st.session_state["score"] += 10
                st.session_state["badges"].append(f"🏅 Good in {st.session_state['quiz_topic']}")
            users[st.session_state["username"]]["score"] = st.session_state["score"]
            users[st.session_state["username"]]["badges"] = list(set(st.session_state["badges"])) # Ensure unique badges
            save_users(users)

            if st.button("Restart Quiz"):
                st.session_state['quiz_started'] = False
                st.session_state['quiz_questions'] = {}
                st.session_state['quiz_answers'] = {}
                st.session_state['quiz_score'] = 0
                st.session_state['question_index'] = 0
                st.session_state['quiz_topic'] = None
                st.session_state['quiz_start_time'] = None
                st.session_state['auto_submit'] = False
                st.rerun()

        if st.session_state['quiz_started'] and st.session_state['quiz_questions']:
            remaining_time = start_quiz_timer(st.session_state['quiz_timer_duration'])
            minutes = remaining_time // 60
            seconds = remaining_time % 60
            st.info(f"⏳ Time Left: {minutes:02}:{seconds:02}")
            if remaining_time == 0 or st.session_state.get('auto_submit', False):
                st.warning("⏰ Time's up! Submitting your quiz...")
                st.session_state['question_index'] = len(st.session_state['quiz_questions'])  # Show results
                st.session_state['auto_submit'] = False
                st.rerun()

            if st.button("Submit Quiz", key=f"submit_quiz_btn_{st.session_state['question_index']}"):
                st.session_state['question_index'] = len(st.session_state['quiz_questions'])  # Show results
                st.rerun()


def get_gemini_answer(question, grade, subject):
    prompt = f"You are a helpful tutor for a Grade {grade} student studying {subject}. Answer the following question clearly and in a student-friendly way:\n\n{question}"
    try:
        response = text_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"❌ Gemini API error: {e}")
        return "Sorry, I couldn't get the answer at the moment."

def export_to_pdf(chat_history, username):
    file_path = os.path.join(tempfile.gettempdir(), f"{username}_chat.pdf")
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter
    y = height - 50

    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Chat History - {username}")
    y -= 30

    for q, a in chat_history:
        for line in [f"Q: {q}", f"A: {a}", ""]:
            for subline in line.split("\n"):
                if y < 50:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 12)
                c.drawString(50, y, subline)
                y -= 20
    c.save()
    return file_path

def save_users(users):
    with open(USER_DATA_FILE, "w") as file:
        json.dump(users, file, indent=4)

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def speak(text, lang_code="en"):
    try:
        tts = gTTS(text=text, lang=lang_code)  # Correct way to pass lang
        temp_path = os.path.join(tempfile.gettempdir(), "temp_audio.mp3")
        tts.save(temp_path)
        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
    except Exception as e:
        st.error(f"🔊 TTS Error: {e}")

def stop_speech():
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
    except Exception as e:
        st.warning(f"⏹ Error stopping speech: {e}")

def speech_to_text():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Speak your question (you have 5 seconds)...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5)
            st.success("✅ Audio captured! Transcribing...")
            return recognizer.recognize_google(audio)
    except sr.WaitTimeoutError:
        st.warning("⏳ Timeout: No speech detected.")
    except sr.UnknownValueError:
        st.warning("❌ Sorry, I couldn't understand your voice.")
    except sr.RequestError as e:
        st.error(f"⚠ Google Speech API error: {e}")
    except Exception as e:
        st.error(f"🎤 Microphone error: {e}")
    return ""


def extract_text_from_image(image_file):
    try:
        # Open the uploaded image using PIL
        image = Image.open(image_file).convert("RGB")

        # Directly send the image (PIL.Image.Image object) to Gemini Vision
        response = vision_model.generate_content([
            "Extract the question text from this image.",
            image
        ])

        return response.text.strip()
    except Exception as e:
        st.error(f"🖼 Image processing error: {e}")
        return ""


def get_answer(question, grade, subject, offline_mode=False):
    if offline_mode:
        return local_model_response(question, grade, subject)  # Placeholder
    else:
        return get_gemini_answer(question, grade, subject)

def local_model_response(question, grade, subject):
    tokenizer, model = load_local_model()

    prompt = f"You are a helpful tutor for grade {grade} in {subject}. Answer the question clearly:\n\n{question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the answer part (remove prompt)
    return answer.split("Answer:")[-1].strip()


def translate_text(text, target_lang):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        st.error(f"🌐 Translation error: {e}")
        return text

st.set_page_config(page_title="🔐 Login System", layout="wide")
st.title("🎓📚🧠 VersaLearn 🔍🤖💡")

st.markdown(css, unsafe_allow_html=True)

users = load_users()

if "is_logged_in" not in st.session_state:
    st.session_state["is_logged_in"] = False
    st.session_state["username"] = ""
    st.session_state["chat_history"] = []
    st.session_state["badges"] = []
    st.session_state["score"] = 0
    st.session_state["progress"] = {subject: 0 for subject in SUBJECTS}
    st.session_state["cache"] = {}
    st.session_state["language_name"] = "English" # Default language

# ------------------ AUTH -------------------
if not st.session_state["is_logged_in"]:
    mode = st.sidebar.radio("Select Mode", ["Login", "Register"])

    if mode == "Register":
        st.subheader("📝 Register")
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        if st.button("Register"):
            if username in users:
                st.warning("🚫 Username already exists.")
            elif password != confirm:
                st.warning("🚫 Passwords do not match.")
            elif len(password) < 6:
                st.warning("🔐 Password must be at least 6 characters long.")
            else:
                users[username] = {
                    "name": name,
                    "email": email,
                    "password": hash_password(password),
                    "badges": [],
                    "score": 0,
                    "progress": {subject: 0 for subject in SUBJECTS}
                }
                save_users(users)
                st.success("✅ Registered successfully! Please log in.")
    elif mode == "Login":
        st.subheader("🔓 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = users.get(username)
            if user and check_password(password, user["password"]):
                if "progress" not in user:
                    user["progress"] = {subject: 0 for subject in SUBJECTS}

                st.session_state.update({
                    "is_logged_in": True,
                    "username": username,
                    "chat_history": [],
                    "badges": user.get("badges", []),
                    "score": user.get("score", 0),
                    "progress": user["progress"]
                })

                users[username] = user
                save_users(users)
                st.success(f"✅ Welcome, {user['name']}!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

if st.session_state["is_logged_in"]:
    app_mode = st.sidebar.radio("📚 Choose Mode", ["Tutor", "Quiz"])
    if app_mode == "Tutor":
        st.subheader("🧠 Ask a Question")
        input_mode = st.radio("Choose Input Method", ["Text", "Voice", "Image"], horizontal=True)
        question = ""

        if input_mode == "Text":
            st.session_state["question_input"] = st.text_area("✍ Enter your question here:",value=st.session_state.get("question_input", ""))
        elif input_mode == "Voice":
            if st.button("🎙 Start Recording"):
                with st.spinner("Listening..."):
                    question = speech_to_text()
                    if question:
                        st.session_state["question_input"] = question
                        st.success("✅ Question captured")
                        st.write(question)

        elif input_mode == "Image":
            uploaded_img = st.file_uploader("📷 Upload an image of your question", type=["jpg", "jpeg", "png"])
            if uploaded_img and st.button("📤 Extract Text"):
                with st.spinner("Processing image..."):
                    question = extract_text_from_image(uploaded_img)
                    if question:
                        st.session_state["question_input"] = question
                        st.success("✅ Text extracted from image")
                        st.write(question)

        question = st.session_state.get("question_input", "")

        if question and st.button("💡 Get AI Answer"):
            with st.spinner("Thinking..."):
                # Auto language detect + translate
                try:
                    detected_lang = detect(question)
                    if detected_lang != "en":
                        translated_question = translate_text(question, "en")
                    else:
                        translated_question = question
                except Exception as e:
                    st.warning(f"🌍 Language detection failed: {e}")
                    translated_question = question

                # Check if already answered
                if translated_question in st.session_state["cache"]:
                    ai_response = st.session_state["cache"][translated_question]
                else:
                    ai_response = get_answer(translated_question, st.session_state.get('grade', '6'),
                                             st.session_state.get('subject', 'Mathematics'),
                                             st.session_state.get('offline_mode', False))
                    st.session_state["cache"][translated_question] = ai_response

                # Translate back to user language
                translated_answer = translate_text(ai_response, st.session_state.get('language_code', 'en'))

                # Store chat history
                st.session_state.chat_history.append((question, translated_answer))
                st.session_state["last_answer"] = translated_answer

                # Update score + badge
                st.session_state["score"] += 10
                subject = st.session_state.get('subject', 'Mathematics')
                if subject in st.session_state["progress"]:
                    st.session_state["progress"][subject] += 10
                else:
                    st.session_state["progress"][subject] = 10

                current_user = users[st.session_state["username"]]
                current_user["progress"] = st.session_state["progress"]

                if st.session_state["score"] >= 50 and "🎖 Smart Thinker" not in st.session_state["badges"]:
                    st.session_state["badges"].append("🎖 Smart Thinker")
                users[st.session_state["username"]]["score"] = st.session_state["score"]
                users[st.session_state["username"]]["badges"] = list(set(st.session_state["badges"]))
                users[st.session_state["username"]]["progress"] = st.session_state["progress"]
                save_users(users)

        if "last_answer" in st.session_state:
            st.subheader("📘 AI Tutor's Answer")
            st.markdown(st.session_state["last_answer"])

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔊 Read Aloud"):
                    speak(st.session_state["last_answer"], lang_code=st.session_state.get('language_code', 'en'))
            with col2:
                if st.button("⏹ Stop Speech"):
                    stop_speech()
        if st.button("📄 Export Chat as PDF"):
            pdf_path = export_to_pdf(st.session_state.chat_history, st.session_state["username"])
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="⬇ Download Chat PDF",
                    data=pdf_file,
                    file_name=f"{st.session_state['username']}_chat.pdf",
                    mime="application/pdf"
                )

    elif app_mode == "Quiz":
        run_quiz()

    current_user = users[st.session_state["username"]]
    with st.sidebar:
        st.header("🎓 Student Settings")
        offline_mode = st.sidebar.toggle("📴 Offline Mode")
        st.session_state['offline_mode'] = offline_mode  # Store in session state
        grade = st.selectbox("Grade", [str(i) for i in range(6, 13)], index=int(st.session_state.get('grade', '6')) - 6)
        st.session_state['grade'] = grade
        subject = st.selectbox("Subject", SUBJECTS,
                               index=SUBJECTS.index(st.session_state.get('subject', 'Mathematics')))
        st.session_state['subject'] = subject
        language_name = st.selectbox("Language", list(LANGUAGES.keys()), index=list(LANGUAGES.keys()).index(
            st.session_state.get('language_name', 'English')))
        st.session_state['language_name'] = language_name
        st.session_state['language_code'] = LANGUAGES[language_name]

        st.markdown("---")
        st.markdown("🧍 Profile")
        st.markdown(f"Name: {current_user['name']}")
        st.markdown(f"Email: {current_user['email']}")
        st.markdown(f"Score: {st.session_state['score']}")
        st.markdown(
            f"Badges: {', '.join(st.session_state['badges']) if st.session_state['badges'] else 'No badges yet'}")
        st.markdown("📈 Subject Progress")

        progress = current_user.get("progress", {subject: 0 for subject in SUBJECTS})

        fig, ax = plt.subplots()
        ax.bar(progress.keys(), progress.values(), color="#4CAF50")
        ax.set_ylabel("Points")
        ax.set_title("Your Learning Progress")
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("🕘 Chat History")
        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"Q: {q}")
            st.markdown(f"A: {a}")
            st.markdown("---")

    if st.button("🔓 Logout"):
        st.session_state["is_logged_in"] = False
        st.session_state["username"] = ""
        st.session_state["chat_history"] = []
        st.session_state["badges"] = []
        st.session_state["score"] = 0
        print("hello,git")
        st.rerun()