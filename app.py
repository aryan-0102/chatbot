import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data
with open("faq.json", "r") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# Vectorize questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def get_answer(user_query):
    query_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(query_vec, X).flatten()
    idx = similarity.argmax()
    if similarity[idx] < 0.3:   # confidence threshold
        return "Sorry, I don't know the answer to that."
    return answers[idx]

# Streamlit UI
st.set_page_config(page_title="FAQ Chatbot", layout="centered")
st.title("💬 Simple FAQ Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question:")

if st.button("Submit") and user_input:
    answer = get_answer(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", answer))

# Display chat history
for role, text in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**{role}:** {text}")
    else:
        st.markdown(f"_{role}: {text}_")
