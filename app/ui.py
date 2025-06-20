import streamlit as st
import requests
from requests.auth import HTTPBasicAuth

st.set_page_config(page_title="FinSolve Role-Based Chatbot", page_icon="🤖", layout="wide")

# --- Custom CSS for cleaner look ---
st.markdown("""
<style>
.big-title {
    font-size: 2.5rem;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 0.5em;
}
.role-badge {
    display: inline-block;
    background: #e1ecf4;
    color: #0366d6;
    border-radius: 12px;
    padding: 0.2em 0.8em;
    font-size: 1.1em;
    margin-left: 0.5em;
}
.ref-box {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 0.5em 1em;
    margin-bottom: 0.5em;
    border-left: 4px solid #0366d6;
    font-size: 0.98em;
}
.response-box {
    background: #f0f7fa;
    border-radius: 8px;
    padding: 1em;
    margin-bottom: 1em;
    border: 1px solid #b2d6f6;
    font-size: 1.1em;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar Login ---
with st.sidebar:
    st.markdown('<div class="big-title">🤖</div>', unsafe_allow_html=True)
    st.header("Login")
    if "auth" not in st.session_state:
        st.session_state["auth"] = None
    if "role" not in st.session_state:
        st.session_state["role"] = None

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        try:
            resp = requests.get(
                "http://localhost:8000/login",
                auth=HTTPBasicAuth(username, password)
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state["auth"] = (username, password)
                st.session_state["role"] = data.get("role", "")
                st.success(f"Welcome, {data['message']} ")
            else:
                st.error("Login failed. Check your credentials.")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Main Chat Area ---
if st.session_state["auth"]:
    st.markdown('<div class="big-title">FinSolve Role-Based Chatbot</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="role-badge">Role: {st.session_state["role"]}</span>', unsafe_allow_html=True)
    st.subheader("Chat")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    message = st.text_input("Enter your message", key="chat_input")
    if st.button("Send", key="send_btn"):
        with st.spinner("Getting response..."):
            resp = requests.post(
                "http://localhost:8000/chat",
                params={"message": message},
                auth=HTTPBasicAuth(*st.session_state["auth"])
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state["chat_history"].append({
                    "user": message,
                    "bot": data["response"],
                    "refs": data["document_references"]
                })
                # Keep only last 5 conversations
                st.session_state["chat_history"] = st.session_state["chat_history"][-5:]
            else:
                st.error("Error: " + resp.text)
    # Display chat history (last 5)
    for entry in reversed(st.session_state["chat_history"]):
        st.markdown(f'<div class="response-box"><b>You:</b> {entry["user"]}<br><b>Bot:</b> {entry["bot"]}</div>', unsafe_allow_html=True)
        if entry["refs"]:
            st.markdown("<b>References:</b>", unsafe_allow_html=True)
            for ref in entry["refs"]:
                st.markdown(f'<div class="ref-box">📄 <b>{ref["filename"]}</b> (Page {ref.get("page_number", "?")})<br><span style="color:#888;font-size:0.95em">{ref["file_path"]}</span></div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="text-align:center; margin-top: 3em; color: #888;">Please login from the left sidebar to start chatting.</div>', unsafe_allow_html=True)