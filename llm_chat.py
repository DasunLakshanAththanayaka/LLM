import streamlit as st
from llama_index.llms.groq import Groq
import os

# Use environment variable or Streamlit secrets for API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "your-api-key-here")

llm=Groq(model="llama3-8b-8192",api_key=GROQ_API_KEY)

# Streamlit App Configuration
st.set_page_config(
    page_title="LLM Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# App title
st.title("🤖 LLM Chat Assistant")
st.markdown("Chat with Llama 3 using Groq API")

# Sidebar
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.info("**Model:** Llama 3 8B\n**Provider:** Groq")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask Anything"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Waiting..."):
            try:
                # Use your existing llm object
                response = llm.complete(prompt)
                response_text = str(response)
                
                # Display response
                st.markdown(response_text)
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("*Powered by Llama 3 and Groq API*")