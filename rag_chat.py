import os
import streamlit as st
from llama_index.llms.groq import Groq


#RAG
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings


# Use environment variable or Streamlit secrets for API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "your-api-key-here")

llm=Groq(model="llama3-8b-8192",api_key=GROQ_API_KEY)