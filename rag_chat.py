import os
import streamlit as st
from llama_index.llms.groq import Groq


#RAG
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Use environment variable or Streamlit secrets for API key
try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "your-api-key-here")
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-api-key-here")

def retrieve_generate(prompt):
    """
    Function to create and return the LLM model response
    """
    llm = Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
    response = llm.complete(prompt)
    return response

def rag(prompt):
    """
    Function to create the RAG (Retrieval-Augmented Generation) system
    
    Args:
        prompt (str): The question/query to ask
    
    Returns:
        response: The RAG response
    """

    # Load documents
    documents = SimpleDirectoryReader("./data").load_data()

    # Initialize the LLM
    llm = Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
    
    # Configure embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Configure settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    
    # Create vector store index
    index = VectorStoreIndex.from_documents(documents)
    
    # Persist the index
    index.storage_context.persist()
    
    # Create query engine
    query_engine = index.as_query_engine()
    response = query_engine.query(prompt)
    
    return response

# Streamlit UI
def main():
    st.set_page_config(
        page_title="RAG Chat Assistant",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 RAG Chat Assistant")
    st.markdown("Ask questions about your documents using AI")

    # Check API key
    if GROQ_API_KEY == "your-api-key-here" or GROQ_API_KEY == "gsk_your_actual_groq_api_key_goes_here":
        st.error("🔑 API Key Required!")
        st.markdown("""
        **To use this app, you need to set your Groq API key:**
        
        **Option 1: Environment Variable**
        ```bash
        $env:GROQ_API_KEY="your-actual-groq-api-key-here"
        ```
        
        **Option 2: Edit the .env file**
        - Open the `.env` file in this directory
        - Replace the placeholder with your actual API key
        
        **Option 3: Streamlit Secrets**
        - Create `.streamlit/secrets.toml` file
        - Add: `GROQ_API_KEY = "your-actual-key"`
        """)
        st.stop()

    # Sidebar for document info
    with st.sidebar:
        st.header("� Document Information")
        
        # Check if data folder exists
        if os.path.exists("./data"):
            files = os.listdir("./data")
            if files:
                st.success(f"✅ Found {len(files)} documents")
                with st.expander("� Document List"):
                    for file in files:
                        st.write(f"• {file}")
            else:
                st.warning("⚠️ Data folder is empty")
                st.info("Add your documents to the './data' folder")
        else:
            st.error("❌ Data folder not found")
            st.info("Create a './data' folder and add your documents")
            st.code("mkdir data")
        
        # Clear chat button
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input (responds to Enter key)
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if data folder exists before processing
        if not os.path.exists("./data") or not os.listdir("./data"):
            st.error("❌ No documents found! Please add documents to the './data' folder first.")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate RAG response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching documents and generating response..."):
                    try:
                        response = rag(prompt)
                        response_text = str(response)
                        
                        # Display response
                        st.markdown(response_text)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.info("💡 Make sure you have valid documents in the './data' folder")
                        # Add error to chat history
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Instructions
    st.markdown("---")
    with st.expander("📖 How to use RAG"):
        st.markdown("""
        **RAG (Retrieval-Augmented Generation):**
        - Upload documents to the `./data` folder
        - Ask questions about your documents
        - The AI will search through your documents and provide relevant answers
        
        **Supported file types:** 
        - `.txt` - Plain text files
        - `.pdf` - PDF documents  
        - `.docx` - Word documents
        - `.md` - Markdown files
        
        **Tips:**
        - Be specific in your questions
        - Ask about content that exists in your documents
        - Try different phrasings if you don't get the answer you expect
        """)

if __name__ == "__main__":
    main()