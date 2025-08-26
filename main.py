from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access variables
openai_key = os.getenv("OPENAI_API_KEY")
# langsmith_key = os.getenv("LANGSMITH_API_KEY")

# print(f"OpenAI Key: {openai_key}")
# print(f"LangSmith Key: {langsmith_key}")

from langchain_openai import ChatOpenAI

# Initialize ChatOpenAI with API key
llm = ChatOpenAI(api_key=openai_key)

# Invoke the model and print the result
response = llm.invoke("Hello, world!")
print(response.content)