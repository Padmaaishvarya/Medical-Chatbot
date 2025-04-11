from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import re

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "llmapp"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = GoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.4,
    max_tokens=500,
    google_api_key=os.environ.get('GOOGLE_API_KEY')
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    # Get input message from either JSON or form
    if request.is_json:
        msg = request.json.get("message", "").lower()
    else:
        msg = request.form.get("msg", "").lower()

    print("User input:", msg)

    # Logic-based responses for fever chatbot
    if "what is fever" in msg or "define fever" in msg:
        response_text = (
            "Fever is a temporary increase in your body temperature, often due to an illness. "
            "It's a common symptom of infections and is part of the body's immune response."
        )
    elif "i have a fever" in msg or "suffering from fever" in msg or "got fever" in msg:
        response_text = "I'm sorry to hear that. What is your body temperature currently?"

    elif re.search(r"\b(\d{2,3}(\.\d{1,2})?)\b", msg):
        # Extract number from text
        temp_match = re.search(r"\b(\d{2,3}(\.\d{1,2})?)\b", msg)
        temp = float(temp_match.group(1))
        print("Detected temperature:", temp)

        # Basic classification
        if temp < 98.0:
            response_text = "That's below normal."
        elif 98.0 <= temp <= 99.4:
            response_text = "That's within the normal range. Keep monitoring if you feel unwell."
        elif 99.5 <= temp <= 100.4:
            response_text = "That's a mild fever. Stay hydrated and rest."
        elif 100.5 <= temp <= 102.0:
            response_text = "That's a moderate fever. Consider taking medication and monitor your symptoms."
        else:
            response_text = "That's a high fever. Do you also have symptoms like cough, chills, or body aches?"

    elif "yes" in msg:
        response_text = (
            "It sounds like you might have a viral infection. Please stay hydrated, get plenty of rest, and consider seeing a doctor if symptoms persist."
        )
    elif "no" in msg:
        response_text = (
            "Keep monitoring your temperature. If it goes above 100.4°F (38°C), please consult a physician."
        )

    else:
        # Fallback to AI-powered response
        response = rag_chain.invoke({"input": msg})
        response_text = response["answer"]

    print("Response:", response_text)

    # Return appropriate response format
    if request.is_json:
        return jsonify({"reply": response_text})
    else:
        return str(response_text)




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)