from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import cassio
import pdfplumber
from PIL import Image
from io import BytesIO
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from googleapiclient.discovery import build
from duckduckgo_search import DDGS
import tempfile
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import base64
import requests

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

# API keys and configuration
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
HF_TOKEN = os.getenv("HF_API_TOKEN")  # Get from https://huggingface.co/settings/tokens
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Initialize Hugging Face Inference Client
hf_client = InferenceClient(token=HF_TOKEN)

# Initialize Cassandra
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Global variables
vector_store = None
chat_history = []
current_image = None

# Text splitter configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# Initialize tools
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())

def get_blog_articles(query):
    results = DDGS().text(f"{query} site:medium.com OR site:dev.to OR site:towardsdatascience.com", max_results=5)
    return "\n".join([f"{result['title']} - {result['href']}" for result in results])

def get_youtube_videos(query):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(q=query, part="snippet", maxResults=3, type="video")
    response = request.execute()
    return "\n".join([f"🎥 {video['snippet']['title']} - https://www.youtube.com/watch?v={video['id']['videoId']}" for video in response["items"]])

tools = [
    Tool(name="Wikipedia", func=wikipedia.run, description="Search Wikipedia for information"),
    Tool(name="ArXiv", func=arxiv.run, description="Retrieve academic papers from ArXiv"),
    Tool(name="Blogs", func=get_blog_articles, description="Find blog articles about technical topics"),
    Tool(name="YouTube", func=get_youtube_videos, description="Find educational YouTube videos"),
]

agent_executor = initialize_agent(
    tools=tools,
    llm=None,  # We'll handle text queries directly
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Helper Functions
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def process_image(image_file):
    """Use HF Inference API for image captioning"""
    global current_image
    try:
        img = Image.open(image_file).convert("RGB")
        current_image = img
        
        # Convert to base64 for HF API
        img_base64 = image_to_base64(img)
        
        # Use BLIP-2 model via API
        response = hf_client.image_to_text(
            image=img_base64,
            model="Salesforce/blip-image-captioning-large"
        )
        return response
    except Exception as e:
        return f"Error processing image: {str(e)}"

def extract_text_from_pdf(pdf_file):
    raw_text = ""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        pdf_file.save(temp_file.name)
        with pdfplumber.open(temp_file.name) as pdf:
            for page in pdf.pages:
                raw_text += page.extract_text() + "\n"
        os.unlink(temp_file.name)
    return raw_text if raw_text.strip() else "Error: Could not extract text from PDF."

def chunk_text(raw_text):
    return text_splitter.split_text(raw_text)

def initialize_vector_store(text_chunks):
    global vector_store
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Cassandra(
        embedding=embedding,
        table_name="qa_demo",
        session=None,
        keyspace=None
    )
    vector_store.clear()
    vector_store.add_texts(text_chunks)
    return VectorStoreIndexWrapper(vectorstore=vector_store)

def query_with_learning_resources(vector_store_wrapper, user_query):
    global chat_history
    retrieved_docs = vector_store_wrapper.vectorstore.similarity_search(user_query, k=3)
    combined_context = " ".join([doc.page_content for doc in retrieved_docs])
    
    if retrieved_docs:
        # Use HF Inference API for text generation
        response = hf_client.text_generation(
            prompt=f"Answer based on context:\n{combined_context}\n\nQuestion: {user_query}\nAnswer:",
            model="mistralai/Mistral-7B-Instruct-v0.1",
            max_new_tokens=200
        )
        response_text = response
    else:
        response_text = agent_executor.run(user_query)
    
    chat_history.extend([f"User: {user_query}", f"AI: {response_text}"])
    return response_text

# Routes
@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    global chat_history, vector_store, current_image
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    chat_history = []
    current_image = None
    
    if file.filename.lower().endswith('.pdf'):
        raw_text = extract_text_from_pdf(file)
        if raw_text.startswith("Error:"):
            return jsonify({'error': raw_text}), 400
        text_chunks = chunk_text(raw_text)
        initialize_vector_store(text_chunks)
        return jsonify({'message': 'PDF uploaded successfully'}), 200
    
    elif file.filename.lower().split('.')[-1] in ['jpg', 'jpeg', 'png', 'gif']:
        image_description = process_image(file)
        if image_description.startswith("Error"):
            return jsonify({'error': image_description}), 400
        
        text_chunks = chunk_text(image_description)
        initialize_vector_store(text_chunks)
        return jsonify({
            'message': 'Image uploaded successfully',
            'description': image_description
        }), 200
    
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

@app.route('/query', methods=['POST'])
@cross_origin()
def query():
    global vector_store, current_image
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    user_query = data['query']
    
    if current_image:
        try:
            # Query about the uploaded image
            img_base64 = image_to_base64(current_image)
            response = hf_client.image_to_text(
                image=img_base64,
                model="Salesforce/blip-image-captioning-large",
                prompt=user_query  # Ask specific question about the image
            )
            chat_history.extend([f"User: {user_query}", f"AI: {response}"])
            return jsonify({'response': response}), 200
        except Exception as e:
            return jsonify({'error': f"Image query error: {str(e)}"}), 400
    
    elif vector_store:
        vector_store_wrapper = VectorStoreIndexWrapper(vectorstore=vector_store)
        response = query_with_learning_resources(vector_store_wrapper, user_query)
        return jsonify({'response': response}), 200
    
    else:
        return jsonify({'error': 'Please upload a file first'}), 400

@app.route('/clear', methods=['POST'])
@cross_origin()
def clear_history():
    global chat_history, current_image, vector_store
    chat_history = []
    current_image = None
    vector_store = None
    return jsonify({"message": "Session cleared"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)