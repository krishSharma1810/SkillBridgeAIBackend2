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
from langchain_google_genai import ChatGoogleGenerativeAI
from googleapiclient.discovery import build
from duckduckgo_search import DDGS
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

# API keys and configuration
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Initialize Gemini models
genai.configure(api_key=GEMINI_API_KEY)
vision_model = genai.GenerativeModel('gemini-2.0-flash')
text_model = genai.GenerativeModel('gemini-1.5-pro')

# Initialize Cassandra and LangChain
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY)

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
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    raw_text = ""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        pdf_file.save(temp_file.name)
        with pdfplumber.open(temp_file.name) as pdf:
            for page in pdf.pages:
                raw_text += page.extract_text() + "\n"
        os.unlink(temp_file.name)
    return raw_text if raw_text.strip() else "Error: Could not extract text from PDF."

def chunk_text(raw_text):
    """Split text into chunks for vector storage"""
    return text_splitter.split_text(raw_text)

def initialize_vector_store(text_chunks):
    """Initialize Cassandra vector store with text chunks"""
    global vector_store
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Cassandra(
        embedding=embedding,
        table_name="QA_Mini_Demo",
        session=None,
        keyspace=None
    )
    vector_store.clear()
    vector_store.add_texts(text_chunks)
    return VectorStoreIndexWrapper(vectorstore=vector_store)

def process_image(image_file):
    """Process uploaded image and extract text/description"""
    global current_image
    try:
        img_bytes = image_file.read()
        current_image = Image.open(BytesIO(img_bytes))
        response = vision_model.generate_content([
            "Describe this image in detail, including any text present.", 
            current_image
        ])
        return response.text
    except Exception as e:
        return f"Error processing image: {str(e)}"

def query_with_learning_resources(vector_store_wrapper, user_query):
    """Handle queries with fallback to external resources"""
    global chat_history
    retrieved_docs = vector_store_wrapper.vectorstore.similarity_search(user_query, k=3)
    combined_context = " ".join([doc.page_content for doc in retrieved_docs])
    
    if retrieved_docs:
        response_text = text_model.generate_content(
            f"You are an AI tutor. Answer using the document first.\n{combined_context}\n{user_query}"
        ).text
    else:
        response_text = agent_executor.run(user_query)
    
    chat_history.append(f"User: {user_query}")
    chat_history.append(f"AI: {response_text}")
    return response_text

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
            response = vision_model.generate_content([user_query, current_image])
            chat_history.append(f"User: {user_query}")
            chat_history.append(f"AI: {response.text}")
            return jsonify({'response': response.text}), 200
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