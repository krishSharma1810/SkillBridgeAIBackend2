from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import cassio
import pdfplumber
from langchain.vectorstores.cassandra import Cassandra
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

# Load environment variables
load_dotenv()

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)
CORS(app, resources={r"/*": {"origins": "*"}})

# API keys and configuration
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY)
print("LLM initialized successfully.")

vector_store = None
chat_history = []

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    return text_splitter.split_text(raw_text)

def initialize_vector_store(text_chunks):
    global vector_store
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Cassandra(embedding=embedding, table_name="QA_Mini_Demo", session=None, keyspace=None)
    vector_store.clear()
    vector_store.add_texts(text_chunks)
    return VectorStoreIndexWrapper(vectorstore=vector_store)

def get_blog_articles(query):
    results = list(DDGS().text(f"{query} site:medium.com OR site:dev.to OR site:towardsdatascience.com", max_results=5))
    return "\n".join([f"{result['title']} - {result['href']}" for result in results])

def get_youtube_videos(query):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(q=query, part="snippet", maxResults=3, type="video")
    response = request.execute()
    return "\n".join([f"🎥 {video['snippet']['title']} - https://www.youtube.com/watch?v={video['id']['videoId']}" for video in response["items"]])

# Tool initialization
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
tools = [
    Tool(name="Wikipedia", func=wikipedia_tool.run, description="Search Wikipedia for information."),
    Tool(name="ArXiv", func=arxiv_tool.run, description="Retrieve academic papers from ArXiv."),
    Tool(name="DuckDuckGo", func=get_blog_articles, description="Fetch blog articles related to the topic."),
    Tool(name="YouTube", func=get_youtube_videos, description="Fetch relevant YouTube videos."),
]

agent_executor = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_pdf():
    global chat_history, vector_store
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    chat_history = []
    raw_text = extract_text_from_pdf(file)
    if raw_text.startswith("Error:"):
        return jsonify({'error': raw_text}), 400
    text_chunks = chunk_text(raw_text)
    vector_store_wrapper = initialize_vector_store(text_chunks)
    return jsonify({'message': 'PDF uploaded and processed successfully'}), 200

@app.route('/query', methods=['POST'])
@cross_origin()
def query():
    global vector_store
    if not vector_store:
        return jsonify({'error': 'Please upload a PDF first'}), 400
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    user_query = data['query']
    vector_store_wrapper = VectorStoreIndexWrapper(vectorstore=vector_store)
    response = query_with_learning_resources(vector_store_wrapper, user_query)
    return jsonify({'response': response}), 200

@app.route('/clear', methods=['POST'])
@cross_origin()
def clear_history():
    global chat_history
    chat_history = []
    return jsonify({"message": "History cleared successfully"}), 200


def query_with_learning_resources(vector_store_wrapper, user_query):
    global chat_history
    retrieved_docs = vector_store_wrapper.vectorstore.similarity_search(user_query, k=3)
    combined_context = " ".join([doc.page_content for doc in retrieved_docs])
    system_prompt = "You are an AI tutor. Answer using the document first. If the document lacks relevant information, search Wikipedia, ArXiv, DuckDuckGo, and YouTube."
    if retrieved_docs:
        response_text = llm.invoke([{"role": "user", "content": f"{system_prompt}\n{combined_context}\n{user_query}"}]).content
    else:
        response_text = agent_executor.run(user_query)
    chat_history.append(f"User: {user_query}")
    chat_history.append(f"AI: {response_text}")
    return response_text

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)