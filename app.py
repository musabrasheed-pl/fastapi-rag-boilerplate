import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Configuration
class Config:
    VECTORSTORE_TYPE = os.getenv("VECTORSTORE_TYPE", "pinecone").lower()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'default-index')
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

class RAGApp:
    def __init__(self):
        self.vectorstore = None
        self.chat_history = []
        if not Config.OPENAI_API_KEY:
            raise ValueError("Please set up your OPENAI_API_KEY in the .env file")
        self.embedding = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY, model=Config.EMBEDDING_MODEL)

    def process_files(self, files):
        all_chunks = []
        for file in files:
            try:
                if file.filename.endswith(".pdf"):
                    text = self.extract_text_from_pdf(file)
                    chunks = self.split_text_into_chunks(text)
                    all_chunks.extend([(chunk, file.filename) for chunk in chunks])
            except Exception as e:
                print(f"Error processing {file.filename}: {str(e)}")
        return all_chunks

    def extract_text_from_pdf(self, file) -> str:
        pdf_reader = PdfReader(file.file)
        return ''.join(page.extract_text() or "" for page in pdf_reader.pages)

    def split_text_into_chunks(self, text: str):
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        return splitter.split_text(text)

    def initialize_vectorstore(self, chunks):
        texts = [chunk for chunk, _ in chunks]
        metadatas = [{"source": filename} for _, filename in chunks]

        if Config.VECTORSTORE_TYPE == "pinecone":
            return self.initialize_pinecone(texts, metadatas)
        elif Config.VECTORSTORE_TYPE == "faiss":
            return self.initialize_faiss(texts)
        elif Config.VECTORSTORE_TYPE == "chroma":
            return self.initialize_chroma(texts)
        else:
            raise ValueError("Unsupported VECTORSTORE_TYPE.")

    def initialize_pinecone(self, texts, metadatas):
        from pinecone import Pinecone, ServerlessSpec
        from langchain.vectorstores import Pinecone as LangchainPinecone

        if not Config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is missing")

        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        if Config.PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
            pc.create_index(
                name=Config.PINECONE_INDEX_NAME,
                dimension=3072,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            while not pc.describe_index(Config.PINECONE_INDEX_NAME).status["ready"]:
                time.sleep(1)

        index = pc.Index(Config.PINECONE_INDEX_NAME)
        vectorstore = LangchainPinecone(index, self.embedding, "text")
        vectorstore.add_texts(texts, metadatas=metadatas)
        return vectorstore

    def initialize_faiss(self, texts):
        from langchain.vectorstores import FAISS
        return FAISS.from_texts(texts, self.embedding)

    def initialize_chroma(self, texts):
        from langchain.vectorstores import Chroma
        return Chroma.from_texts(texts, self.embedding, collection_name="documents")

    def setup_qa_chain(self, vectorstore):
        llm = self.create_llm()
        if not llm:
            return None

        prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""
            You're an assistant that answers questions strictly based on the provided documents.
            Consider the following chat history for additional context:
            {chat_history}

            Question: {question}
            Context: {context}

            Answer:
            """
        )
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

    def create_llm(self):
        from langchain.chat_models import ChatOpenAI, ChatAnthropic, ChatGooglePalm
        if Config.LLM_PROVIDER == "openai":
            return ChatOpenAI(temperature=0.3, model_name=Config.LLM_MODEL, openai_api_key=Config.OPENAI_API_KEY)
        elif Config.LLM_PROVIDER == "claude":
            return ChatAnthropic(temperature=0.3, model=Config.LLM_MODEL, anthropic_api_key=Config.CLAUDE_API_KEY)
        elif Config.LLM_PROVIDER == "gemini":
            return ChatGooglePalm(temperature=0.3, model_name=Config.LLM_MODEL, google_api_key=Config.GEMINI_API_KEY)
        else:
            return None

    def answer_query(self, vectorstore, query):
        chain = self.setup_qa_chain(vectorstore)
        result = chain({"question": query, "chat_history": self.chat_history[-5:]})
        answer = result["answer"]
        self.chat_history.append((query, answer))
        return answer

app_instance = RAGApp()

@app.post("/upload/")
async def upload_files(files: list[UploadFile] = File(...)):
    chunks = app_instance.process_files(files)
    if chunks:
        app_instance.vectorstore = app_instance.initialize_vectorstore(chunks)
        return {"message": "Files processed successfully", "vectorstore_initialized": True}
    else:
        raise HTTPException(status_code=400, detail="Failed to process files.")

@app.post("/query/")
async def query_documents(query: str = Form(...)):
    if not app_instance.vectorstore:
        try:
            app_instance.vectorstore = app_instance.initialize_vectorstore([])  # Load existing Pinecone vectorstore
        except Exception as e:
            raise HTTPException(status_code=400, detail="Vectorstore not initialized and no prior data found.")
    answer = app_instance.answer_query(app_instance.vectorstore, query)
    history = "\n".join([f"Q: {q}\nA: {a}" for q, a in app_instance.chat_history])
    return {"answer": answer, "chat_history": history}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return "RAG Application API"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
