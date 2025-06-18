import os
import streamlit as st
import pickle
import json
from typing import List, Dict, Any
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
import PyPDF2
import docx
from io import BytesIO
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot with Gemini",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #f0f2f6;
    }
    .user-message {
        background-color: #e1f5fe;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-box {
        background-color: #fff3e0;
        border: 1px solid #ffcc02;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DocumentProcessor:
    """Handles document loading and processing"""
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    @staticmethod
    def process_uploaded_file(uploaded_file) -> str:
        """Process uploaded file and extract text"""
        if uploaded_file is None:
            return ""
        
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension in ['.txt', '.md']:
            return str(uploaded_file.read(), "utf-8")
        elif file_extension == '.pdf':
            return DocumentProcessor.extract_text_from_pdf(uploaded_file.read())
        elif file_extension == '.docx':
            return DocumentProcessor.extract_text_from_docx(uploaded_file.read())
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return ""

class RAGChatbot:
    """Main RAG Chatbot class with model persistence"""
    
    def __init__(self, model_dir: str = "rag_model"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.embedding_model = None
        self.vector_store = None
        self.documents = []
        self.llm_model = None
        self.model_metadata = {
            "created_at": None,
            "last_updated": None,
            "document_count": 0,
            "chunk_count": 0
        }
        
        self.initialize_models()
   def initialize_models(self):
    """Initialize or load the embedding model"""
    try:
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        st.success("✅ Embedding model loaded successfully")
    except Exception as e:
        self.embedding_model = None
        st.error("❌ Failed to load embedding model. See logs for details.")
        print(f"[DEBUG] Embedding model load failure: {e}")

    def setup_gemini_client(self, api_key: str):
        """Setup Gemini client with API key"""
        try:
            genai.configure(api_key=api_key)
            # Test the connection by listing models
            models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if not models:
                raise Exception("No suitable Gemini models found. Please check your API key and permissions.")
            self.llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            st.success("✅ Gemini API connected successfully")
            return True
        except Exception as e:
            st.error(f"❌ Failed to connect to Gemini API: {str(e)}")
            return False
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def add_documents(self, documents: List[str], filenames: List[str] = None):
        """Add documents to the vector store and save model"""
        if not documents:
            return False
        
        all_chunks = []
        chunk_metadata = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, doc in enumerate(documents):
            status_text.text(f"Processing document {i+1}/{len(documents)}")
            progress_bar.progress((i + 1) / len(documents))
            
            filename = filenames[i] if filenames else f"Document_{i+1}"
            chunks = self.chunk_text(doc)
            all_chunks.extend(chunks)
            
            for j, chunk in enumerate(chunks):
                chunk_metadata.append({
                    'filename': filename,
                    'chunk_id': j,
                    'content': chunk,
                    'doc_index': i
                })
        
        if all_chunks:
            status_text.text("Generating embeddings...")
            embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
            
            # Initialize or recreate FAISS index
            dimension = embeddings.shape[1]
            if self.vector_store is None:
                self.vector_store = faiss.IndexFlatL2(dimension)
            else:
                # If adding to existing index
                pass
            
            self.vector_store.add(embeddings.astype('float32'))
            self.documents.extend(chunk_metadata)
            
            # Update metadata
            self.model_metadata.update({
                "last_updated": datetime.now().isoformat(),
                "document_count": len(set([doc['filename'] for doc in self.documents])),
                "chunk_count": len(self.documents)
            })
            
            if self.model_metadata["created_at"] is None:
                self.model_metadata["created_at"] = datetime.now().isoformat()
            
            # Save the model
            self.save_model()
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Successfully processed {len(documents)} documents into {len(all_chunks)} chunks")
            return True
        
        progress_bar.empty()
        status_text.empty()
        return False
    
    def save_model(self):
        """Save the vector store and documents to disk"""
        try:
            # Save FAISS index
            if self.vector_store is not None:
                faiss.write_index(self.vector_store, str(self.model_dir / "vector_store.index"))
            
            # Save documents and metadata
            with open(self.model_dir / "documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)
            
            with open(self.model_dir / "metadata.json", "w") as f:
                json.dump(self.model_metadata, f, indent=2)
            
            st.success("✅ Model saved successfully")
            
        except Exception as e:
            st.error(f"❌ Error saving model: {str(e)}")
    
    def load_model(self):
        """Load the vector store and documents from disk"""
        try:
            # Check if model files exist
            index_path = self.model_dir / "vector_store.index"
            docs_path = self.model_dir / "documents.pkl"
            metadata_path = self.model_dir / "metadata.json"
            
            if not all([index_path.exists(), docs_path.exists(), metadata_path.exists()]):
                return False
            
            # Load FAISS index
            self.vector_store = faiss.read_index(str(index_path))
            
            # Load documents
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
            
            # Load metadata
            with open(metadata_path, "r") as f:
                self.model_metadata = json.load(f)
            
            st.success("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_exists": self.vector_store is not None,
            "document_count": self.model_metadata.get("document_count", 0),
            "chunk_count": self.model_metadata.get("chunk_count", 0),
            "created_at": self.model_metadata.get("created_at"),
            "last_updated": self.model_metadata.get("last_updated"),
            "unique_files": list(set([doc['filename'] for doc in self.documents])) if self.documents else []
        }
    
    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the most relevant document chunks for a query"""
        if self.vector_store is None or len(self.documents) == 0:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.vector_store.search(query_embedding.astype('float32'), k)
        
        relevant_chunks = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                chunk_info = self.documents[idx].copy()
                chunk_info['similarity_score'] = float(score)
                chunk_info['rank'] = i + 1
                relevant_chunks.append(chunk_info)
        
        return relevant_chunks
    
    def generate_response(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate response using Gemini"""
        if not self.llm_model:
            return "❌ Gemini API client not initialized. Please check your API key."
        
        context = "\n\n".join([
            f"[Source: {chunk['filename']}, Chunk {chunk['chunk_id']}]\n{chunk['content']}"
            for chunk in relevant_chunks
        ])
        
        system_prompt = """You are a helpful RAG (Retrieval-Augmented Generation) assistant. Your task is to answer questions based on the provided context documents.

Guidelines:
1. Answer based primarily on the provided context.
2. If the context doesn't contain sufficient information, clearly state this.
3. Be concise but comprehensive in your responses.
4. When relevant, mention which document(s) your answer comes from, citing the source filename.
5. If you're uncertain about something, express that uncertainty.
6. Maintain a helpful and professional tone."""
        
        user_prompt = f"""Context from uploaded documents:

{context}

User Question: {query}

Please provide a helpful answer based on the context above. If the context doesn't contain relevant information, please let me know."""
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            with st.spinner("🤔 Thinking with Gemini..."):
                response = self.llm_model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=1500,
                        temperature=0.7
                    )
                )
            
            return response.text
            
        except Exception as e:
            return f"❌ Error generating response from Gemini: {str(e)}"
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Main chat function"""
        if not query.strip():
            return {"response": "Please ask a question.", "sources": []}
        
        relevant_chunks = self.retrieve_relevant_chunks(query, k=5)
        
        if not relevant_chunks:
            return {
                "response": "I don't have any relevant information in the uploaded documents to answer your question. Please upload some documents first or try rephrasing your question.",
                "sources": []
            }
        
        response = self.generate_response(query, relevant_chunks)
        
        sources = [
            {
                "filename": chunk["filename"],
                "chunk_id": chunk["chunk_id"],
                "content_preview": chunk["content"][:300] + "..." if len(chunk["content"]) > 300 else chunk["content"],
                "similarity_score": chunk["similarity_score"],
                "rank": chunk["rank"]
            }
            for chunk in relevant_chunks
        ]
        
        return {
            "response": response,
            "sources": sources,
            "query": query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def main():
    # App title and description
    st.title("🤖 RAG Chatbot with Gemini")
    st.markdown("Upload documents, ask questions, and get intelligent answers based on your content!")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
        # Try to load existing model
        st.session_state.chatbot.load_model()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'api_key_set' not in st.session_state:
        st.session_state.api_key_set = False
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key Section
        st.subheader("🔑 Gemini API Key")
        api_key = st.text_input(
            "Enter your Google AI Studio API Key",
            type="password",
            help="Get your API key from Google AI Studio (aistudio.google.com)"
        )
        
        if api_key and not st.session_state.api_key_set:
            if st.session_state.chatbot.setup_gemini_client(api_key):
                st.session_state.api_key_set = True
        
        st.markdown("---")
        
        # Model Information
        st.subheader("📊 Model Status")
        model_info = st.session_state.chatbot.get_model_info()
        
        if model_info["model_exists"]:
            st.success("✅ Model Ready")
            st.write(f"**Documents:** {model_info['document_count']}")
            st.write(f"**Chunks:** {model_info['chunk_count']}")
            
            if model_info["created_at"]:
                created_date = datetime.fromisoformat(model_info["created_at"]).strftime("%Y-%m-%d %H:%M")
                st.write(f"**Created:** {created_date}")
            
            if model_info["last_updated"]:
                updated_date = datetime.fromisoformat(model_info["last_updated"]).strftime("%Y-%m-%d %H:%M")
                st.write(f"**Updated:** {updated_date}")
            
            if model_info["unique_files"]:
                with st.expander("📄 Loaded Files"):
                    for filename in model_info["unique_files"]:
                        st.write(f"• {filename}")
        else:
            st.warning("⚠️ No model loaded")
        
        st.markdown("---")
        
        # Document Upload Section
        st.subheader("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['txt', 'pdf', 'docx', 'md'],
            accept_multiple_files=True,
            help="Supported: TXT, PDF, DOCX, MD"
        )
        
        if uploaded_files:
            if st.button("🔄 Process Documents", type="primary"):
                documents = []
                filenames = []
                
                for uploaded_file in uploaded_files:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        text = DocumentProcessor.process_uploaded_file(uploaded_file)
                        if text.strip():
                            documents.append(text)
                            filenames.append(uploaded_file.name)
                        else:
                            st.warning(f"Could not extract text from {uploaded_file.name}")
                
                if documents:
                    st.session_state.chatbot.add_documents(documents, filenames)
                    st.rerun()
        
        st.markdown("---")
        
        # Model Management
        st.subheader("💾 Model Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Model"):
                st.session_state.chatbot.save_model()
        
        with col2:
            if st.button("📂 Load Model"):
                if st.session_state.chatbot.load_model():
                    st.rerun()
        
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.checkbox("I understand this will delete all data"):
                # Clear session state
                st.session_state.chatbot = RAGChatbot()
                st.session_state.chat_history = []
                st.session_state.api_key_set = False
                
                # Remove saved files
                import shutil
                if Path("rag_model").exists():
                    shutil.rmtree("rag_model")
                
                st.success("All data cleared!")
                st.rerun()
    
    # Main Chat Interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10
            with st.container():
                # User message
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🧑 You:</strong> {chat['query']}
                    <small style="color: #666; float: right;">{chat['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Bot response
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Assistant:</strong><br>{chat['response']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources
                if chat.get('sources'):
                    with st.expander(f"📚 Sources ({len(chat['sources'])} found)"):
                        for j, source in enumerate(chat['sources']):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>#{source['rank']} - {source['filename']}</strong> 
                                <small>(Similarity: {source['similarity_score']:.3f})</small><br>
                                <em>{source['content_preview']}</em>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")
    
    # Chat input
    if not st.session_state.api_key_set:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar to start chatting.")
    elif not model_info["model_exists"]:
        st.info("ℹ️ Upload some documents first to start asking questions!")
    else:
        # Chat input form
        with st.form("chat_form", clear_on_submit=True):
            query = st.text_area(
                "Ask a question about your documents:",
                placeholder="What would you like to know?",
                height=100,
                key="chat_input"
            )
            
            col1, col2, col3 = st.columns([1, 1, 4])
            
            with col1:
                send_button = st.form_submit_button("Send 🚀", type="primary")
            
            with col2:
                clear_chat = st.form_submit_button("Clear Chat 🗑️")
        
        if clear_chat:
            st.session_state.chat_history = []
            st.rerun()
        
        if send_button and query.strip():
            with st.spinner("Processing your question..."):
                result = st.session_state.chatbot.chat(query)
                st.session_state.chat_history.append(result)
                st.rerun()
    
    # Footer with instructions
    with st.expander("ℹ️ How to Use This Application"):
        st.markdown("""
        ### Getting Started:
        1. **API Key**: Get your API key from [Google AI Studio](https://aistudio.google.com/) and enter it in the sidebar.
        2. **Upload Documents**: Use the file uploader to add your documents (supports TXT, PDF, DOCX, MD).
        3. **Process**: Click "Process Documents" to add them to the knowledge base.
        4. **Chat**: Ask questions about your documents in the chat interface.
        
        ### Features:
        - **Document Processing**: Automatically chunks documents for optimal retrieval.
        - **Semantic Search**: Uses advanced embeddings to find relevant content.
        - **Source Citations**: Shows which documents were used to answer your questions.
        - **Model Persistence**: Your processed documents are saved and can be reloaded.
        - **Chat History**: Keeps track of your conversation.
        
        ### Supported File Types:
        - **TXT/MD**: Plain text and Markdown files.
        - **PDF**: Extracts text from PDF documents.
        - **DOCX**: Microsoft Word documents.
        
        ### Tips for Better Results:
        - Upload relevant, high-quality documents.
        - Ask specific questions.
        - Use clear, descriptive language.
        - Check the sources to understand where answers come from.
        """)

if __name__ == "__main__":
    main()
