import streamlit as st
import os
import tempfile
from typing import List, Dict, Tuple
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from groq import Groq
import hashlib
from datetime import datetime
import re
import io

# Page configuration
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state['documents'] = {}
if 'qa_system' not in st.session_state:
    st.session_state['qa_system'] = None
if 'system_initialized' not in st.session_state:
    st.session_state['system_initialized'] = False

class DocumentQASystem:
    def __init__(self):
        # Initialize embedding model
        self._init_embedding_model()
        
        # Initialize ChromaDB (in-memory for Streamlit Cloud)
        self._init_vector_db()
        
        self.documents = {}
    
    @st.cache_resource
    def _init_embedding_model(_self):
        """Initialize and cache the embedding model"""
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    def _init_vector_db(self):
        """Initialize ChromaDB"""
        self.chroma_client = chromadb.Client()
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection("documents")
        except:
            self.collection = self.chroma_client.create_collection("documents")
        
        # Use cached embedding model
        self.embedding_model = self._init_embedding_model()
    
    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF file bytes using pdfplumber"""
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:  # Check if text was extracted
                        text += page_text + "\n"
        except Exception as e:
            st.error(f"Error extracting PDF: {str(e)}")
        return text
    
    def extract_text_from_docx(self, file_bytes: bytes) -> str:
        """Extract text from Word document bytes"""
        text = ""
        try:
            docx_file = io.BytesIO(file_bytes)
            doc = docx.Document(docx_file)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            st.error(f"Error extracting DOCX: {str(e)}")
        return text
    
    def extract_text_from_txt(self, file_bytes: bytes) -> str:
        """Extract text from text file bytes"""
        try:
            return file_bytes.decode('utf-8')
        except Exception as e:
            st.error(f"Error extracting TXT: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def process_document(self, file_bytes: bytes, file_name: str) -> str:
        """Process uploaded document and add to vector database"""
        try:
            # Extract text based on file type
            if file_name.lower().endswith('.pdf'):
                text = self.extract_text_from_pdf(file_bytes)
            elif file_name.lower().endswith('.docx'):
                text = self.extract_text_from_docx(file_bytes)
            elif file_name.lower().endswith('.txt'):
                text = self.extract_text_from_txt(file_bytes)
            else:
                return f"❌ Unsupported file type: {file_name}"
            
            if not text.strip():
                return f"❌ No text extracted from {file_name}"
            
            # Create document ID
            doc_id = hashlib.md5(file_name.encode()).hexdigest()
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            # Generate embeddings
            with st.spinner("Generating embeddings..."):
                embeddings = self.embedding_model.encode(chunks)
            
            # Store in ChromaDB
            ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": file_name, "chunk_id": i} for i in range(len(chunks))]
            
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            # Store document metadata
            self.documents[doc_id] = {
                "name": file_name,
                "chunks": len(chunks),
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Update session state
            st.session_state['documents'] = self.documents
            
            return f"✅ Successfully processed {file_name} ({len(chunks)} chunks)"
            
        except Exception as e:
            return f"❌ Error processing {file_name}: {str(e)}"
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant document chunks"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            # Format results
            search_results = []
            for i in range(len(results['documents'][0])):
                search_results.append({
                    'content': results['documents'][0][i],
                    'source': results['metadatas'][0][i]['source'],
                    'chunk_id': results['metadatas'][0][i]['chunk_id']
                })
            
            return search_results
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using Groq API"""
        try:
            # Get API key from secrets
            groq_api_key = st.secrets.get("GROQ_API_KEY", "")
            if not groq_api_key:
                return "❌ Groq API key not configured. Please add it to Streamlit secrets."
            
            # Initialize Groq client
            groq_client = Groq(api_key=groq_api_key)
            
            # Prepare context
            context = "\n\n".join([f"Source: {chunk['source']}\nContent: {chunk['content']}" 
                                 for chunk in context_chunks])
            
            # Create prompt
            prompt = f"""Based on the following documents, please answer the question. If the answer is not in the documents, say so.

Context:
{context}

Question: {query}

Instructions:
1. Answer based only on the provided context
2. Include source citations in your answer
3. If information is not available in the context, clearly state that
4. Be concise but comprehensive

Answer:"""
            
            # Call Groq API
            with st.spinner("Generating answer..."):
                response = groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1000
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}"
    
    def ask_question(self, question: str) -> str:
        """Main Q&A function"""
        if not question.strip():
            return "❌ Please enter a question."
        
        # Check if any documents are loaded
        if not self.documents:
            return "❌ No documents loaded. Please upload documents first."
        
        # Search for relevant chunks
        relevant_chunks = self.search_documents(question)
        
        if not relevant_chunks:
            return "❌ No relevant information found in the documents."
        
        # Generate answer
        answer = self.generate_answer(question, relevant_chunks)
        
        # Add source information
        sources = list(set([chunk['source'] for chunk in relevant_chunks]))
        source_info = f"\n\n📚 **Sources**: {', '.join(sources)}"
        
        return answer + source_info

# Initialize system
@st.cache_resource
def initialize_qa_system():
    """Initialize and cache the QA system"""
    return DocumentQASystem()

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📚 Document Q&A System</h1>', unsafe_allow_html=True)
    st.markdown("Upload your documents and ask questions about them!")
    
    # Initialize QA system
    if not st.session_state['system_initialized']:
        with st.spinner("Initializing system..."):
            st.session_state['qa_system'] = initialize_qa_system()
            st.session_state['system_initialized'] = True
        st.success("✅ System initialized successfully!")
    
    qa_system = st.session_state['qa_system']
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Document upload
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    result = qa_system.process_document(
                        uploaded_file.read(),
                        uploaded_file.name
                    )
                
                if "✅" in result:
                    st.markdown(f'<div class="success-box">{result}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="error-box">{result}</div>', unsafe_allow_html=True)
        
        # Document status
        st.subheader("📊 Loaded Documents")
        if st.session_state['documents']:
            for doc_id, info in st.session_state['documents'].items():
                st.write(f"📄 **{info['name']}**")
                st.write(f"   • {info['chunks']} chunks")
                st.write(f"   • {info['processed_at']}")
                st.write("---")
        else:
            st.write("No documents loaded yet.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("❓ Ask Questions")
        
        # Question input
        question = st.text_area(
            "Your Question",
            placeholder="What would you like to know about your documents?",
            height=100
        )
        
        if st.button("🔍 Ask Question", type="primary", disabled=not question.strip()):
            if not st.session_state['documents']:
                st.markdown('<div class="error-box">❌ Please upload documents first.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Searching and generating answer..."):
                    answer = qa_system.ask_question(question)
                
                st.subheader("💡 Answer")
                st.markdown(answer)
    
    with col2:
        st.header("ℹ️ Instructions")
        
        st.markdown("""
        ### How to Use:
        
        1. **Upload Documents**:
           - Use the sidebar file uploader
           - Supports PDF, DOCX, TXT
           - Click "Process Document"
        
        2. **Ask Questions**:
           - Type your question in the text area
           - Click "Ask Question"
           - Get answers with sources
        
        ### API Configuration:
        Add your Groq API key to Streamlit secrets:
        ```
        GROQ_API_KEY = "your_key_here"
        ```
        
        ### Features:
        - ✅ Multiple document formats
        - ✅ Source citations
        - ✅ 24/7 availability
        - ✅ Team access
        """)

if __name__ == "__main__":
    main()