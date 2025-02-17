import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    UnstructuredFileLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Initialize LLM
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.3,
    max_tokens=4096
)

# Configure prompt template
prompt_template = ChatPromptTemplate.from_template(
    """Analyze the provided data and answer the question. Include:
    - Key statistics
    - Trends/patterns
    - Data limitations
    - Visualization suggestions
    
    Context: {context}
    
    Question: {question}
    
    Answer in markdown:"""
)

def process_uploaded_file(uploaded_file):
    """Handle different file formats and return documents"""
    file_ext = uploaded_file.name.split('.')[-1].lower()
    temp_path = f"temp.{file_ext}"
    
    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if file_ext == "csv":
            loader = CSVLoader(temp_path)
            df = pd.read_csv(temp_path)
            st.session_state.df = df
            return loader.load()
        elif file_ext == "pdf":
            return PyPDFLoader(temp_path).load()
        elif file_ext in ["doc", "docx"]:
            return UnstructuredFileLoader(temp_path).load()
        elif file_ext == "txt":
            return TextLoader(temp_path).load()
        else:
            st.error(f"Unsupported file type: {file_ext}")
            return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def setup_retrieval_chain(docs):
    """Create RAG pipeline with ChromaDB"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
    )

def generate_visualization(query):
    """Create interactive visualizations based on data"""
    if 'df' not in st.session_state:
        return
    
    df = st.session_state.df
    query = query.lower()
    
    try:
        # Time series analysis
        if any(kw in query for kw in ["trend", "time", "year"]):
            date_col = next((col for col in df.columns if "date" in col.lower()), None)
            if date_col:
                numeric_cols = df.select_dtypes(include='number').columns
                for col in numeric_cols:
                    fig = px.line(df, x=date_col, y=col, title=f"{col} Over Time")
                    st.plotly_chart(fig)
        
        # Correlation analysis
        if any(kw in query for kw in ["correlat", "relationship"]):
            numeric_df = df.select_dtypes(include='number')
            if len(numeric_df.columns) >= 2:
                corr = numeric_df.corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
                st.plotly_chart(fig)
                
        # Distribution analysis
        if any(kw in query for kw in ["distribut", "histogram"]):
            numeric_cols = df.select_dtypes(include='number').columns
            for col in numeric_cols:
                fig = px.histogram(df, x=col, title=f"{col} Distribution")
                st.plotly_chart(fig)
                
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")

def main():
    st.title("📊 Data Analysis Assistant")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload data files (CSV, PDF, DOC, TXT)",
        type=["csv", "pdf", "doc", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        all_docs = []
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                docs = process_uploaded_file(uploaded_file)
                if docs:
                    all_docs.extend(docs)
        
        if all_docs:
            st.session_state.rag_chain = setup_retrieval_chain(all_docs)
            st.success("Data processing complete! Ask questions below.")
            
            if 'df' in st.session_state:
                st.subheader("Data Preview")
                st.dataframe(st.session_state.df.head())
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about your data:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
            with st.spinner("Analyzing..."):
                response = st.session_state.rag_chain.invoke(prompt)
                answer = response.content
                
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    generate_visualization(prompt)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
        
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()