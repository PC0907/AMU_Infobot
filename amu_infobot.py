from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.chat_message_histories import RedisChatMessageHistory
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

@dataclass
class ContextWindow:
    max_tokens: int = 2000
    relevance_threshold: float = 0.7
    time_weight_factor: float = 0.8

class EnhancedContextManager:
    def __init__(self, vector_store):
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=vector_store,
            decay_rate=0.01,
            k=3
        )
        
        self.context_window = ContextWindow()
        self.conversation_state = {}
    
    def get_relevant_context(self, query: str) -> List[Document]:
        current_context = self.retriever.get_relevant_documents(query)
        chat_history = self.memory.chat_memory.messages
        merged_context = self._merge_context(current_context, chat_history)
        return self._filter_by_relevance(merged_context, query)
    
    def _merge_context(self, current_context, chat_history):
        history_docs = [
            Document(
                page_content=msg.content,
                metadata={"timestamp": datetime.now().timestamp()}
            ) for msg in chat_history
        ]
        all_docs = current_context + history_docs
        return self._remove_duplicates(all_docs)
    
    def _filter_by_relevance(self, docs: List[Document], query: str) -> List[Document]:
        scores = self._calculate_relevance_scores(docs, query)
        relevant_docs = [
            doc for doc, score in zip(docs, scores)
            if score >= self.context_window.relevance_threshold
        ]
        return self._trim_to_token_limit(relevant_docs)
    
    def update_conversation_state(self, query: str, response: str):
        self.memory.save_context({"input": query}, {"answer": response})
        self.conversation_state.update({
            "last_query": query,
            "last_response": response,
            "timestamp": datetime.now().timestamp()
        })

    def _calculate_relevance_scores(self, docs: List[Document], query: str) -> List[float]:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        query_embedding = embeddings.embed_query(query)
        doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in docs]
        scores = [np.dot(query_embedding, doc_emb) for doc_emb in doc_embeddings]
        return scores

    def _remove_duplicates(self, docs: List[Document]) -> List[Document]:
        seen_content = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
        return unique_docs

    def _trim_to_token_limit(self, docs: List[Document]) -> List[Document]:
        total_tokens = 0
        trimmed_docs = []
        for doc in docs:
            token_count = len(doc.page_content.split())
            if total_tokens + token_count <= self.context_window.max_tokens:
                trimmed_docs.append(doc)
                total_tokens += token_count
        return trimmed_docs

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_vector_store(text_chunks):
    try:
        # Check if saved index exists
        if os.path.exists("faiss_store"):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.load_local(
                "faiss_store", 
                embeddings, 
                allow_dangerous_deserialization=True  # Added this parameter
            )
            print("Loaded existing vector store from disk")
            return vector_store
        
        # If not exists, create new index and save it
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_store")
        print("Created and saved new vector store")
        return vector_store
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {str(e)}")
        print(f"Detailed error: {e}")
        return None


def get_conversational_chain(vector_store):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question", 
        output_key="answer",
        return_messages=True
    )

    prompt_template = """You are Gemini, an expert AMU admission counselor. You help students by providing complete, accurate information in a friendly and helpful way.

    Important Guidelines:
    1. Analyze the type of question being asked
    2. Structure your response based on the question type:
       - For course queries: Give complete details about branches, eligibility, fees, etc.
       - For admission queries: Explain the process step by step
       - For eligibility queries: Provide clear criteria and requirements
       - For general queries: Give relevant information in a conversational way
    
    3. Always:
       - Be comprehensive but clear
       - Include specific details (numbers, dates, requirements)
       - Explain any technical terms
       - Be conversational and encouraging
       - Suggest related helpful information
    
    4. NEVER tell users to refer to the guide or specific pages

    Context: {context}
    Chat History: {chat_history}
    Human: {question}
    Assistant: """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.5
        ),
        retriever=vector_store.as_retriever(
            search_kwargs={
                "k": 8,  # Increased for more comprehensive context
                "search_type": "mmr",  # Using MMR for diverse results
                "fetch_k": 12,
                "lambda_mult": 0.7
            }
        ),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        chain_type="stuff",
        return_source_documents=True,
        output_key="answer"
    )
   
    return chain, EnhancedContextManager(vector_store)

# Modify the text chunking for better context preservation
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Increased chunk size
        chunk_overlap=200,  # Good overlap for context
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
# Modify the text chunking to create overlapping chunks that preserve context

def process_user_input(user_question, conversation_chain, context_manager):
    try:
        response = conversation_chain.invoke({
            "question": user_question
        })
        answer = response["answer"]  # Extract just the answer
        context_manager.update_conversation_state(user_question, answer)
        return answer
    except Exception as e:
        print(f"Error in process_user_input: {str(e)}")
        raise


def main():
    st.set_page_config(page_title="Enhanced Chat with Gemini")
    
    # Debug initialization
    if 'debug_messages' not in st.session_state:
        st.session_state.debug_messages = []

    def debug_print(msg):
        print(msg)
        st.session_state.debug_messages.append(msg)
        st.sidebar.write(msg)

    debug_print("Starting initialization...")

    # API Key check
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("API Key not found!")
            return
        genai.configure(api_key=api_key)
        debug_print("API configured successfully")
    except Exception as e:
        st.error(f"API configuration error: {str(e)}")
        return

    try:
        if 'vector_store' not in st.session_state:
            debug_print("Checking for existing vector store")
            
            # First try to load existing vector store
            if os.path.exists("faiss_store"):
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.load_local(
                    "faiss_store", 
                    embeddings,
                    allow_dangerous_deserialization=True  # Added this parameter
                )
                debug_print("Loaded existing vector store from disk")
            else:
                # If no existing store, process PDF and create new one
                debug_print("Starting PDF processing")
                file_path = r"C:\Users\Hassan\myenv2\guide to admissions.pdf"
                if not os.path.exists(file_path):
                    st.error(f"PDF file not found at {file_path}")
                    return
                    
                with open(file_path, 'rb') as file:
                    pdf_text = get_pdf_text(file)
                    debug_print(f"PDF text extracted: {len(pdf_text)} characters")
                    
                    if not pdf_text.strip():
                        st.error("No text extracted from PDF")
                        return
                    
                    text_chunks = get_text_chunks(pdf_text)
                    debug_print(f"Created {len(text_chunks)} chunks")
                    
                    vector_store = get_vector_store(text_chunks)
                    if vector_store is None:
                        st.error("Failed to create vector store")
                        return
                    
                    debug_print("Created and saved new vector store")
            
            st.session_state.vector_store = vector_store
            st.session_state.chain, st.session_state.context_manager = get_conversational_chain(vector_store)
            debug_print("Initialization complete")

        # Rest of your chat interface code remains unchanged
        st.header("Chat with Gemini about Admissions")
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        user_question = st.chat_input("Ask a question:")
        if user_question:
            debug_print(f"Processing question: {user_question}")
            try:
                response = process_user_input(user_question, st.session_state.chain, st.session_state.context_manager)
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        debug_print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
