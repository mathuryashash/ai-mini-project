# main.py
import logging
import threading
from tkinter import Tk, Text, Entry, Button, END, WORD, scrolledtext
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq  # --- CHANGE: Import ChatGroq ---

# Make sure you have a config.py file with your settings
import config

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load environment variables ---
load_dotenv()

# --- 1. PDF Processing and Vector Store ---
class PDFProcessor:
    def __init__(self, pdf_dir: Path, pdf_files: list[str], embedding_model_name: str, faiss_index_dir: Path):
        self.pdf_dir = pdf_dir
        self.pdf_files = pdf_files
        self.embedding_model_name = embedding_model_name
        self.faiss_index_dir = faiss_index_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.db = self._load_or_create_faiss_index()

    def _load_or_create_faiss_index(self):
        if self.faiss_index_dir.exists():
            logging.info(f"Loading FAISS index from {self.faiss_index_dir}")
            return FAISS.load_local(str(self.faiss_index_dir), self.embeddings, allow_dangerous_deserialization=True)
        else:
            logging.info("Creating FAISS index from PDFs...")
            docs = []
            for pdf_name in self.pdf_files:
                pdf_path = self.pdf_dir / pdf_name
                if pdf_path.exists():
                    loader = PyPDFLoader(str(pdf_path))
                    docs.extend(loader.load())
                    logging.info(f"Loaded {pdf_name}")
                else:
                    logging.warning(f"PDF file not found: {pdf_path}")
            
            if not docs:
                raise ValueError("No documents loaded. Cannot create FAISS index.")

            db = FAISS.from_documents(docs, self.embeddings)
            db.save_local(str(self.faiss_index_dir))
            logging.info(f"FAISS index saved to {self.faiss_index_dir}")
            return db

    def get_retriever(self):
        return self.db.as_retriever()

# --- 2. Core Chatbot Logic (Refactored for Conversational History) ---
class ChatbotCore:
    # --- CHANGE: Simplified the __init__ signature ---
    def __init__(self, retriever, llm_repo_id: str, llm_temperature: float):
        # --- CHANGE: Replaced HuggingFaceEndpoint with ChatGroq ---
        self.llm = ChatGroq(
            model_name=llm_repo_id,
            temperature=llm_temperature
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.retriever = retriever
        self.conversational_rag_chain = self._create_conversational_rag_chain()

    def _create_conversational_rag_chain(self):
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain

    def query(self, user_query: str) -> str:
        try:
            chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
            response = self.conversational_rag_chain.invoke({
                "input": user_query,
                "chat_history": chat_history
            })
            self.memory.save_context({"input": user_query}, {"answer": response["answer"]})
            answer = response.get("answer", "No answer returned.")
            logging.info(f"Query: {user_query}, Response: {answer}")
            return answer
        except Exception as e:
            logging.error(f"Error during chain invocation: {e}")
            return f"Error: {e}"

# --- 3. GUI (Refactored for Threading) ---
class ChatbotGUI:
    def __init__(self, master, chatbot_core: ChatbotCore):
        self.master = master
        self.chatbot_core = chatbot_core
        self.setup_gui()

    def setup_gui(self):
        self.master.title(config.WINDOW_TITLE)
        self.master.geometry(config.WINDOW_GEOMETRY)
        self.chat_display = scrolledtext.ScrolledText(self.master, wrap=WORD, font=config.CHAT_FONT, bg=config.CHAT_BG, fg=config.CHAT_FG)
        self.chat_display.pack(padx=10, pady=10, fill="both", expand=True)
        self.chat_display.insert(END, config.WELCOME_MESSAGE)
        self.chat_display.config(state="disabled")
        self.query_entry = Entry(self.master, font=config.ENTRY_FONT)
        self.query_entry.pack(padx=10, pady=5, fill="x")
        self.query_entry.bind("<Return>", self._handle_enter_key)
        self.ask_button = Button(self.master, text="Ask", font=config.BUTTON_FONT, command=self.ask_question_threaded)
        self.ask_button.pack(pady=5)

    def _handle_enter_key(self, event=None):
        self.ask_question_threaded()

    def ask_question_threaded(self):
        user_query = self.query_entry.get().strip()
        if not user_query:
            return
        self._display_message(f"You: {user_query}\n")
        self.query_entry.delete(0, END)
        self.ask_button.config(state="disabled")
        self.query_entry.config(state="disabled")
        thread = threading.Thread(target=self._get_bot_response, args=(user_query,))
        thread.start()

    def _get_bot_response(self, user_query):
        answer = self.chatbot_core.query(user_query)
        self.master.after(0, self._display_message, f"Bot: {answer}\n\n")

    def _display_message(self, message):
        self.chat_display.config(state="normal")
        self.chat_display.insert(END, message)
        self.chat_display.config(state="disabled")
        self.chat_display.see(END)
        self.ask_button.config(state="normal")
        self.query_entry.config(state="normal")

# --- Main Application Execution ---
if __name__ == "__main__":
    try:
        pdf_processor = PDFProcessor(
            pdf_dir=config.PDF_DIR,
            pdf_files=config.PDF_FILES,
            embedding_model_name=config.EMBEDDING_MODEL_NAME,
            faiss_index_dir=config.FAISS_INDEX_DIR
        )
        # --- CHANGE: Updated the ChatbotCore initialization ---
        chatbot_core = ChatbotCore(
            retriever=pdf_processor.get_retriever(),
            llm_repo_id=config.LLM_REPO_ID,
            llm_temperature=config.LLM_TEMPERATURE,
        )
        window = Tk()
        chatbot_gui = ChatbotGUI(window, chatbot_core)
        window.mainloop()
    except Exception as e:
        logging.critical(f"Application failed to start: {e}", exc_info=True)