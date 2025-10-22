from pathlib import Path

# --- Paths ---
# Base directory for PDFs, relative to the project root
PDF_DIR = Path("books")
PDF_FILES = ["module_1.pdf", "module_2.pdf", "module_3.pdf", "module_4.pdf", "module_5.pdf"]
FAISS_INDEX_DIR = Path("faiss_index")

# --- Embeddings ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM Model (Configured for Groq) ---
LLM_REPO_ID = "llama-3.1-8b-instant"  # Changed to a model supported by Groq
LLM_TEMPERATURE = 0.5
# LLM_TASK and LLM_MAX_NEW_TOKENS are no longer needed for the Groq implementation

# --- GUI Configuration ---
WINDOW_TITLE = "PDF Chatbot"
WINDOW_GEOMETRY = "700x500"
CHAT_FONT = ("Helvetica", 14)
CHAT_BG = "white"
CHAT_FG = "black"
ENTRY_FONT = ("Helvetica", 14)
BUTTON_FONT = ("Helvetica", 14)
WELCOME_MESSAGE = "Welcome! Ask me anything from the PDFs.\n\n"