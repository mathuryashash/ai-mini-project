# AI Mini Project: PDF Chatbot

This project is a desktop chatbot application that answers questions based on the content of multiple PDF documents. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide context-aware responses and maintains a conversational history.

  <!-- TODO: Add a screenshot of your application -->

## Features

- **PDF Document Processing**: Loads and processes text from multiple PDF files.
- **Vector Store**: Creates a local FAISS vector store for efficient similarity searches.
- **Conversational Memory**: Remembers the context of the current conversation to answer follow-up questions.
- **RAG Pipeline**: Uses a history-aware retriever to find relevant document chunks before generating an answer.
- **Fast LLM Inference**: Powered by the Groq API for near-instantaneous language model responses.
- **Simple GUI**: A clean and simple user interface built with Python's native Tkinter library.
- **Asynchronous Responses**: The GUI remains responsive while the bot is "thinking" thanks to threading.

## Tech Stack

- **Python 3**
- **LangChain**: The core framework for building the RAG application.
- **Groq**: Provides fast LLM inference (`llama-3.1-8b-instant`).
- **HuggingFace Embeddings**: For generating vector embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
- **FAISS**: For creating and managing the local vector store.
- **Tkinter**: For the graphical user interface.

---

## Setup and Installation

Follow these steps to get the chatbot running on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/mathuryashash/ai-mini-project.git
cd ai-mini-project
```

### 2. Create a Python Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.
```bash
# For Windows
python -m venv .venv
.\.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
First, you should create a `requirements.txt` file. Then, install the required packages.
```bash
# It's good practice to create this file to lock your dependencies
pip freeze > requirements.txt

# Install from the requirements file
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
You will need a Groq API key.

1.  Create a file named `.env` in the root of the project directory.
2.  Add your Groq API key to it:
    ```
    GROQ_API_KEY="gsk_YourSecretGroqApiKey"
    ```

### 5. Add Your Documents
Place the PDF files you want the chatbot to read into the `books/` directory. You can update the list of files to be loaded in `config.py`.

---

## How to Run

Once the setup is complete, run the application with the following command:

```bash
python test.py
```

On the first run, the application will process the PDFs and create a `faiss_index/` directory to store the vector index. Subsequent runs will be faster as they will load the pre-existing index.