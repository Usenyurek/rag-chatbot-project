# 🤖 Local RAG Chatbot: Llama 3 & Streamlit
This project is a privacy-focused AI assistant that runs entirely on a local machine, utilizing the Retrieval-Augmented Generation (RAG) architecture. It analyzes uploaded PDF documents and answers user questions based strictly on the provided context, without requiring an internet connection or external APIs.

The project is built upon the modern LCEL (LangChain Expression Language) architecture, completely abandoning legacy chain structures for better control and stability.

## 🏗️ Architecture & Tech Stack
The core logic resides in service.py, which implements a RAG pipeline using a modern, component-based approach.

Technologies Used
LLM: Llama-3 (8B) running on Ollama. (Configured with Temperature=0 to minimize hallucinations).

Orchestration: LangChain (Utilizing LCEL architecture).

Vector Database: ChromaDB (Local persistence via persist_directory).

Embedding Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Optimized for multilingual semantic search).

Frontend: Streamlit (Includes Session State management for chat history).

## 🚀 Challenges & Engineering Solutions
During development, several technical challenges were encountered. Below are the engineering solutions implemented in service.py to overcome them:

1. Legacy Chains & Deprecation Issues
Problem: Rapid updates in LangChain versions and package instability caused the classic RetrievalQA and langchain.chains modules to become deprecated and unreliable (e.g., missing imports).
Solution: Abandoned the "Black Box" legacy chain approach. Implemented a fully transparent LCEL (LangChain Expression Language) pipeline. This allows for granular control over how context is formatted and passed to the LLM.

Python
## Legacy (Deprecated) Approach:
qa_chain = RetrievalQA.from_chain_type(...)

## Modern LCEL Approach:
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
2. Hallucinations & "Context" Fixation
Problem: Even when answering correctly, the model often included redundant prefixes like "According to the context..." or "Based on the text...". Additionally, it sometimes struggled with negative constraints in prompts.
Solution:

Prompt Engineering: Designed a strict "Expert Researcher" persona.

Post-Processing (Surgical Intervention): Implemented Python-side string manipulation (.replace() and .strip()) to programmatically remove unwanted phrases before rendering the response to the UI.

3. Windows File Locking (PermissionError)
Problem: When implementing the "Reset System" feature in Streamlit, Windows file locking mechanisms prevented chroma_db from being deleted, causing the application to crash.
Solution: Utilized the shutil library with rmtree and wrapped the logic in robust try-except blocks to handle permissions gracefully. Added automated cleanup for temporary files (tempfile) to prevent storage bloat.

## 📂 Project Structure
Bash
├── app.py              # Frontend & Session Management (Streamlit)
├── service.py          # Backend Logic, RAG Pipeline & Vector Ops
├── requirements.txt    # Project dependencies
├── .gitignore          # Excludes ChromaDB & Virtual Env
└── chroma_db/          # Vector Database (Auto-generated, ignored by git)

## 🛠️ Installation & Setup
Clone the Repository:

Bash
git clone https://github.com/Usenyurek/rag-chatbot-project.git
cd rag-chatbot-project

Create a Virtual Environment:
Bash
python -m venv .venv
# For Windows:
.venv\Scripts\activate
# For Mac/Linux:
source .venv/bin/activate
Install Dependencies:

Bash
pip install -r requirements.txt
Setup Ollama & Llama-3:

Download and install Ollama.

Pull the model via terminal: ollama pull llama3

Run the Application:

Bash
streamlit run app.py
## 🔮 Future Improvements
[ ] Implement streaming responses (Typewriter effect).

[ ] Add source citations (Page numbers/metadata).

[ ] Support for additional file formats (.docx, .txt).
