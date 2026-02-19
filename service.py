from re import search
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import pickle



def pdf_to_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    splits = text_splitter.split_documents(docs)

    # ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # BM25 (TF-IDF based)
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5

    # Saving the BM25 Index
    with open("bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)

    return vectorstore




def ask_pdf(question):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    if not os.path.exists("./chroma_db"):
        return "Lütfen önce bir PDF dosyası yükleyin."
    # ChromaDB Initialisation
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    #retrieve aşamasını güçlendirmek için: retriever = BM25Retriever.from_texts(["foo", "bar", "world", "hello", "foo bar"])
    vector_retriever = db.as_retriever(search_kwargs={"k": 5})

    # BM25
    with open("bm25_index.pkl", "rb") as f:
        bm25_retriever = pickle.load(f)

    # Ensemble Retriever (Hybrid Search)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    template_str = """
    Sen bir bilgi çıkarım sistemisin. Yalnızca aşağıdaki bağlamı kullanarak soruyu yanıtla.
    
    KESİN KURALLAR:
    1. YANITIN TAMAMI TÜRKÇE OLMALIDIR. İngilizce kelime veya "According to the text" gibi ifadeler kullanmak yasaktır.
    2. Soru birden fazla bağlantı içeriyorsa (Örn: X'in Y'sinin Z'si), bağlam içinde sırasıyla ilerleyerek çıkarım yap.
    3. Cevap bağlamda doğrudan yer almıyorsa, kendi veritabanındaki bilgileri ASLA kullanma ve sadece "Bu bilgi dökümanda bulunmamaktadır." çıktısını ver.
    4. Yanıta doğrudan bilgi ile başla, giriş cümlesi kullanma.
    
    Bağlam:
    {context}
    
    ---
    
    Soru: {question}
    
    Türkçe Yanıt:
    """
    prompt = ChatPromptTemplate.from_template(template_str)

    llm = ChatOllama(model="llama3", temperature=0)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    #pipeline
    rag_chain = (
        {"context": vector_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)









