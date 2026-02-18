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



def pdf_to_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

def ask_pdf(question):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    if not os.path.exists("./chroma_db"):
        return "Lütfen önce bir PDF dosyası yükleyin."

    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 5})

    template_str = """
    Sen lafı uzatmayan, net ve robotik bir asistansın.
    
    GÖREVİN:
    Aşağıdaki bağlamı oku ve soruya cevap ver.
    
    KESİN KURALLAR:
    1. Cevaba DOĞRUDAN bilgi ile başla.
    2. Asla giriş cümlesi kullanma (Örn: "Bağlama göre", "Metinde yazdığı üzere" vb. YASAK).
    3. Sanki bu bilgiyi ezbere biliyormuşsun gibi konuş. Referans verme.
    
    Bağlam:
    {context}
    
    Soru:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template_str)

    llm = ChatOllama(model="llama3", temperature=0)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    #pipeline
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)









