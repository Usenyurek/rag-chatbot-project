import streamlit as st
import os
import tempfile
import shutil
from langchain_core.globals import set_debug
from service import pdf_to_vectorstore, ask_pdf

set_debug(True)

# 1. SAYFA AYARLARI
st.set_page_config(page_title="RAG Asistanı", page_icon="🤖", layout="wide")
st.title("🤖 PDF RAG Chatbot")
st.markdown("---")

# 2. SESSION STATE (Hafıza)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. YAN MENÜ - DOSYA YÜKLEME
with st.sidebar:
    st.header("📂 Döküman Yönetimi")
    uploaded_file = st.file_uploader("PDF Dosyanızı Yükleyin", type="pdf")

    if uploaded_file:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name


        if st.button("Analiz Et ve Veritabanına Ekle"):
            with st.spinner("PDF Analiz ediliyor... Bu işlem biraz sürebilir."):
                try:

                    pdf_to_vectorstore(tmp_path)
                    st.success("✅ PDF başarıyla vektörleştirildi! Artık soru sorabilirsin.")
                except Exception as e:
                    st.error(f"Hata oluştu: {e}")
                finally:

                    os.remove(tmp_path)
    st.markdown("---")

    if st.button("🧹 Sistemi Sıfırla"):
        # Dosyaların var olup olmadığını kontrol et
        db_exists = os.path.exists("./chroma_db")
        pkl_exists = os.path.exists("bm25_index.pkl")

        # Eğer ikisinden biri bile varsa silme işlemini başlat
        if db_exists or pkl_exists:
            try:
                # 1. ChromaDB Klasörünü Sil
                if db_exists:
                    shutil.rmtree("./chroma_db")

                # 2. BM25 Pickle Dosyasını Sil
                if pkl_exists:
                    os.remove("bm25_index.pkl")

                st.success("Hafıza başarıyla silindi!")

                # 3. Sohbet Geçmişini Sıfırla
                st.session_state.messages = []

                # 4. Sayfayı Yenile (Bu komuttan sonraki kodlar çalışmaz)
                st.rerun()

            except PermissionError:
                st.error("Windows dosyayı kullanıyor. Lütfen uygulamayı durdurup (Ctrl+C) tekrar başlatın.")
            except Exception as e:
                st.error(f"Silinirken hata oluştu: {e}")

        # Eğer ikisi de zaten yoksa
        else:
            st.info("Zaten hafıza boş.")


# 4. SOHBET GEÇMİŞİNİ GÖSTER
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. SORU SORMA KISMI
if prompt := st.chat_input("Döküman hakkında sorunuzu yazın..."):
    # Kullanıcı mesajını ekrana bas
    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistanın cevabını ekrana bas
    with st.chat_message("assistant"):
        # ask_pdf artık bir metin (string) değil, bir veri akışı (stream) döndürüyor
        stream_generator = ask_pdf(prompt)

        # st.write_stream, gelen parçaları anında ekrana daktilo gibi yazar
        # ve en sonunda tam metni (full_response) bize geri verir
        full_response = st.write_stream(stream_generator)

    # Tamamlanan cevabı sohbet geçmişine (session_state) kaydet
    st.session_state.messages.append({"role": "assistant", "content": full_response})