import os  # Access environment variables

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import (HuggingFaceInstructEmbeddings,
                                  OpenAIEmbeddings)
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader

from htmlTemplates import bot_template, css, user_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    EMBEDDINGS_OPENAI = int(os.getenv("EMBEDDINGS_OPENAI"))
    if bool(EMBEDDINGS_OPENAI) :
        # 1.) Paid - cloud version (OpenAI) for creating of embeddings / vector store
        st.write("-> OpenAI Embeddings")
        embeddings = OpenAIEmbeddings()
    else:
        # 2.) Free - local version (Instructor) for creating of embeddings / vector store
        # This operation takes too long time to complete!
        st.write("-> HuggingFace Embeddings")
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDF", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # If application re-runs itself then check if "conversation" is already
    # in the state the session state of the application. Set it "None" if it's
    # not been initialized otherwise do nothing. So now the "conversation" is 
    # able to use anywhere in during the application (not only inside of 
    # the st.sidebar block).
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask question about your documents:")

    st.write(user_template.replace("{{MSG}}", "HELLO BOT"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "HELLO HUMAN"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs and click on Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get PDF text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)
                
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                st.write(vectorstore)

                # create conversation chain
                # st.session_state. - means that will be not reinitialized after an button is pressed or page is reloaded
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()



