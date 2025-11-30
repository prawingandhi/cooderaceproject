import os
from dotenv import load_dotenv
load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")
#os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

import streamlit as st
from langchain_groq import ChatGroq as groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

#groq_api_key = st.secrets.get("GROQ_API_KEY") | os.getenv("GROQ_API_KEY")

"""if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in Streamlit Secrets or .env.")
else:
    st.success("✅ Groq API key loaded successfully")"""

st.title("Wellcome to Q/A in Tamil for CodeRace 🇮🇳")

file = st.file_uploader("Select the file  : ", type=["pdf"])
if file is not None:
        temp_path = "tem_file.pdf"
        with open(temp_path, "wb") as f:
             f.write(file.read())

        loader = PyPDFLoader(temp_path)
        doc = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        doc_splits = text_splitter.split_documents(doc)


        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


        vectorstore = Chroma.from_documents(doc_splits, embedding)
        retriver = vectorstore.as_retriever(search_type="similarity")


        llm = groq(api_key=groq_api_key,model="mixtral-8x7b-32768")

        prompt_template = ChatPromptTemplate.from_template("""
        You are a  assistant. 
        Your tasks are:

        You are a helper. 
        1. Use the context given below to give the correct answer to the question.
        2. if contex have any poem or songs give it along with that                                                   
        3. Always give the answer in Tamil.
        4. if the question is in english , translate the question to Tamil and then answer it in tamil.                                                   
        4. if the question is not answerable from the context, reply with "The answer is not available in the provided document."                                                   


        Question: {question}
        Context (Tamil): {context}
        Answer in Tamil:
        """
        )
        # Build the chain
        doc_chain = (
            {"context": retriver, "question": RunnablePassthrough()}
            | prompt_template
            | llm
        )


        question = st.text_input("உங்கள் கேள்வியை உள்ளிடவும்:")

        #answer = doc_chain.invoke({"question":" வைகாசி பிரம்மோற்ஸவத்தின் போது சௌரிராஜ பெருமாள் எந்த மூர்த்திகளாக, எந்த நேரங்களில் காட்சி தருகிறார்?"})
        if question:
                answer = doc_chain.invoke(question)
                st.subheader("பதில்:")
                st.write(answer)

else:
        st.write("Please enter a valid file .")             