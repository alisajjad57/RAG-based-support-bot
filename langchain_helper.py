from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
from dotenv import load_dotenv
load_dotenv()

# from langchain.globals import set_debug
# set_debug(True)


def create_embeddings_n_store_in_db(document_data):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=800
    )
    document_chunks = text_splitter.split_documents(document_data)
    print(f"Loaded {len(document_chunks)} chunks")

    document_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    chroma_db = Chroma.from_documents(
        documents=document_chunks,
        embedding=document_embeddings,
        persist_directory="chroma_db"
    )
    chroma_db.persist()


def get_similarities_from_chroma_db_test(question):
    document_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
    )

    chroma_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=document_embeddings
    )

    retriever = chroma_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )
    related_docs = retriever.invoke(question)
    return related_docs
    


def get_similarities_n_call_llm(question):
    groq_api_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        groq_api_key = groq_api_key,
        model = "llama-3.1-8b-instant",
        temperature=0.2,
    )

    template = """
    You are an expert assistant. Answer the following question using ONLY the context below.
    - If the answer is not found in the context, respond strictly with: "The answer is not available in the database."
    - Do NOT use any outside knowledge. Stay strictly within the given context.
    - Be concise and factual.

    Context: {context}

    Question: {input}
    """
    prompt = ChatPromptTemplate.from_template(template)

    document_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
    )

    chroma_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=document_embeddings
    )
    retriever = chroma_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)

    # Execute the chain with your question
    response = retrieval_chain.invoke({"input": question})
    answer = response.get('answer')
    
    return answer



if __name__ == "__main__":
    question = "Add Test Question"
    # question = "can GPUs help me in online shopping?"
    # question = "can i use numba with jupyter notbook?"
    # question = "what was the purpose of award to tgen team?"
    related_docs = get_similarities_from_chroma_db_test(question)
    print("Related docs:")
    for doc in related_docs:
        print(doc.page_content)
        print("---------------")
