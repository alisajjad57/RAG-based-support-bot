import os
import streamlit as st
import tempfile
from langchain_helper import create_embeddings_n_store_in_db
from langchain_helper import get_similarities_n_call_llm
from langchain_core.messages import AIMessage, HumanMessage
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import csv_loader, UnstructuredURLLoader

#########################################################################
# Main Section
#########################################################################
st.title("🤖 Support Bot")

# User input
user_query = st.chat_input("Type your message here...")

# Placeholder for chat display
main_placeholder = st.container()

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if user_query:
    response = get_similarities_n_call_llm(user_query)
    
    # Append user + AI messages to history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

# Display full chat history
with main_placeholder:
    for message in st.session_state.chat_history:
        role = "You" if isinstance(message, HumanMessage) else "AI"
        st.markdown(f"**{role}:** {message.content}")


#########################################################################
# Sidebar Form
#########################################################################
st.sidebar.title("🛠️ Admin Dashboard")
with st.sidebar.form(key="admin_form"):
    # Display File Upload section
    st.subheader("Add CSV or PDF files 📁")
    uploaded_files = st.file_uploader("Upload CSV or PDF", type=["csv", "pdf"], accept_multiple_files=True)

    # Display URL input fields
    st.subheader("🌐 Add website URLs")
    urls = []
    for i in range(3):
        url = st.text_input(f"🔗 URL {i+1}")
        urls.append(url)
    
    # Process button
    process_button = st.form_submit_button(label="🚀 Process Data")




#########################################################################
# Process Data
#########################################################################
document_data = []
if process_button:
    with st.spinner("⏳ Processing your data... please wait..."):

        if uploaded_files:
            st.sidebar.markdown("Uploaded Files are:")
            for file in uploaded_files:
                st.sidebar.write(f"📝 {file.name}")
                if file.type == "text/csv":
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                        temp_file.write(file.getvalue())
                        temp_file_path = temp_file.name

                    csv_load = csv_loader.CSVLoader(file_path=temp_file_path, source_column="question")
                    csv_docs = csv_load.load()
                    document_data.extend(csv_docs)

                elif file.type == "application/pdf":

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(file.getvalue())
                        temp_file_path = temp_file.name
                    pdf_loader = PyPDFLoader(temp_file_path)
                    pdf_docs = pdf_loader.load()
                    document_data.extend(pdf_docs)

                else:
                    st.sidebar.write("📝 File type: Unknown")
        else:
            st.sidebar.write("No 📝 Files Uploaded.")

        if any(url.strip() for url in urls):
            st.sidebar.markdown("Entered URLs are:")
            for url in urls:
                if url.strip():
                    st.sidebar.write(f"🔗 {url}")
                    url_loader = UnstructuredURLLoader(urls=urls)
                    url_docs = url_loader.load()
                    document_data.extend(url_docs)
        else:
            st.sidebar.write("No 🔗 URLs Entered.")

        create_embeddings_n_store_in_db(document_data)
        st.sidebar.success("✅ Processing Completed")
