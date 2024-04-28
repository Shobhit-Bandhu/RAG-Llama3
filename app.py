from flask import Flask, request
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate



# Create the Flask app

app = Flask(__name__)

cached_llm = Ollama(model="llama3")

# <s>[INST] You are a technical assistant good at searching documents. Answer the questions that follow. If you do not have the relevant information then say so. [/INST] </s>
raw_prompt = PromptTemplate.from_template("""
    [INST] {input} 
        Context: {context}
        Answer:
    [/INST]
""")

embeddings = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False
)

@app.route("/ai", methods=["POST"])
def query_post():
    print("Post /ai called")

    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = cached_llm.invoke(f"{query}")

    response_answer = {
        "Query":query,
        "Response":response
    }
    return response_answer

@app.route("/ask_pdf", methods=["POST"])
def ask_pdf_post():
    print("Post /ask_pdf called")

    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(
        persist_directory="db",
        embedding_function=FastEmbedEmbeddings()
    )

    print("Creating retriever")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1
        }
    )

    print("Creating document chain")
    document_chain = create_stuff_documents_chain(
        cached_llm,
        raw_prompt
    )

    print("Creating chain")
    chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    print("Generating response using chain.invoke")
    response = chain.invoke({"input":query})

    print(response)

    sources = []
    for doc in response["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = {
        "Query":query,
        "Response":response["answer"],
        # "Sources": sources
    }
    return response_answer

@app.route("/pdf", methods=["POST"])
def pdf_post():
    if request.content_type.startswith('multipart/form-data'):
        pdf_file = request.files["file"]
        file_name = pdf_file.filename
        save_file = "pdf/"+file_name

        pdf_file.save(save_file)
        print(f"filename: {file_name}")

        loader = PDFPlumberLoader(save_file)
        docs = loader.load_and_split()
        print (f"docs len={len(docs)}")

        chunks = text_splitter.split_documents(docs)
        print(f"chunks len={len(chunks)}")

        vector_store = Chroma.from_documents(
            documents = chunks,
            embedding=FastEmbedEmbeddings(),
            persist_directory = "db"
        )

        vector_store.persist()

        response = {
            "status": "Successfully uploaded",
            "filename": file_name,
            "document_length": len(docs),
            "chunk_length": len(chunks)
        }

        return response
    
    return f"Error, filetype is {request.content_type}", 415

if __name__ == "__main__":
    app.run(
        host="localhost", 
        port=8000, 
        debug=True
    )