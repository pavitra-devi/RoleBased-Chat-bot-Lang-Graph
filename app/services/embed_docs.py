from langchain_community.document_loaders import DirectoryLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain import hub
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store_new")
print("Vector Store Path:", VECTOR_STORE_PATH)

os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY","AIzaSyCZt0Vrqv69IfvgzPId4WB_KpnIUIf8fJk")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
#defining the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# Define role mappings (from metadata)
ROLE_MAPPING = {
    "engineering": {"allowed_roles": ["engineering", "c-level"]},
    "finance": {"allowed_roles": ["finance", "c-level"]},
    "marketing": {"allowed_roles": ["marketing", "c-level"]},
    "hr": {"allowed_roles": ["hr", "c-level"]},
    "general": {"allowed_roles": ["general", "c-level"]},
}

# Load documents from subfolders and attach metadata
def load_and_tag_documents():
    documents = []
    for department in ["engineering", "finance", "hr", "marketing", "general"]:
        folder_path = f"resources/data/{department}/"
        if not os.path.exists(folder_path):
            print(f"Warning: Directory not found: {folder_path}")
            continue
        # Load .md files
        md_loader = DirectoryLoader(folder_path, glob="**/*.md")
        md_docs = md_loader.load()
        for doc in md_docs:
            doc.metadata = {
                "allowed_roles": ",".join(ROLE_MAPPING[department]["allowed_roles"]),
                "file_path": doc.metadata.get("source", doc.metadata.get("file_path", "")) or getattr(doc, "source", ""),
                "filename": os.path.basename(getattr(doc, "source", doc.metadata.get("source", "")))
            }
        documents.extend(md_docs)
        # Load .csv files (if any)
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                csv_path = os.path.join(folder_path, file)
                csv_loader = CSVLoader(csv_path)
                csv_docs = csv_loader.load()
                for doc in csv_docs:
                    doc.metadata = {
                        "allowed_roles": ",".join(ROLE_MAPPING[department]["allowed_roles"]),
                        "file_path": csv_path,
                        "filename": os.path.basename(csv_path)
                    }
                documents.extend(csv_docs)
    return documents


def embed_documents():
    # Split and embed
    documents = load_and_tag_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # Assign page numbers per original document
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_documents([doc])
        for idx, chunk in enumerate(doc_chunks, start=1):
            chunk.metadata['page_number'] = idx
        chunks.extend(doc_chunks)
    # Store in Chroma with metadata
    vector_store = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )


#test with a sample query
def query_vector_store(query: str, user_role: str):
    # Check if user role is allowed
    allowed_roles = ROLE_MAPPING.get(user_role, {}).get("allowed_roles", [])
    if not allowed_roles:
        return {"error": "Invalid user role or no access to documents."}
    # Load the vector store
    vector_store = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embeddings
    )
    print("Number of docs:", len(vector_store.get()["documents"]))
    # Perform the query
    results = vector_store.similarity_search_with_score(
        query, k=5, filter={"allowed_roles": ",".join(allowed_roles)}
    )
    print("fecthed results:",results)
    # Format results
    # formatted_results = [
    #     {"content": doc.page_content, "score": score} for doc, score in results
    # ]
    
    return results

def generate_response(query: str, user_role: str):
    print(f"Query: {query}, User Role: {user_role}")
    results = query_vector_store(query, user_role)
    # Check if user role is allowed
    print(results)
    prompt = hub.pull("rlm/rag-prompt")
    
    # prompt='''You are a helpful assistant. You will be provided with a question and some context from documents. 
    # Your task is to answer the {question} based on the {context} provided.'''
    docs_content = "\n\n".join(doc.page_content for doc,score in results)
    print("Docs Content:", docs_content)
    messages = prompt.invoke({"question":query, "context": docs_content})
    print("Messages:", messages)
    response = model.invoke(messages)
    print("Response:", response.content)
    return response.content
    

# # Example usage
# if __name__ == "__main__":
#     # embed_documents()  # Ensure documents are embedded before querying
#     user_role = "engineering"  # Example user role
#     # query = "give me a glimpse of Comprehensive Marketing Report - Q4 2024"
#     query ="give me a list of all marketing managers"
#     results = query_vector_store(query, user_role)
#     # for result in results:
#     #     print(f"Content: {result['content']}\nScore: {result['score']}\n")
