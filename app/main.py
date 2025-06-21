from typing import Dict

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from app.services.embed_docs import generate_response
from app.services.Agentic_rag2 import graph


app = FastAPI()
security = HTTPBasic()

# Dummy user database
users_db: Dict[str, Dict[str, str]] = {
    "Tony": {"password": "password123", "role": "engineering"},
    "Bruce": {"password": "securepass", "role": "marketing"},
    "Sam": {"password": "financepass", "role": "finance"},
    "Peter": {"password": "pete123", "role": "engineering"},
    "Sid": {"password": "sidpass123", "role": "marketing"},
    "Natasha": {"password": "hrpass123", "role": "hr"}
}


# Authentication dependency
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    user = users_db.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": username, "role": user["role"]}


# Login endpoint
@app.get("/login")
def login(user=Depends(authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}


# Protected test endpoint
@app.get("/test")
def test(user=Depends(authenticate)):
    return {"message": f"Hello {user['username']}! You can now chat.", "role": user["role"]}


# Protected chat endpoint
@app.post("/chat")
def query(user=Depends(authenticate), message: str = "Hello"):
    username=user['username']
    role=user['role']
    role = role.lower()
    state={
        'question': message,
        'user_role': role,
        "iteration": 0,
        "has_access":"",
        "context": [],
        "response": "",
        "final_response": "",
    }
    result=graph.invoke(state)
    # Compose response and document references
    response = result.get("response", "User has no access to this document or question.")
    # Extract document references if present
    doc_info = []
    for doc in result.get("context", []):
        if hasattr(doc, 'metadata'):
            info = {
                "filename": doc.metadata.get('filename', ''),
                "file_path": doc.metadata.get('file_path', ''),
                "page_number": doc.metadata.get('page_number', None)
            }
            doc_info.append(info)
    print("Document Info:", doc_info)
    
    if(response is None or response == ""):
        response = f"User with role {role} has no access to this data."
    return {"response": response, "document_references": doc_info}

