# Code Basics Resume project challenge : Internal chatbot with role based access control
 [Resume Project Challenge](https://codebasics.io/challenge/codebasics-gen-ai-data-science-resume-project-challenge) of building a RAG based Internal Chatbot with role based access control. 

 ### Roles Provided
 - **engineering**
 - **finance**
 - **general**
 - **hr**
 - **marketing**

# Implementation details:
- Basic Authentication using FastAPI's `HTTPBasic`
- User interface  - `Streamlit`
- core logic for retrieval - `Lang Graph`
- vector store : `Chorma`
- Generative Ai model : Gemini 2.0 flash
- EMbedding model provider : Gemini

## Architecture of the graph:
![image](https://github.com/user-attachments/assets/c3e56dca-980f-4208-94b7-961fd250fe8b)



## key concepts covered :
1. RAG : to retrive relevant chunks for a given user query from the knowledge base.
2. Metadata : Metadata plays a main role in this chatbot. for every query being asked, metadata search will be leveraged to extract only relevant chunks from the role of the logged in user.
for eg : a user from general or employee department cannot have access to the HR data.
Metadata search is the core part of the Role Based access.
3. Additional Access check : LLM Acts as an additonal access check assitant. let say a user with employee department trying to query HR data, llm identifies the department of question being asked , and checks the role of logged in user. if both departments are same then it will continue normal execution. 
If both didn't match directly ends the flow.
4. To improve the relevance of extracted documents , added one more extra node `grade documnets ` check the relevanve os extracted context to the given question. if documents are not relevant, LLM tries to reframe the query moo=re semantically , to get the relevant information availble from the knowledge store.
5. Reursion limit handled : In graph loop exists between rewrite query and extract  documents, this loop will be executed for 3 times.
   











