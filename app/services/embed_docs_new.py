import os
from typing import List, Tuple
from langchain_community.document_loaders import DirectoryLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState 
from pydantic import BaseModel, Field
from typing import Literal   
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from langgraph.prebuilt import tools_condition

from langchain import hub

# Set up paths and environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY", "AIzaSyCZt0Vrqv69IfvgzPId4WB_KpnIUIf8fJk")
MAX_ITERATIONS = 3

# Model and embeddings
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


ROLE_MAPPING = {
    "engineering": {"allowed_roles": ["engineering", "c-level"]},
    "finance": {"allowed_roles": ["finance", "c-level"]},
    "marketing": {"allowed_roles": ["marketing", "c-level"]},
    "hr": {"allowed_roles": ["hr", "c-level"]},
    "general": {"allowed_roles": ["general", "c-level"]},
}

#graph implementation creating retriver tool
retriever_tool = None  # Global or per-session variable

def get_retriever_for_role(user_role: str):
    global retriever_tool
    allowed_roles = ROLE_MAPPING.get(user_role, {}).get("allowed_roles", [])
    retriever= Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    ).as_retriever(search_kwargs={"k": 5, "filter": {"allowed_roles": ",".join(allowed_roles)}})
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_information",  # Valid function name
        "Search and return information about HR, Engineering, Finance, Marketing, and General documents"
    )
    return retriever_tool

def check_access(user_role: str, question: str) -> bool:
    # Simple keyword-based department check (replace with LLM for more accuracy)
    department_keywords = ["engineering", "finance", "marketing", "hr", "general"]
    for dept in department_keywords:
        if dept in question.lower():
            if dept == user_role or user_role == "c-level":
                return True
            else:
                return False
    return True  #


def llm_access_check(user_role: str, question: str) -> bool:
    """Use LLM to decide if the user has access to the department/topic in the question."""
    access_prompt = (
        f"You are an access control assistant.\n"
        f"The logged-in user's department/role is: {user_role}\n"
        f"Here is the user question: {question}\n"
        f"Decide if the user is allowed to access information for this question.\n"
        f"If the question is about a department the user does NOT have access to, respond with 'deny'.\n"
        f"If the question is generic or the user should be allowed, respond with 'allow'.\n"
        f"Respond with only 'allow' or 'deny'."
    )
    result = model.invoke([{"role": "user", "content": access_prompt}])
    return result.content.strip().lower() == "allow"


def generate_query_or_respond(state: MessagesState):
    user_role = state.get("user_role", "")
    question = state["messages"][0].content
    # LLM-based access check
    # if not llm_access_check(user_role, question):
    #     return {"messages": [{"role": "system", "content": "Access denied: You do not have access to this department's information."}]}
    response = (
        model
        .bind_tools([retriever_tool]).invoke(state["messages"])
    )
    return {"messages": [response]} 



# GRADE_PROMPT = (
#     "You are a grader for a role-based access system.\n"
#     "The logged-in user's department is: {user_department}\n"
#     "Here is the user question: {question}\n"
#     "Your task: Decide if the question is about the user's own department.\n"
#     "If the question is about the user's department, respond with 'yes'.\n"
#     "If the question is about a different department, respond with 'no'.\n"
#     "Give a binary score 'yes' or 'no' to indicate whether the question is about the user's department."
# )

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

grader_model = init_chat_model("gemini-2.0-flash",model_provider="google_genai",temperature=0)

def should_continue(state: MessagesState):
    # Default to 0 if not present
    iteration = state.get('iteration', 0)
    if iteration >= MAX_ITERATIONS:
        return "generate_answer"
    return "generate_query_or_respond"

def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"
    

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    iteration = state.get('iteration', 0)
    iteration += 1
    state["iteration"] = iteration 
    prompt = REWRITE_PROMPT.format(question=question)
    response = model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}],"iteration": iteration}


GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use five sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# display(Image(graph.get_graph().draw_mermaid_png()))

# When initializing the graph, pass user_role and retriever_tool as part of the state
def run_graph_with_user_role(user_role: str, user_question: str):
    workflow = StateGraph(MessagesState)
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)
    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END},
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )
    workflow.add_edge("generate_answer", END)
    # workflow.add_edge("rewrite_question", "generate_query_or_respond")
    workflow.add_conditional_edges(
        "rewrite_question",
        should_continue,
        {
            "generate_query_or_respond": "generate_query_or_respond",
            "generate_answer": "generate_answer"
        }
    )
    graph = workflow.compile()
    final_message = None
    for chunk in graph.stream({
        "messages": [
            {"role": "user", "content": user_question}
        ],
        "user_role": user_role,
        "retriever_tool": retriever_tool,
        "iteration": 0 
    }):
        for node, update in chunk.items():
            print("Update from node", node)
            print(update["messages"][-1])
            print("\n\n")
            final_message = update["messages"][-1]
    # Return the content of the last message
    if isinstance(final_message, dict) and "content" in final_message:
        return final_message["content"]
    return final_message


user_role = "engineering"  # Example user role, can be changed to test different roles
get_retriever_for_role(user_role)
# Correct usage: pass the query string as the first argument
# retriever_tool.invoke("What are the latest HR policies?")

response=run_graph_with_user_role(user_role, "What are the latest HR policies")
print("Final Response:", response)
