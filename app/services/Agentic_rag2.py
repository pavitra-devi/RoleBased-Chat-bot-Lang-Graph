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
from typing import TypedDict, Any
from langchain import hub
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, add_messages

# Set up paths and environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store_new")
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


class AgentState(TypedDict):
    user_role: str
    question: str
    response: str
    context: List[Document]
    iteration: int
    has_access: str
    final_response: str

def user_has_access(state:AgentState):
    """
    Check if the user has access to the requested information based on their role.
    Returns the string name of the next node: 'retrieve_documents' or END.
    """
    user_role = state['user_role']
    question = state['question']
    access_prompt = (
       f"You are an access control assistant.\n"
        f"The logged-in user's department/role is: {user_role}\n"
        f"Here is the user question: {question}\n"
        f"Identify the department the question belongs.\n"
        f"Refer the below role mapping for  access control:\n"
        f"Finance Team: Access to financial reports, marketing expenses, equipment costs, reimbursements, etc.\n"
        f"Marketing Team: Access to campaign performance data, customer feedback, and sales metrics.\n"
        f"HR Team: Access employee data, attendance records, payroll, and performance reviews.\n"
        f"Engineering Department: Access to technical architecture, development processes, and operational guidelines.\n"
        f"C-Level Executives: Full access to all company data.\n"
        f"Employee Level: Access only to general company information such as policies, events, and FAQs.\n"
        f"If the question is about a department the user does NOT have access to, respond with 'deny'.\n"
        f"If the question is generic or the user should be allowed, respond with 'allow'.\n"
        f"Respond with only 'allow' or 'deny'"
    )
    result = model.invoke([{"role": "user", "content": access_prompt}])
    response = result.content.strip().lower()
    state['has_access'] = response
    print(f"---------Access check response: {response}-----------")
    if response == 'allow':
        return "retrieve_documents"
    elif response == 'deny':
        state['response'] = "The user does not have access to this information."
        state['final_response']="The user does not have access to this information."
        return END
    else:
        return END

def retrieve_documents(state:AgentState) -> List[Document]:
    """
    Retrieve documents based on user role and question.
    """
    print("Inside retrieve_documents function")
    user_role=state['user_role']
    question=state['question']
    allowed_roles = ROLE_MAPPING.get(user_role, {}).get("allowed_roles", [])
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    )
    
    results=vector_store.similarity_search_with_relevance_scores(question, k=5,filter= {"allowed_roles": ",".join(allowed_roles)})
    return {'user_role': user_role, 'question': question, 'context': [doc for doc, _ in results], 'iteration': state['iteration']}

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

grader_model = init_chat_model("gemini-2.0-flash",model_provider="google_genai",temperature=0)

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

def grade_documents(state: AgentState,) -> Literal["generate_answer", "rewrite_question"]:
    
    """Determine whether the retrieved documents are relevant to the question."""
    print("Inside grade documents function")
    question = state['question']
    context = state["context"]

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score
    print(f"[DEBUG] Score in grade_documents: {score}")
    if score == "yes":
        return "generate_answer"
    
    else:
        iteration = state['iteration']
        print(f"[DEBUG] Iteration in grade_documents: {iteration}")
        return "rewrite_question" if iteration < MAX_ITERATIONS else "generate_answer"
        
REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state: AgentState):
    """Rewrite the original user question and increment iteration."""
    print("Inside rewrite_question function")
    question= state['question']
    context = state['context']
    
    iteration = state['iteration'] + 1  # Always increment
    print(f"[DEBUG] Iteration in rewrite_question: {iteration}")
    prompt = REWRITE_PROMPT.format(question=question)
    response = model.invoke([{"role": "user", "content": prompt}])
    state['question'] = response.content.strip()
    state['iteration'] = iteration
    print(f"[DEBUG] New question after rewrite: {state['question']}")
    return {
       "user_role": state['user_role'],
        "question": state['question'],  "context": context,
        "iteration": iteration,
    }   


GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use five sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: AgentState):
    """Generate an answer."""
    print("Inside generate_answer function")
    question = state["question"]
    context = state["context"]
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = model.invoke([{"role": "user", "content": prompt}])
    final_response = response.content.strip()
    state['response'] = final_response
    return {"question":state['question'],"context": state['context'], "response": response.content.strip(), "iteration": state['iteration'],state['final_response']: final_response}  # Increment iteration

def build_graph():
    workflow = StateGraph(AgentState)
    # Add nodes
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)
    # Conditional edge from START: user_has_access decides if we go to retrieve_documents or END
    workflow.add_conditional_edges(
        START,
        user_has_access,
        {
            "retrieve_documents": "retrieve_documents",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "retrieve_documents",
        grade_documents,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question"
        }
    )
    workflow.add_edge("rewrite_question", "retrieve_documents")
    workflow.add_edge("generate_answer", END)
    return workflow
graph=build_graph().compile()

# # Example usage:
# if __name__ == "__main__":
#     graph = build_graph().compile()
#     # Example state
#     state = {
#         "user_role": "engineering",
#         "question": "Explain me about How the FAST API can be leveraged?",
#         "iteration": 0,
#         "context": [],
#         "response": "",
#         "has_access": ""
#     }
#     for chunk in graph.stream(state):
#         for node, update in chunk.items():
#             print(f"Node: {node}, Output: {update}")

# # Example usage:
# if __name__ == "__main__":
#     graph = build_graph().compile()
#     # Example state
#     state = {
#         "user_role": "hr",
#         "question": "give me the details of  Aadhya , leave balance working dept",
#         "iteration": 0,
#         "context": [],
#         "response": "",
#         "has_access": False,
#         "final_response": ""
       
#     }
#     final_response = None
#     for chunk in graph.stream(state):
#         for node, update in chunk.items():
#             print(f"Node: {node}, Output: {update}")
#             if 'response' in update:
#                 final_response = update['response']
#     if final_response is None:
#         final_response = "The user does not have access to this information."    
#     print("Final Response:", final_response)