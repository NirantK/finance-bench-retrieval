from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, StateGraph

from src.search_tools import hybrid_search


# Define the state
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "The messages in the conversation"]
    search_results: Annotated[list[str], "The search results from hybrid search"]


# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
)


# Define the search node
def search_node(state: AgentState) -> AgentState:
    """Node that performs hybrid search."""
    # Get the last human message
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        raise ValueError("Last message must be from human")

    # Perform hybrid search
    search_results = hybrid_search(last_message.content, top_k=3)

    # Update state with search results
    return {"messages": state["messages"], "search_results": search_results}


# Define the answer generation node
def answer_node(state: AgentState) -> AgentState:
    """Node that generates an answer based on search results."""
    # Get the last human message
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        raise ValueError("Last message must be from human")

    # Create context from search results
    context = "\n\n".join(state["search_results"])

    # Create system message with context
    system_message = f"""You are a helpful AI assistant. Use the following context to answer the user's question. 
    If you cannot find the answer in the context, say so.
    
    Context:
    {context}
    """

    # Generate response
    response = llm.invoke(
        [{"role": "system", "content": system_message}, {"role": "user", "content": last_message.content}]
    )

    # Add AI response to messages
    new_messages = list(state["messages"]) + [AIMessage(content=response.content)]

    return {"messages": new_messages, "search_results": state["search_results"]}


# Create the graph
def create_rag_graph() -> Graph:
    """Create the RAG graph."""
    # Create a new graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("answer", answer_node)

    # Add edges
    workflow.add_edge("search", "answer")

    # Set entry point
    workflow.set_entry_point("search")

    # Set exit point
    workflow.set_finish_point("answer")

    return workflow.compile()


# Create the graph instance
rag_graph = create_rag_graph()


def rag_chain(query: str) -> str:
    """Run the RAG chain on a query."""
    # Initialize state
    initial_state = {"messages": [HumanMessage(content=query)], "search_results": []}

    # Run the graph
    result = rag_graph.invoke(initial_state)

    # Return the last AI message
    return result["messages"][-1].content


if __name__ == "__main__":
    # Example usage
    query = "What are the key financial metrics for Q4 2023?"
    response = rag_chain(query)
    print(f"Query: {query}")
    print(f"Response: {response}")
