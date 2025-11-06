# agents/travel_agent.py
"""
Travel planning agent using LangGraph and ReAct pattern.
Updated to follow 2025 LangChain best practices with LangGraph.
"""
from typing import Annotated, Sequence, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import logging
import json

logger = logging.getLogger(__name__)


class TravelAgentState(TypedDict):
    """State for the travel planning agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    destination: str
    num_days: int
    research_results: List[Dict[str, Any]]
    itinerary: str


class TravelAgent:
    """
    A ReAct-style travel planning agent using LangGraph.
    
    This agent can:
    - Research destinations using web search
    - Generate optimized search queries
    - Create detailed day-by-day itineraries
    - Maintain conversation state across interactions
    """
    
    def __init__(
        self,
        google_api_key: str,
        tools: List,
        model_name: str = "gemini-2.5-flash-lite",
        temperature: float = 0.3,
        system_prompt: str = None
    ):
        """
        Initialize the travel agent.
        
        Args:
            google_api_key: Google AI API key for Gemini
            tools: List of LangChain tools the agent can use
            model_name: Gemini model to use
            temperature: Model temperature (0.0-1.0)
            system_prompt: Optional custom system prompt
        """
        # Set up the LLM
        import os
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_retries=2
        )
        
        # Bind tools to the model
        self.llm_with_tools = self.llm.bind_tools(tools)
        self.tools = tools
        
        # System prompt for the agent
        self.system_prompt = system_prompt or """You are an expert travel planning assistant.
Your goal is to help users plan amazing trips by:
1. Researching destinations thoroughly using web search
2. Finding the best activities, restaurants, and accommodations
3. Creating detailed, practical day-by-day itineraries

When planning:
- Use search tools to get current, accurate information
- Consider the number of days and pace of travel
- Include specific recommendations with sources
- Provide practical details like timings and locations
- Make itineraries realistic and enjoyable

Always cite your sources and provide links when available."""

        # Build the agent graph
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for the agent."""
        # Create the graph
        workflow = StateGraph(TravelAgentState)
        
        # Create tool node
        tool_node = ToolNode(self.tools)
        
        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", tool_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        return workflow
    
    def _call_model(self, state: TravelAgentState) -> Dict:
        """Call the LLM with the current state."""
        messages = state["messages"]
        
        # Add system message if this is the first call
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self.system_prompt)] + list(messages)
        
        response = self.llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    def _should_continue(self, state: TravelAgentState) -> str:
        """Determine whether to continue with tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the LLM makes a tool call, continue to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        
        # Otherwise, end
        return "end"
    
    def invoke(
        self,
        user_input: str,
        destination: str = "",
        num_days: int = 0,
        config: Dict = None
    ) -> Dict:
        """
        Invoke the agent with a user query.
        
        Args:
            user_input: The user's message/query
            destination: Travel destination
            num_days: Number of days for the trip
            config: Optional configuration (e.g., for thread_id)
        
        Returns:
            Dictionary with the agent's response and updated state
        """
        # Prepare initial state
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "destination": destination,
            "num_days": num_days,
            "research_results": [],
            "itinerary": ""
        }
        
        # Configure with thread_id for conversation persistence
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        # Invoke the graph
        result = self.app.invoke(initial_state, config=config)
        
        return result
    
    async def astream(
        self,
        user_input: str,
        destination: str = "",
        num_days: int = 0,
        config: Dict = None
    ):
        """
        Stream responses from the agent asynchronously.
        
        Args:
            user_input: The user's message/query
            destination: Travel destination
            num_days: Number of days for the trip
            config: Optional configuration (e.g., for thread_id)
        
        Yields:
            Events from the agent execution
        """
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "destination": destination,
            "num_days": num_days,
            "research_results": [],
            "itinerary": ""
        }
        
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        async for event in self.app.astream(initial_state, config=config):
            yield event
    
    def get_last_message(self, result: Dict) -> str:
        """Extract the last AI message from the result."""
        messages = result.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message.content
        return ""


class ResearchAgent:
    """
    Specialized agent for travel research using LangGraph.
    This agent focuses on gathering and organizing travel information.
    """
    
    def __init__(
        self,
        google_api_key: str,
        search_tool,
        model_name: str = "gemini-2.5-flash-lite",
        temperature: float = 0.2
    ):
        """Initialize the research agent."""
        import os
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_retries=2
        )
        self.search_tool = search_tool
        
        self.system_prompt = """You are a travel research specialist.
Generate 3-5 specific, targeted search queries to gather comprehensive information about a destination.

Focus on:
- Top attractions and activities
- Neighborhoods and areas to visit
- Accommodation options
- Dining recommendations
- Local culture and customs

Output ONLY a JSON array of search query strings, nothing else.
Example: ["query1", "query2", "query3"]"""
    
    def generate_search_queries(self, destination: str, num_days: int) -> List[str]:
        """Generate optimized search queries for a destination."""
        prompt = f"""Generate search queries for: {destination} ({num_days} days)
Output format: JSON array of strings only."""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Try to parse JSON
            queries = json.loads(content)
            if isinstance(queries, list):
                return [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        except Exception as e:
            logger.warning(f"Failed to parse LLM output as JSON: {e}")
            # Fallback to simple queries
        
        # Fallback queries
        return [
            f"best things to do in {destination}",
            f"top neighborhoods in {destination}",
            f"best restaurants in {destination}",
            f"where to stay in {destination}",
            f"{destination} {num_days} day itinerary"
        ][:5]
    
    def search_and_aggregate(
        self,
        queries: List[str],
        max_results_per_query: int = 5
    ) -> List[Dict[str, Any]]:
        """Execute searches and aggregate results."""
        all_results = []
        seen_links = set()
        
        for query in queries:
            try:
                results = self.search_tool.invoke({"query": query, "num_results": max_results_per_query})
                
                for result in results:
                    link = result.get("link")
                    if link and link not in seen_links and "error" not in result:
                        result["query"] = query
                        all_results.append(result)
                        seen_links.add(link)
                        
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")
        
        return all_results[:15]  # Return top 15 unique results


class PlannerAgent:
    """
    Specialized agent for creating detailed travel itineraries.
    """
    
    def __init__(
        self,
        google_api_key: str,
        model_name: str = "gemini-2.5-flash-lite",
        temperature: float = 0.3
    ):
        """Initialize the planner agent."""
        import os
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_retries=2
        )
        
        self.system_prompt = """You are an expert travel itinerary planner.
Create detailed, day-by-day itineraries that are:
- Practical and realistic
- Well-paced with appropriate timing
- Include specific recommendations
- Cite sources with links
- Consider travel time between locations

Format each day as:
Day X:
Morning (9:00 AM - 12:00 PM):
- Activity/Location with details
- Source: [Title](link)

Afternoon (12:00 PM - 6:00 PM):
- Activity/Location with details
- Lunch recommendation
- Source: [Title](link)

Evening (6:00 PM onwards):
- Activity/Location with details
- Dinner recommendation
- Source: [Title](link)

Accommodation: Recommended area/hotel
"""
    
    def create_itinerary(
        self,
        destination: str,
        num_days: int,
        research_results: List[Dict[str, Any]]
    ) -> str:
        """Create a detailed itinerary from research results."""
        # Prepare research summary
        research_summary = "\n\n".join([
            f"- {r.get('title', 'N/A')}\n  Link: {r.get('link', 'N/A')}\n  Info: {r.get('snippet', 'N/A')}\n  Query: {r.get('query', 'N/A')}"
            for r in research_results[:15]
        ])
        
        prompt = f"""Create a {num_days}-day itinerary for {destination}.

Research Results:
{research_summary}

Create a detailed itinerary using the information above. Include specific recommendations and cite sources."""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Failed to create itinerary: {e}")
            return f"Error creating itinerary: {str(e)}"
