# AI Travel Planner - LangGraph Edition (2025)

A modern AI-powered travel planning application using **LangGraph**, **LangChain**, and **Google Gemini 2.0**. This application follows the latest 2025 best practices for building AI agents.

## 🚀 What's New (2025 Update)

This version has been completely refactored to use modern LangChain/LangGraph patterns:

### Key Improvements

1. **LangGraph Integration** ✨
   - Replaced deprecated `LLMChain` with LangGraph's `StateGraph`
   - Implements the ReAct (Reasoning + Acting) pattern
   - Stateful agent execution with proper state management
   - Support for multi-turn conversations with memory

2. **Modern Tool Implementation** 🛠️
   - Tools now use the `@tool` decorator (LangChain standard)
   - Proper tool schemas with type hints and docstrings
   - Tools are decoupled from agents for better reusability
   - Automatic schema generation for LLM tool calling

3. **Multi-Agent Architecture** 🤖
   - **ResearchAgent**: Specialized in gathering travel information
   - **PlannerAgent**: Focused on creating detailed itineraries
   - **TravelAgent**: General-purpose agent with tool calling capabilities
   - Agents can be composed and orchestrated independently

4. **Better Error Handling & Observability** 📊
   - Comprehensive logging throughout the application
   - Better error messages and fallback mechanisms
   - Activity logs visible in the UI
   - Ready for LangSmith integration

5. **Streaming Support** 🌊
   - Async streaming capabilities built-in
   - Can show real-time agent progress (ready to implement in UI)

## 🏗️ Architecture

### Agent Pattern: ReAct (Reasoning + Acting)

The application implements the ReAct pattern using LangGraph:

```
┌─────────────────────────────────────────┐
│          User Input                     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Agent Node (LLM + Reasoning)       │
│  - Analyzes request                     │
│  - Decides which tools to use           │
│  - Generates tool calls                 │
└──────────────┬──────────────────────────┘
               │
               ▼
         [Decision Point]
               │
      ┌────────┴─────────┐
      │                  │
      ▼                  ▼
┌──────────┐      ┌──────────┐
│  Tools   │      │   END    │
│  - Search│      └──────────┘
│  - Other │
└────┬─────┘
     │
     │ (Loop back with results)
     │
     └──────────────────┐
                        │
                        ▼
                 ┌──────────────┐
                 │ Agent Node   │
                 │ (Process     │
                 │  results)    │
                 └──────────────┘
```

### File Structure

```
TravelAgent/
├── agents/
│   ├── travel_agent.py      # Main agent classes (LangGraph-based)
│   ├── planner.py            # [DEPRECATED] Use travel_agent.py instead
│   └── researcher.py         # [DEPRECATED] Use travel_agent.py instead
├── tools/
│   ├── search_tool.py        # Modern @tool decorator implementation
│   └── serp_tool.py          # [DEPRECATED] Old class-based tool
├── utils/
│   ├── config.py             # Configuration utilities
│   └── ics.py                # Calendar file generation
├── app.py                    # Streamlit UI (updated)
└── requirements.txt          # Updated dependencies
```

## 📦 Installation

1. **Clone the repository**
```bash
cd /Users/bhushanshah/Documents/All-About-LLMs/AgenticAI/TravelAgent
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up API keys**

Create a `.env` file in the TravelAgent directory:

```env
GOOGLE_API_KEY=your_google_ai_api_key_here
SERPAPI_KEY=your_serpapi_key_here
```

Get your API keys:
- Google AI (Gemini): https://makersuite.google.com/app/apikey
- SerpAPI: https://serpapi.com/

## 🎯 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Using the Agent Programmatically

#### Example 1: Basic Travel Agent

```python
from agents.travel_agent import TravelAgent
from tools.search_tool import search_travel_info

# Initialize the agent
tools = [search_travel_info]
agent = TravelAgent(
    google_api_key="your_key",
    tools=tools,
    model_name="gemini-2.0-flash-exp"
)

# Use the agent
result = agent.invoke(
    user_input="Plan a 3-day trip to Paris",
    destination="Paris",
    num_days=3
)

# Get the response
response = agent.get_last_message(result)
print(response)
```

#### Example 2: Research + Planning Pipeline

```python
from agents.travel_agent import ResearchAgent, PlannerAgent
from tools.search_tool import search_travel_info

# Initialize agents
researcher = ResearchAgent(
    google_api_key="your_key",
    search_tool=search_travel_info
)

planner = PlannerAgent(
    google_api_key="your_key"
)

# Research phase
queries = researcher.generate_search_queries("Tokyo", 5)
results = researcher.search_and_aggregate(queries)

# Planning phase
itinerary = planner.create_itinerary(
    destination="Tokyo",
    num_days=5,
    research_results=results
)

print(itinerary)
```

#### Example 3: Streaming Responses

```python
import asyncio
from agents.travel_agent import TravelAgent
from tools.search_tool import search_travel_info

async def stream_example():
    agent = TravelAgent(
        google_api_key="your_key",
        tools=[search_travel_info]
    )
    
    async for event in agent.astream(
        user_input="What are the top 3 things to do in Barcelona?",
        destination="Barcelona",
        num_days=3
    ):
        print(event)

# Run the async function
asyncio.run(stream_example())
```

## 🔧 Advanced Configuration

### Custom System Prompts

```python
custom_prompt = """You are a luxury travel expert specializing in 
high-end experiences. Focus on premium accommodations, fine dining, 
and exclusive activities."""

agent = TravelAgent(
    google_api_key="your_key",
    tools=tools,
    system_prompt=custom_prompt
)
```

### Conversation Persistence

```python
# Create a conversation with memory
config = {"configurable": {"thread_id": "user_123"}}

# First message
result1 = agent.invoke(
    "Plan a trip to Rome",
    destination="Rome",
    num_days=4,
    config=config
)

# Follow-up message (agent remembers context)
result2 = agent.invoke(
    "Add more food recommendations",
    config=config  # Same thread_id
)
```

### Custom Tools

Create your own tools using the `@tool` decorator:

```python
from langchain_core.tools import tool
from typing import List

@tool
def get_flight_prices(origin: str, destination: str, date: str) -> List[dict]:
    """
    Get flight prices between two cities on a specific date.
    
    Args:
        origin: Departure city (e.g., "New York")
        destination: Arrival city (e.g., "London")
        date: Travel date in YYYY-MM-DD format
    
    Returns:
        List of flight options with prices
    """
    # Your implementation here
    return [{"airline": "Example Air", "price": 500}]

# Add to agent
tools = [search_travel_info, get_flight_prices]
agent = TravelAgent(google_api_key="your_key", tools=tools)
```

## 🆚 Migration Guide (Old vs New)

### Old Way (Deprecated)
```python
# Old: Using LLMChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(inputs)

# Old: Custom tool class
class MyTool:
    def search(self, query):
        # ...
        pass
```

### New Way (2025 Best Practices)
```python
# New: Using LangGraph with ReAct pattern
from langgraph.graph import StateGraph
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """Tool description for the LLM."""
    # ...
    return result

# Build graph with proper state management
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
# ... add edges and compile
```

## 📚 Key Concepts

### 1. ReAct Pattern
The agent alternates between:
- **Reasoning**: Analyzing the task and deciding what to do
- **Acting**: Calling tools to gather information
- **Observing**: Processing tool results
- Repeat until task is complete

### 2. Tool Calling
Modern LLMs can generate structured tool calls:
```json
{
  "name": "search_travel_info",
  "arguments": {
    "query": "best restaurants in Tokyo",
    "num_results": 5
  }
}
```

### 3. State Management
LangGraph maintains state across the agent loop:
- Message history
- Tool results
- Custom state variables
- Automatic state updates with reducers

### 4. Checkpointing
Save and restore agent state:
- Resume conversations
- Time-travel debugging
- Implement human-in-the-loop patterns

## 🐛 Troubleshooting

### Issue: "Module not found" errors
```bash
# Ensure you're in the virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Issue: API key errors
```bash
# Check your .env file exists and has correct keys
cat .env

# Or set them directly in the UI sidebar
```

### Issue: Import errors with old files
The old `planner.py`, `researcher.py`, and `serp_tool.py` are deprecated. 
Use the new `travel_agent.py` and `search_tool.py` instead.

## 🔬 Testing

### Unit Tests (Example)

```python
import pytest
from tools.search_tool import search_travel_info

def test_search_tool():
    """Test the search tool returns proper structure."""
    result = search_travel_info.invoke({
        "query": "test query",
        "num_results": 3
    })
    
    assert isinstance(result, list)
    if len(result) > 0 and "error" not in result[0]:
        assert "title" in result[0]
        assert "link" in result[0]
        assert "snippet" in result[0]
```

## 📈 Performance Tips

1. **Use appropriate model sizes**
   - `gemini-2.0-flash-exp`: Fast, good for most tasks
   - `gemini-2.0-pro-exp`: More capable, slower

2. **Limit search results**
   - More results = more tokens = slower responses
   - 5-10 results per query is usually sufficient

3. **Enable caching** (when available)
   - Gemini supports context caching for repeated queries

## 🤝 Contributing

To add new features:

1. **New tools**: Add to `tools/` using `@tool` decorator
2. **New agents**: Extend `TravelAgent` or create specialized agents
3. **UI improvements**: Modify `app.py` (Streamlit)

## 📖 Resources

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [LangChain Tools Guide](https://python.langchain.com/docs/how_to/custom_tools/)
- [Gemini API Docs](https://ai.google.dev/docs)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)

## 📝 License

[Your License Here]

## 🙏 Acknowledgments

Built with:
- LangChain & LangGraph
- Google Gemini 2.0
- SerpAPI
- Streamlit

---

**Note**: This is a modernized version using 2025 LangChain best practices. The old implementation using `LLMChain` is now deprecated but still available in `planner.py` and `researcher.py` for reference.
