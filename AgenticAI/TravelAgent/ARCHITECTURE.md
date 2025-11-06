# Architecture Diagrams

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ User Inputs  │  │ Agent Logs   │  │ Download Options   │   │
│  └──────┬───────┘  └──────────────┘  └────────────────────┘   │
└─────────┼───────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Layer (agents/)                       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    TravelAgent (Main)                     │  │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────────┐   │  │
│  │  │ LangGraph  │→ │ ReAct Loop │→ │ State Management │   │  │
│  │  │  Workflow  │  │            │  │                  │   │  │
│  │  └────────────┘  └────────────┘  └──────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐  │
│  │  ResearchAgent      │  │      PlannerAgent               │  │
│  │  - Query generation │  │  - Itinerary creation           │  │
│  │  - Search execution │  │  - Result formatting            │  │
│  └─────────────────────┘  └─────────────────────────────────┘  │
└─────────────┬─────────────────────────┬─────────────────────────┘
              │                         │
              ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Tool Layer (tools/)                        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  @tool search_travel_info                                │   │
│  │  - Web search via SerpAPI                                │   │
│  │  - Returns: title, link, snippet                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  @tool generate_search_queries                           │   │
│  │  - Creates optimized search queries                      │   │
│  │  - Returns: list of query strings                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  External Services                              │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   SerpAPI    │  │   Gemini AI  │  │  LangSmith (opt.)    │  │
│  │  Web Search  │  │  LLM Model   │  │  Observability       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## ReAct Agent Flow (TravelAgent)

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Input                              │
│         "Plan a 3-day trip to Tokyo with focus on food"         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Initial State      │
                    │  - messages: [user]  │
                    │  - destination: Tokyo│
                    │  - num_days: 3       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Agent Node         │
                    │   (LLM Reasoning)    │
                    │                      │
                    │ "I need to search    │
                    │  for Tokyo food      │
                    │  recommendations"    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Should Continue?    │
                    └────┬─────────────┬───┘
                         │             │
                   Tool Calls?     No tool calls?
                         │             │
                         ▼             ▼
              ┌──────────────────┐  ┌─────┐
              │   Tools Node     │  │ END │
              │                  │  └─────┘
              │ Execute:         │
              │ search_travel_   │
              │ info(            │
              │   query="Tokyo   │
              │   food"          │
              │ )                │
              └────────┬─────────┘
                       │
                       │ Tool Results
                       │
                       ▼
              ┌──────────────────────┐
              │   Agent Node         │
              │   (Process Results)  │
              │                      │
              │ "Based on results,   │
              │  here's a plan..."   │
              └────────┬─────────────┘
                       │
                       ▼
              ┌──────────────────────┐
              │  Should Continue?    │
              └────┬─────────────┬───┘
                   │             │
         Need more info?    Ready to answer?
                   │             │
                   ▼             ▼
            ┌──────────┐      ┌─────┐
            │  Tools   │      │ END │
            │  (Loop)  │      │     │
            └──────────┘      └─────┘
```

## Tool Calling Mechanism

```
┌─────────────────────────────────────────────────────────────────┐
│                        LLM with Tools                           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                    LLM sees tool schemas:
                    {
                      "name": "search_travel_info",
                      "description": "Search the web...",
                      "parameters": {
                        "query": "string",
                        "num_results": "integer"
                      }
                    }
                               │
                               ▼
                    ┌──────────────────────┐
                    │  LLM Decision        │
                    │                      │
                    │  "I need current     │
                    │   info, I'll call    │
                    │   search_travel_info"│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Tool Call Generated │
                    │                      │
                    │  tool_calls: [       │
                    │    {                 │
                    │      name: "search_  │
                    │      travel_info",   │
                    │      args: {         │
                    │        query: "best  │
                    │        Tokyo food",  │
                    │        num_results: 5│
                    │      }               │
                    │    }                 │
                    │  ]                   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  ToolNode Executes   │
                    │                      │
                    │  search_travel_info( │
                    │    query="best Tokyo │
                    │    food",            │
                    │    num_results=5     │
                    │  )                   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Tool Results        │
                    │                      │
                    │  [                   │
                    │    {title: "...",    │
                    │     link: "...",     │
                    │     snippet: "..."}  │
                    │  ]                   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Back to Agent       │
                    │  (Process results)   │
                    └──────────────────────┘
```

## State Management

```
┌─────────────────────────────────────────────────────────────────┐
│                     TravelAgentState                            │
│                                                                  │
│  messages: [                                                     │
│    SystemMessage("You are a travel expert..."),                 │
│    HumanMessage("Plan trip to Tokyo"),                          │
│    AIMessage(content="...", tool_calls=[...]),                  │
│    ToolMessage(content="[search results]", tool_call_id="..."), │
│    AIMessage(content="Based on research...")                    │
│  ]                                                               │
│  ↑ Managed by add_messages reducer (auto-appends)               │
│                                                                  │
│  destination: "Tokyo"      ← Custom state fields                │
│  num_days: 3               ← Persist across nodes               │
│  research_results: [...]   ← Accumulated data                   │
│  itinerary: "..."          ← Final output                       │
└─────────────────────────────────────────────────────────────────┘
```

## Specialized Agent Pipeline

```
User Request: "Plan 4-day trip to Barcelona"
                        │
                        ▼
        ┌───────────────────────────────┐
        │    ResearchAgent              │
        │                               │
        │  1. Generate Queries:         │
        │     - "Barcelona attractions" │
        │     - "Barcelona food"        │
        │     - "Barcelona hotels"      │
        │                               │
        │  2. Execute Searches:         │
        │     [search_travel_info]      │
        │                               │
        │  3. Aggregate Results:        │
        │     [15 unique sources]       │
        └─────────────┬─────────────────┘
                      │
                      │ research_results
                      │
                      ▼
        ┌───────────────────────────────┐
        │    PlannerAgent               │
        │                               │
        │  1. Analyze Research          │
        │  2. Structure by Days         │
        │  3. Add Details & Sources     │
        │  4. Format Output             │
        │                               │
        │  Output: Detailed Itinerary   │
        └───────────────────────────────┘
                      │
                      ▼
              Generated Itinerary
```

## Conversation Memory Flow

```
Session 1 (thread_id: "user_123")
┌─────────────────────────────────────────────┐
│ Turn 1: "Plan Tokyo trip"                   │
│ → State saved to MemorySaver                │
│   messages: [user_msg, ai_response]         │
└─────────────────────────────────────────────┘
                    │
                    │ Same thread_id
                    ▼
┌─────────────────────────────────────────────┐
│ Turn 2: "Add more food options"             │
│ → State loaded from MemorySaver             │
│   messages: [previous..., new_user_msg]     │
│ → Agent has full context                    │
│ → Response considers previous conversation  │
└─────────────────────────────────────────────┘
                    │
                    │ Same thread_id
                    ▼
┌─────────────────────────────────────────────┐
│ Turn 3: "What was my destination?"          │
│ → Agent remembers: "Tokyo"                  │
└─────────────────────────────────────────────┘


Session 2 (thread_id: "user_456")
┌─────────────────────────────────────────────┐
│ Fresh conversation, no shared context       │
└─────────────────────────────────────────────┘
```

## Comparison: Old vs New Architecture

### Old Architecture (Deprecated)
```
Streamlit UI
     ↓
┌─────────────────┐
│  Manual Chain   │  LLMChain (deprecated)
└────────┬────────┘
         │
         ├→ Researcher (custom class)
         │   └→ SerpTool (custom class)
         │       └→ Manual tool calls
         │
         └→ Planner (custom class)
             └→ LLMChain (deprecated)

Issues:
❌ No tool calling
❌ No state management
❌ Tight coupling
❌ Manual orchestration
❌ No conversation memory
```

### New Architecture (2025)
```
Streamlit UI
     ↓
┌─────────────────────────┐
│  LangGraph StateGraph   │  Modern orchestration
└────────┬────────────────┘
         │
         ├→ TravelAgent
         │   ├→ Agent Node (ReAct)
         │   ├→ Tool Node (automatic)
         │   └→ State Management
         │
         ├→ ResearchAgent (specialized)
         └→ PlannerAgent (specialized)

Tools (decoupled):
  └→ @tool search_travel_info
  └→ @tool custom_tools...

Benefits:
✅ Automatic tool calling
✅ Built-in state management
✅ Loose coupling
✅ Autonomous orchestration
✅ Conversation memory
✅ Easy to extend
```

## Data Flow

```
Input: "Plan 3-day Tokyo trip"
         │
         ▼
    [StateGraph Entry]
         │
         ▼
    [Agent Node]
    "I need Tokyo info"
         │
         ▼
    [Conditional Edge]
    Tool calls detected?
         │
         ├─ Yes ─→ [Tools Node]
         │            │
         │            ├→ search_travel_info
         │            │   ↓
         │            │  [SerpAPI]
         │            │   ↓
         │            │  Results
         │            │
         │            └→ [ToolMessage]
         │                │
         └──────────────┬─┘
                        │
                        ▼
                   [Agent Node]
                   "Based on results..."
                        │
                        ▼
                   [Conditional Edge]
                   Need more tools?
                        │
                        ├─ Yes ─→ [Loop back]
                        │
                        └─ No ──→ [END]
                                   │
                                   ▼
                              Final Response
```

## Key Takeaways

1. **LangGraph** provides the orchestration framework
2. **ReAct pattern** enables autonomous tool calling
3. **@tool decorator** makes tools discoverable by LLM
4. **State management** maintains context across interactions
5. **Decoupled architecture** allows easy extension

The new architecture is:
- ✅ More maintainable
- ✅ More flexible
- ✅ More powerful
- ✅ Production-ready
- ✅ Aligned with 2025 standards
