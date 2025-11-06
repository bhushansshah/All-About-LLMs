# app.py
"""
Streamlit app for AI Travel Planner using LangGraph.
Updated to use modern LangChain/LangGraph patterns (2025).
"""
import streamlit as st
from utils.ics import generate_ics_content
from tools.search_tool import search_travel_info, generate_search_queries
from agents.travel_agent import TravelAgent, ResearchAgent, PlannerAgent
from utils.config import get_env
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Travel Planner (LangGraph)", layout="wide")

st.title("🌍 AI Travel Planner — LangGraph + Gemini 2.0")
st.markdown("""
Plan your perfect trip using **LangGraph** (2025), **Gemini 2.0**, and **ReAct agents**.

This app uses:
- 🤖 **LangGraph** for agent orchestration
- 🧠 **Gemini 2.0 Flash** for intelligent reasoning
- 🔍 **SerpAPI** for real-time travel information
- 📅 **ICS export** for calendar integration
""")

# Sidebar: API keys configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    google_api_key = st.text_input(
        "Google AI API Key (Gemini)",
        type="password",
        value=get_env("GOOGLE_API_KEY", ""),
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )
    serp_api_key = st.text_input(
        "SerpAPI Key",
        type="password",
        value=get_env("SERPAPI_KEY", ""),
        help="Get your API key from https://serpapi.com/"
    )
    
    st.divider()
    st.markdown("### About")
    st.markdown("""
    This app demonstrates modern AI agent patterns:
    - **ReAct pattern** (Reasoning + Acting)
    - **Tool calling** with LangChain
    - **State management** with LangGraph
    - **Multi-agent collaboration**
    """)

# Main form
col1, col2 = st.columns([2, 1])

with col1:
    destination = st.text_input("🎯 Destination", value="Tokyo, Japan", placeholder="e.g., Paris, Tokyo, New York")
    
with col2:
    num_days = st.number_input("📅 Number of days", min_value=1, max_value=30, value=5)

start_date = st.date_input("🗓️ Start date (for calendar export)", value=datetime.utcnow().date())

# Session state initialization
if "itinerary" not in st.session_state:
    st.session_state.itinerary = None
if "research_results" not in st.session_state:
    st.session_state.research_results = None
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []

# Main layout
col_left, col_right = st.columns([3, 1])

with col_left:
    if st.button("🚀 Generate Itinerary", type="primary", use_container_width=True):
        if not google_api_key or not serp_api_key:
            st.error("⚠️ Please provide both Google AI API key and SerpAPI key in the sidebar.")
        elif not destination:
            st.error("⚠️ Please enter a destination.")
        else:
            # Clear previous results
            st.session_state.agent_logs = []
            st.session_state.research_results = None
            st.session_state.itinerary = None
            
            try:
                # Step 1: Initialize agents
                with st.spinner("🔧 Initializing AI agents..."):
                    # Create tools
                    tools = [search_travel_info]
                    
                    # Initialize specialized agents
                    researcher = ResearchAgent(
                        google_api_key=google_api_key,
                        search_tool=search_travel_info
                    )
                    
                    planner = PlannerAgent(
                        google_api_key=google_api_key
                    )
                    
                    st.session_state.agent_logs.append("✅ Agents initialized")
                
                # Step 2: Generate search queries
                with st.spinner("🔍 Generating search queries..."):
                    search_queries = researcher.generate_search_queries(destination, num_days)
                    st.session_state.agent_logs.append(f"✅ Generated {len(search_queries)} search queries")
                    
                    # Display queries in expander
                    with st.expander("🔎 Search Queries"):
                        for i, query in enumerate(search_queries, 1):
                            st.markdown(f"{i}. `{query}`")
                
                # Step 3: Execute searches
                with st.spinner("🌐 Searching the web for travel information..."):
                    research_results = researcher.search_and_aggregate(
                        search_queries,
                        max_results_per_query=5
                    )
                    st.session_state.research_results = research_results
                    st.session_state.agent_logs.append(f"✅ Found {len(research_results)} unique sources")
                    
                    # Display results in expander
                    with st.expander(f"📚 Research Results ({len(research_results)} sources)"):
                        for i, result in enumerate(research_results[:10], 1):
                            st.markdown(f"""
                            **{i}. [{result.get('title', 'N/A')}]({result.get('link', '#')})**  
                            *Query: {result.get('query', 'N/A')}*  
                            {result.get('snippet', 'N/A')[:200]}...
                            """)
                        if len(research_results) > 10:
                            st.markdown(f"*...and {len(research_results) - 10} more results*")
                
                # Step 4: Create itinerary
                with st.spinner("✨ Creating your personalized itinerary..."):
                    itinerary = planner.create_itinerary(
                        destination=destination,
                        num_days=num_days,
                        research_results=research_results
                    )
                    st.session_state.itinerary = itinerary
                    st.session_state.agent_logs.append("✅ Itinerary created successfully")
                
                st.success("🎉 Your itinerary is ready!")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                logger.exception("Error during itinerary generation")

    # Display itinerary
    if st.session_state.itinerary:
        st.divider()
        st.header("📋 Your Personalized Itinerary")
        st.markdown(st.session_state.itinerary)

with col_right:
    st.markdown("### 📥 Export")
    
    # Download button for ICS
    if st.session_state.itinerary:
        try:
            ics_bytes = generate_ics_content(
                st.session_state.itinerary,
                start_date=datetime.combine(start_date, datetime.min.time())
            )
            st.download_button(
                label="📅 Download .ics",
                data=ics_bytes,
                file_name=f"{destination.replace(' ', '_').replace(',', '')}_itinerary.ics",
                mime="text/calendar",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error generating calendar file: {str(e)}")
    
    # Activity log
    if st.session_state.agent_logs:
        st.divider()
        st.markdown("### 📊 Activity Log")
        for log in st.session_state.agent_logs:
            st.markdown(f"- {log}")
    
    # Statistics
    if st.session_state.research_results:
        st.divider()
        st.markdown("### 📈 Statistics")
        st.metric("Sources Found", len(st.session_state.research_results))
        if st.session_state.itinerary:
            st.metric("Days Planned", num_days)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    Built with ❤️ using LangGraph, LangChain, and Gemini 2.0<br>
    <a href="https://python.langchain.com/docs/langgraph" target="_blank">LangGraph Documentation</a> | 
    <a href="https://python.langchain.com/docs/integrations/chat/google_generative_ai" target="_blank">Gemini Integration</a>
</div>
""", unsafe_allow_html=True)
