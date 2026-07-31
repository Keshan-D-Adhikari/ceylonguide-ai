import streamlit as st
import os
import sys

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import config
from src.graph.workflow import create_workflow

# 1. Page Configuration
st.set_page_config(
    page_title="CeylonGuide AI - Sri Lanka Travel Planner",
    page_icon="🇱🇰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "itinerary" not in st.session_state:
    st.session_state.itinerary = None
if "events" not in st.session_state:
    st.session_state.events = []
if "state_trace" not in st.session_state:
    st.session_state.state_trace = {}
if "error_msgs" not in st.session_state:
    st.session_state.error_msgs = []

# Title & Subheader
st.title("🇱🇰 CeylonGuide AI")
st.subheader("Multi-Agent RAG-Based Sri Lanka Travel Planner")

st.markdown(
    """
    Welcome to CeylonGuide AI, an intelligent, multi-agent travel planner for Sri Lanka.
    This application utilizes a structured agentic architecture orchestration via **LangGraph**, 
    reinforced by **Retrieval-Augmented Generation (RAG)** using a local vector database to plan 
    your custom Sri Lankan adventure.
    """
)

# 2. Sidebar Implementation
st.sidebar.image("https://img.icons8.com/color/96/sri-lanka-map.png", width=80)
st.sidebar.markdown("### About CeylonGuide AI")
st.sidebar.markdown(
    """
    **CeylonGuide AI** uses a collaborative agent workflow to process travel requests:
    1. **Router Agent** (Groq): Extracts intent and parameters.
    2. **Retrieval Agent** (Local ChromaDB): Searches indexed guides.
    3. **Planner Agent** (OpenRouter): Generates a structured multi-day itinerary.
    4. **Critic Agent** (Groq/OpenRouter): Reviews the draft for feasibility.
    5. **Final Synthesizer**: Formats and outputs the final markdown result.
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System API Status")

# Check API Keys (masked representation)
groq_ok = len(config.GROQ_API_KEY.strip()) > 0
or_ok = len(config.OPENROUTER_API_KEY.strip()) > 0

st.sidebar.markdown(
    f"🟢 **Groq Router Model**: `Configured`" if groq_ok else "🔴 **Groq Router Model**: `Not Configured`"
)
st.sidebar.markdown(
    f"🟢 **OpenRouter Planner**: `Configured`" if or_ok else "🔴 **OpenRouter Planner**: `Not Configured`"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Configuration Details")
st.sidebar.markdown(f"- **RAG Collections**: `ceylonguide_tourism`")
st.sidebar.markdown(f"- **Embedding Model**: `all-MiniLM-L6-v2` (Local)")
st.sidebar.markdown(f"- **Router model**: `{config.GROQ_ROUTER_MODEL}`")
st.sidebar.markdown(f"- **Planner model**: `{config.OPENROUTER_REASONING_MODEL}`")

# 3. Main Screen Layout: Input Form
st.markdown("### Plan Your Journey")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Option A: Structured Specifications")
    
    duration = st.slider("Trip Duration (Days)", min_value=1, max_value=14, value=5)
    start_city = st.text_input("Starting Location", value="Colombo")
    
    interests = st.multiselect(
        "Interests / Focus Area",
        options=["Beaches", "Wildlife", "Culture", "History", "Nature", "Hiking", "Adventure", "Food", "Relaxation"],
        default=["Wildlife", "Beaches", "Culture"]
    )
    
    budget = st.selectbox(
        "Travel Style / Budget",
        options=["Budget (Local stays / Backpacking)", "Moderate (Boutique hotels / Private driver)", "Premium (Luxury hotels / Premium tourer)"],
        index=1
    )

with col2:
    st.markdown("#### Option B: Natural Language Request")
    user_prompt = st.text_area(
        "Describe your dream vacation in your own words:",
        height=150,
        placeholder="I am visiting Sri Lanka for 5 days starting from Colombo. I want a relaxing trip focused on wildlife and beaches.",
        value="I am visiting Sri Lanka for 5 days. I am starting from Colombo and I like wildlife, beaches and cultural attractions. Create a suitable itinerary."
    )
    
    input_mode = st.radio(
        "Query Inputs Mode",
        options=["Use Structured Specifications (Option A)", "Use Natural Language Prompt (Option B)"],
        index=1
    )

# Form Action Button
if st.button("Plan My Trip", type="primary"):
    # Clear old results
    st.session_state.itinerary = None
    st.session_state.events = []
    st.session_state.state_trace = {}
    st.session_state.error_msgs = []
    
    # 4. Formulate final query
    if input_mode == "Use Structured Specifications (Option A)":
        budget_level = "moderate"
        if "Budget" in budget:
            budget_level = "budget"
        elif "Premium" in budget:
            budget_level = "premium"
            
        spec_interests = ", ".join(interests) if interests else "sightseeing"
        query_str = f"Plan a {duration}-day Sri Lanka trip starting from {start_city} focused on {spec_interests}. Style is {budget_level}."
        start_loc_extracted = start_city
        duration_extracted = duration
        interests_extracted = [i.lower() for i in interests]
        budget_extracted = budget_level
    else:
        # Prompt mode
        if not user_prompt.strip():
            st.error("Please enter a prompt or switch to Structured Specifications.")
            st.stop()
        query_str = user_prompt
        # Let the Router extract these values from LLM, but provide UI defaults as fallback
        duration_extracted = duration
        start_loc_extracted = start_city
        interests_extracted = [i.lower() for i in interests]
        budget_extracted = "moderate"

    # API key validation guards
    if not groq_ok or not or_ok:
        st.warning("⚠️ **Missing API Credentials**")
        st.markdown(
            """
            This application requires Groq and OpenRouter keys to make model requests.
            Please set the following keys in your `.env` file or Streamlit Cloud Secrets:
            - `GROQ_API_KEY`
            - `OPENROUTER_API_KEY`
            """
        )
        st.stop()
        
       # Check if vector DB is loaded; build it automatically on first cloud run
    if not os.path.exists(config.CHROMA_DIR) or len(os.listdir(config.CHROMA_DIR)) == 0:
        with st.spinner("Preparing the travel knowledge base for the first run..."):
            try:
                from scripts.ingest import main as run_ingestion
                run_ingestion()
                st.success("Knowledge base prepared successfully.")
            except Exception as e:
                st.error(f"Failed to prepare the vector database: {e}")
                st.stop()

    # 5. Executing Workflow
    with st.status("Agentic workflow planning trip...", expanded=True) as status:
        try:
            workflow = create_workflow()
            
            initial_state = {
                "user_request": query_str,
                "duration_days": duration_extracted,
                "starting_location": start_loc_extracted,
                "interests": interests_extracted,
                "budget": budget_extracted,
                "revision_count": 0,
                "max_revisions": 1,
                "errors": [],
                "workflow_events": ["✈ Initiating CeylonGuide AI multi-agent workflow..."]
            }
            
            # Invoke Graph
            result = workflow.invoke(initial_state)
            
            # Save results to session state
            st.session_state.itinerary = result.get("final_answer")
            st.session_state.events = result.get("workflow_events", [])
            st.session_state.error_msgs = result.get("errors", [])
            st.session_state.state_trace = {
                "intent": result.get("intent"),
                "duration_days": result.get("duration_days"),
                "starting_location": result.get("starting_location"),
                "interests": result.get("interests"),
                "budget": result.get("budget"),
                "complexity": result.get("complexity"),
                "requires_rag": result.get("requires_rag"),
                "requires_planning": result.get("requires_planning"),
                "revision_count": result.get("revision_count"),
                "critique": result.get("critique"),
                "docs_count": len(result.get("retrieved_documents", []))
            }
            
            status.update(label="Itinerary planning complete!", state="complete")
        except Exception as e:
            status.update(label="Workflow failed to execute.", state="error")
            st.error(f"Execution Error: An error occurred during agent orchestration. Details: {e}")

# 6. Render Outputs
if st.session_state.itinerary:
    st.markdown("---")
    st.success("### Generated Itinerary")
    
    # Display the final compiled synthesis
    st.markdown(st.session_state.itinerary)
    
    # 7. Expandable Workflow Event Logs
    st.markdown("---")
    with st.expander("🔍 Agent Process Details", expanded=False):
        st.markdown("#### High-Level Event Flow")
        for ev in st.session_state.events:
            st.markdown(ev)
            
        trace = st.session_state.state_trace
        if trace:
            st.markdown("#### Extracted Decision Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Intent**: `{trace['intent']}`")
                st.write(f"**Duration**: `{trace['duration_days']} days`")
                st.write(f"**Starting Point**: `{trace['starting_location']}`")
            with col2:
                st.write(f"**Interests**: `{', '.join(trace['interests']) if trace['interests'] else 'None'}`")
                st.write(f"**Budget Level**: `{trace['budget']}`")
                st.write(f"**Complexity**: `{trace['complexity']}`")
            with col3:
                st.write(f"**Requires RAG**: `{trace['requires_rag']}`")
                st.write(f"**Requires Planning**: `{trace['requires_planning']}`")
                st.write(f"**Revision Cycles**: `{trace['revision_count']}`")
                
            crit = trace.get("critique")
            if crit:
                st.markdown("#### Critic Reflection Report")
                st.write(f"**Critic Evaluation Score**: `{crit.get('score')}/10`")
                st.write(f"**Critic Approved Status**: `{crit.get('approved')}`")
                if crit.get("issues"):
                    st.write("**Critic Highlighted Issues**:")
                    for issue in crit["issues"]:
                        st.write(f"- ⚠ {issue}")
                if crit.get("suggestions"):
                    st.write("**Critic Suggestions**:")
                    for sugg in crit["suggestions"]:
                        st.write(f"- 💡 {sugg}")

    # 8. RAG Source Documents
    with st.expander("📚 Retrieved Knowledge Base Chunks", expanded=False):
        st.markdown(
            """
            > [!IMPORTANT]
            > **Data Integrity Notice**: The local tourism guides are template/academic placeholders.
            > Source URLs and dates are empty as they represent unverified content requiring validation.
            """
        )
