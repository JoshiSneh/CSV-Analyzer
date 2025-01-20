import streamlit as st

def setup_session_state():
    """Initialize session state variables."""
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'execution_complete' not in st.session_state:
        st.session_state.execution_complete = False
    if 'summary_complete' not in st.session_state:
        st.session_state.summary_complete = False
    if 'question_complete' not in st.session_state:
        st.session_state.question_complete = False
    if 'current_query' not in st.session_state:
        st.session_state.current_query = None
    if 'task_plan' not in st.session_state:
        st.session_state.task_plan = None
    if 'summary_data' not in st.session_state:
        st.session_state.summary_data = None
    if 'graph_data' not in st.session_state:
        st.session_state.graph_data = None