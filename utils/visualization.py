import streamlit as st

def setup_page():
    """Configure page settings and custom CSS."""
    st.set_page_config(
        page_title="CSV Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .status-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .processing {
            background-color: #cce5ff;
            border: 1px solid #b8daff;
            color: #004085;
        }
        </style>
    """, unsafe_allow_html=True)