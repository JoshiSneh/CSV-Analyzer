# app/main.py
import streamlit as st
from utils.data_loader import load_data
from utils.visualization import setup_page
from services.analysis_service import AnalysisService
from services.openai_service import OpenAIService
from config.settings import setup_session_state

def main():
    setup_page()
    
    # Initialize session state
    setup_session_state()
    
    openai_service = OpenAIService()
    if not openai_service.setup_api_key():
        st.stop()
        
    # Main app header
    st.title("🎯 CSV Analyzer")
    st.markdown("""
        Upload your CSV file, ask questions in natural language, and get instant insights!
        Perfect for quick data analysis and visualization.
    """)
    
    # Data upload section
    with st.container():
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv","xlsx"],help="Upload your CSV/XLSX file here.")
    
    if uploaded_file:
        try:
            df = load_data(uploaded_file)
            display_data_preview(df)
            
            analysis_service = AnalysisService(openai_service, df)
            
            st.subheader("🔍 Ask Questions About Your Data")
            user_query = st.text_input(
                "What would you like to know about your data?",
                placeholder="Example: Show me the transactions by month",
                help="Ask questions in plain English - our AI will understand and analyze your data accordingly!"
            )
            
            if st.button("🚀 Analyze Data", use_container_width=True):
                if not user_query:
                    st.error("Please enter a question to analyze your data.")
                else:
                    analysis_service.run_analysis_pipeline(user_query)
                    
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

def display_data_preview(df):
    with st.expander("🔍 Preview Your Data", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df, use_container_width=True)
        with col2:
            st.write("📊 Data Overview")
            st.info(f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}")

if __name__ == "__main__":
    main()