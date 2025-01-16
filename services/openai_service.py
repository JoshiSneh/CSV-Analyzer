import streamlit as st
from openai import OpenAI

class OpenAIService:
    def __init__(self):
        self.client = None
    
    def setup_api_key(self):
        """Setup OpenAI API key and client."""
        with st.sidebar:
            st.subheader("OpenAI API Key")
            api_key = st.text_input("Enter your OpenAI API key", type="password")
            
            if api_key:
                st.session_state.openai_api_key = api_key
                self.client = OpenAI(api_key=api_key)
                return True
            else:
                st.warning("Please enter your API key")
                return False
    
    def create_completion_task(self, prompt, model="gpt-4o", temperature=0):
        """Create OpenAI chat completion."""
        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": st.session_state.current_query}
            ]
        )
        return response
    
    def create_completion_summary(self, prompt, model="gpt-4o-mini", temperature=0):
        """Create OpenAI chat completion."""
        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": st.session_state.current_query}
            ]
        )
        return response