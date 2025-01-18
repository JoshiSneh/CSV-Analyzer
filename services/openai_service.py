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
    
    def create_completion_task_planner(self, task_planner_prompt,available_columns,column_data_types,data_frame_preview):
        """Create OpenAI chat completion."""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": task_planner_prompt},
                {"role": "user", "content": f"===Dataframe Schema:\n{data_frame_preview}\n\n===Available Columns:\n{available_columns}\n\n===Column Data Types:\n{column_data_types}\n\n===User Question:\n{st.session_state.current_query}\n"}
            ]
        )
        return response
    
    def create_completion_code_generation(self, task_execution_prompt,execution_plan,available_columns,column_data_types,data_frame_preview):
        """Create OpenAI chat completion."""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": task_execution_prompt},
                {"role": "user", "content": f"===Dataframe Schema:\n{data_frame_preview}\n\n===Available Columns:\n{available_columns}\n\n===Column Data Types:\n{column_data_types}\n\n===Execution Plan:\n{execution_plan}\n\n===User Question:\n{st.session_state.current_query}\n\n"}
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