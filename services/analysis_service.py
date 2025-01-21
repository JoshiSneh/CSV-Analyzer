import streamlit as st
import time
import pandas as pd
import plotly.graph_objects as go
import io
import plotly.express as px
import numpy as np
import re
from contextlib import redirect_stdout, redirect_stderr

class AnalysisService:
    def __init__(self, openai_service, df):
        self.openai_service = openai_service
        self.df = df
        
    def run_analysis_pipeline(self, user_query):
        """Run the complete analysis pipeline."""
        st.session_state.current_query = user_query
        
        with st.container():
            st.subheader("📋 Analysis Pipeline")
            self._generate_analysis_plan()
            
            if st.session_state.analysis_complete:
                self._generate_code()
            
            if st.session_state.code_generate:
                self._generate_analysis()

            if st.session_state.execution_complete:
                self._generate_summary()
                
            if st.session_state.summary_complete:
                self._generate_questions()


    def _generate_analysis_plan(self):
        """Generate analysis plan."""
        with st.status("Generating Analysis Plan") as status:

            task_planner_prompt = """
            ### Task Planning System
            You are a specialized task planning agent. Your role is to create precise, executable schema based task plans for analyzing DataFrame 'df'. Think step-by-step and generate a detailed task plan.

            ### Core Requirements
            1. Each task must be:
            - Specific and directly executable with the `exec()` function of Python
            - Each task and sub-task should involve Python code execution to derive results. For example, identifying trends or patterns should be explicitly linked to Python code that performs the necessary computations or visualizations, rather than being presented as a high-level instruction
            - Based solely on available columns. Do not assume additional data or columns
            - Focused on DataFrame operations
            - Make sure all the data types are handled properly. Look for the data types first, of the columns and then give the task accordingly. Never assume the data types of the columns on your own
            - Contributing to the final solution
            - Evaluate the context of the user query to determine the appropriate string comparison method
            - Apply flexible string matching techniques when broader criteria are required
            - Building logically on previous steps
            - When doing string extraction from columns make sure to handle the existence or not
            - Make sure to handle all edge cases and potential data issues gracefully. For example, missing values, incorrect data types etc
            - Plotly does not support non-serializable data types like Period. To ensure compatibility, convert dt.to_period() to serializable types, such as using .astype(str) to transform Period objects into strings
            - Do not generate tasks that can't be executed on the given dataframe and throw an error
            - Make sure all non-JSON serializable column types (e.g., pd.Period) in the DataFrame
            - Convert such columns to serializable formats like strings using .astype(str)
            - Confirm the updated DataFrame is compatible with Plotly visualizations
            - At last, convert all the important operations into a dataframe and give the result
            - If there is a final dataframe then DO NOT convert that to the dictionary format. Keep the dataframes as it is
            - Mention all the Keys required in the required format for the final result with the last task

            2. Task Structure:
            - Each main task should be broken down into detailed sub-tasks
            - Sub-tasks should represent atomic operations that build up to complete the main task
            - Each sub-task should be directly executable and handle one specific aspect
            - Sub-tasks should include data type validations and conversions where needed
            - Sub-tasks should handle error cases and edge conditions
            - Sub-tasks should build progressively toward the main task goal

            3. Variable Management:
            - Store key intermediate results as DataFrame operations
            - Use descriptive variable names related to their content
            - Maintain data types appropriate for the analysis
            - Each sub-task should clearly specify any new variables created

            4. Visualization Requirements (if needed):
            - Use Plotly exclusively
            - Make sure to generate the visualization based on the user query and the previous task. Look for the previous steps and then generate the visualization accordingly
            - Never generate the task with wrong x and y axis. Always look for the previous steps and then generate the visualization accordingly
            - Store plot in variable 'fig' and if multiple plots are needed, then use suffix as `fig_`
            - Specify exact chart type and columns
            - Include all necessary parameters

            5. Final Output Structure:
            - Create output_dict as the final dictionary to store results
            - Key should be descriptive of the result
            - Visualization should start with 'fig' if applicable
            - Meaningful, case-sensitive keys describing contents
            - Final result should be stored in a variable named `output_dict`
            - Include all relevant dataframes and visualizations in `output_dict`. Identify based on the user query and then provide the output
            - If there are any important dataframes, like such dataframes which are important for the analysis, then include them as well in the `output_dict`. For example, final result dataframe, comparison dataframe etc
            - Reset the indexes of the final dataframes (if there) before giving the final result

            ### Quality Standards
            - No assumptions about unavailable data
            - No skipped or redundant steps
            - Clear progression toward solution
            - Complete but concise descriptions
            - Focus on DataFrame operations only
            - Always maintain the Keys formation in the `output_dict` as mentioned above. First word should start with uppercase with space separated words

            ### Output Format
            Task-1: [Precise action description]
                Sub-Task-1.1: [Detailed breakdown of first component]
                Sub-Task-1.2: [Detailed breakdown of second component]
                ...
                Sub-Task-1.n: [Final component of Task-1]
                - Column Names: [Provide all the column names used for Task-1, formatted as a comma-separated list:
                ["Column-1", "Column-2", "Column-3", ...]]

            Task-2: [Precise action description]
                Sub-Task-2.1: [Detailed breakdown of first component]
                Sub-Task-2.2: [Detailed breakdown of second component]
                ...
                Sub-Task-2.n: [Final component of Task-2]
                - Column Names: [Provide all the column names used for Task-2, formatted as a comma-separated list:
                ["Column-1", "Column-2", "Column-3", ...]]

            [...]

            Task-N: [Precise action description - Compile the processed results and store them in the final output dictionary named `output_dict`]
                Sub-Task-N.1: [Detailed breakdown of first component]
                Sub-Task-N.2: [Detailed breakdown of second component]
                ...
                Sub-Task-N.n: [Final component of Task-N]
                - Key Names: [Provide all the key names used in the `output_dict` dictionary.]
                - Values: [For each key, describe the expected value, including details of the information it should contain, formatted as a dictionary: {{"Key-1": "Description of the information contained in this key", "Key-2": "Description of the information contained in this key", ...}}]

            **Provide only the task plan description. Do not include any additional explanations or commentary or python code or output or any other information**
            """
            
            response = self.openai_service.create_completion_task_planner(task_planner_prompt,available_columns=', '.join(self.df.columns),column_data_types="\n".join([f"- **{col}**: {dtype}" for col, dtype in self.df.dtypes.items()]),data_frame_preview="\n".join([f"- **{col}**: {dtype}" for col, dtype in self.df.items()]))
                    
            time.sleep(1)
            status.update(label="✅ Analysis Plan Generated!", state="complete")
            st.code(response.choices[0].message.content)
            st.caption(f"Task Planner Token usage: {response.usage.total_tokens}")
            st.caption(f"Cached Token: {response.usage.prompt_tokens_details.cached_tokens}")
            
            st.session_state.analysis_complete = True
            st.session_state.task_plan = response.choices[0].message.content

    def _execute_task_code(self,code):
        
        try:
            exec_globals = {"df": self.df, "pd": pd, "px": px, "io": io, "np": np,"re":re}
            exec_locals = {}

            exec(code, exec_globals, exec_locals)
            
            # st.warning(exec_locals)
            # Check if output_dict exists in the executed code
            if "output_dict" not in exec_locals:
                raise ValueError("Missing output_dict")      
                
            return exec_locals["output_dict"], None
            
        except Exception as e:
            return None, e

    def _generate_code(self):
        """Generating Code"""
        with st.status("Generating Code") as status:
            task_execution_prompt ="""
            ### Task Execution System

            You are an expert data analysis assistant with deep expertise in pandas, numpy, and data visualization.Your responses will be direct code implementations without explanations, focusing purely on giving the Python Code with the provided task, sub-task plan with optimal efficiency.
            
            ### Role
            Your role is to:
            - Follow a precise task, sub-task plan to execute complex data analysis operations
            - Transform complex data analysis tasks into precise, executable Python code
            - Ensure all operations maintain data integrity and type safety
            - Follow best practices for DataFrame operations and memory efficiency
            - Create clear, professional visualizations that effectively communicate insights
            - Generate production-ready code that adheres to Python standards
            - Handle edge cases and potential data issues gracefully
            - Focus on accuracy and performance in all calculations
            - Handle data operations like filtering, grouping, and aggregations accurately.
            
            ### Core Requirements

            #### Data Operations
            - Dataframe has been already loaded as `df`. Donot create the sample dataframe. Use the existing dataframe `df` for all the operations.
            - All operations must use exact column names from [Available Columns]
            - Donot assume float as a string and do operation of string on it. Same for others type
            - Use the columns name accurately given in the [Execution Plan]
            - Never use any random or vague column names for the dataframe operations
            - Intermediate results stored as pandas DataFrames
            - Variables must have descriptive names reflecting their content
            - All calculations must preserve data types specified in [Column Data Types]
            - Donot fill null values with any another values
            
            #### Instructions for Generating Python Code (to be followed strictly):
            - Think step by step when generating the Python code based on the task plan.
            - Always see the previous tasks block of code and then generate the current task or future task by taking consideration of the current task description.
            - DONOT assume datatypes from your own. Always look into [Column Data Types] for the datatypes of the columns.
            - Handle the cases that can return nan or None from the previous task.
            - Always generate the correct regex pattern for the string operations based on the values present in the columns.
            - To replace values in columns, use the .replace() method. You can use regex with the .replace() method for string replacement.
            - Ensure that the Python code utilizes the appropriate methods or functions accurately and efficiently, prioritizing correctness and optimal usage in all implementations.
            - Ensure all DataFrame columns used are in JSON serializable formats, converting non-serializable types like `pd.Period` to strings using .astype(str) as needed for compatibility. 
            - Import all necessary libraries at the beginning of the code.
            - Check if each value in the column matches the expected format (e.g., datetime format or other expected patterns). Only perform operations (such as parsing or calculations) on values that match the required format, and skip or ignore any non-matching values to avoid errors.
            - Avoid using matplotlib. For plotting, use Plotly exclusively.
            - Interpret user queries and generate functions as needed to fulfill task requirements.
            - Use functions pd.to_datetime() to convert columns to the date format whenever necessary.
            - Add checks or use np.divide with where or np.errstate to handle division by zero safely.
            - Use .str.strip() to remove leading and trailing spaces from strings before performing comparisons or transformations to ensure accuracy.
            - If a final DataFrame is present, ensure it is NOT converted to a dictionary format. Retain the DataFrame in its original structure.
            - Reset the indexs of the final dataframes before giving the final result.
            - Always final output should be stored in a variable named `output_dict` with all the necessary information.

            #### Code Standards
            - Import all the required packages for the tasks
            - Each operation follows task and sub-task plan sequence
            - No deprecated pandas methods
            - Consistent variable naming
            - Type-aware operations

            #### Visualization Standards
            - Use Plotly exclusively
            - Proper figure sizing and formatting
            - Clear labels and titles
            - Appropriate color schemes
            - Interactive elements when relevant

            #### Output Requirements
            - Code only - no explanations.
            - Each step follows from task plan
            - Each step follows from sub-task plan
            - Clean, readable format
            - No print statements unless specified
            - No markdown or text between code blocks
            
            ### Response Format

            - # Imports
            [import statements]

            - # Task Execution 
            [code implementing each task]
            Step-by-Step implementation of the task, sub-task plan based on the `df_task_plan`.

            #Task-1 with proper task description 
            #Sub-Task-1.1 with proper sub-task-1.1 description,
            #Sub-Task-1.2 with proper sub-task-1.2 description,
 
            #Task2 with proper task description
            #Sub-Task-2.1 with proper sub-task-2.1 description,
            #Sub-Task-2.2 with proper sub-task-2.2 description,
            
            [...]

            **Provide only the Correct Python Code which can be run with the `exec()`. Do not include any additional explanations or commentary**
            """

            response = self.openai_service.create_completion_code_generation(task_execution_prompt,st.session_state.task_plan,available_columns=', '.join(self.df.columns),column_data_types="\n".join([f"- **{col}**: {dtype}" for col, dtype in self.df.dtypes.items()]),data_frame_preview="\n".join([f"- **{col}**: {dtype}" for col, dtype in self.df.items()]))
                        
            time.sleep(1.5) 
            status.update(label="✅ Code Generated!", state="complete")
                
            task = response.choices[0].message.content
            task = task.replace('`', '').replace("python", "").strip()
            
            st.code(task, language="python")

            # #Execute the code
            # exec_globals = {"df": self.df, "pd": pd, "px": px, "io": io, "np": np,"re":re}
            # exec_locals = {}
            # exec(task, exec_globals, exec_locals)

            st.caption(f"Code Generation Token usage: {response.usage.total_tokens}")
            st.caption(f"Cached Token: {response.usage.prompt_tokens_details.cached_tokens}")

            st.session_state.code_generate = True
            st.session_state.code = task

    def _generate_analysis(self):

        with st.status("Executing Code") as status:
            output_dict, error = self._execute_task_code(st.session_state.code)

            if error:
                st.error(f"❌ An error occurred during code execution: {str(error)}")
                status.update(label="❌ Code Execution Failed!", state="error")
                st.session_state.execution_complete = False
                st.session_state.summary_complete = False
                return

            visual = False
            
            summary_data = {}
            graph_data = {}
            
            if output_dict:
                for key, value in output_dict.items():
                    if isinstance(value, pd.DataFrame):
                            st.write(f"📈 {key}")
                            st.dataframe(value, use_container_width=True)
                            
                            summary_data[key] = value

                            buffer = io.StringIO()
                            value.to_csv(buffer, index=False)
                            st.download_button(
                                label="📥 Download as CSV",
                                data=buffer.getvalue(),
                                file_name=f"{key.lower().replace(' ', '_')}.csv",
                                mime="text/csv"
                            )
                    elif isinstance(value, go.Figure):
                        st.plotly_chart(value, use_container_width=True)
                        # graph_visual[key] = value.to_json()
                        graph_data[key] = value.to_plotly_json()
                        visual = True
                    else:
                        st.warning(f"{key}: {value}")
                        summary_data[key] = value

                if visual == False:
                    graph_data["fig"] = None
                
                time.sleep(1.0) 
                status.update(label="✅ Code Executed!", state="complete")

                st.session_state.execution_complete = True
                st.session_state.summary_data = summary_data
                st.session_state.graph_data = graph_data

                # print(st.session_state.summary_data)

    def _generate_summary(self):
        """Generate analysis summary."""
        with st.status("Generating Insights") as status:
            summary_prompt ="""
            # Data Summary Assistant

            ## Role
            You are a data insights specialist who transforms analysis results into clear, actionable summaries. Your focus is on:
            - Converting complex data into simple insights
            - Highlighting what matters most to the business
            - Spotting important patterns
            - Creating brief, meaningful summaries

            ## Summary Guidelines

            ### 1. Summary Structure

            **Summary**
            - Direct answer to the user's question
            - 1-2 sentences of essential context (if needed)
            - Main takeaway

            **Key Insights**
            - Show 3-4 representative as sample records in bullet points by letting the user know about the sample records
            - Avoid statistical summaries for the `large` dataframes as it might be misleading
            - Some conclusions based on the data like The dataset provides a comprehensive view... give a brief about the dataset

            **Visualization Analysis** (Only if `User Visualization Data` is not None)
            - What the visualization shows
            - Key patterns or trends
            - Important data points
            - Connection to user's question
            - If the visualization donot exist then don't mention anything about the visualization in the summary

            ### 2. Best Practices

            **Do:**
            - Use simple, clear language
            - Focus on what's most relevant to the question
            - Include specific examples from the data
            - Maintain a logical flow

            **Don't:**
            - Include technical jargon unless necessary
            - Show statistical summaries for large datasets
            - Mix question and answer content
            - Use generic statements
            - Include placeholder text
            - Mention missing visualizations

            ### 3. Format

            ### Summary
            [Brief overview of key findings]

            ### Key Insights
            [Bullet points of main findings]

            ### Visualization Analysis (Only if `User Visualization Data` is not None)
            [Brief visualization analysis]
            """
            # "\n".join([f"- **{col}**: {value}" for col, value in st.session_state.summary_data.items()])
            response = self.openai_service.create_completion_summary(summary_prompt,st.session_state.summary_data,st.session_state.graph_data)
                        
            time.sleep(1)
            status.update(label="✅ Insights Generated!", state="complete")
            
            st.subheader("🎯 Key Insights")
            st.markdown(response.choices[0].message.content)
            st.caption(f"Summary Token usage: {response.usage.total_tokens}")
            st.caption(f"Cached Token: {response.usage.prompt_tokens_details.cached_tokens}")
            st.session_state.summary_complete = True

    def _generate_questions(self):
        """Generate follow-up questions."""
        with st.status("Generating Questions") as status:
            follow_up_prompt = """
            # Follow-up Question Generator

            ### Role
            You are a precise data insight explorer who:
            - Generates follow-up questions based strictly on available data
            - Ensures questions are directly answerable using the dataset
            - Maintains focus on business value and actionable insights
            - Avoids assumptions or speculation beyond the data

            ### Core Requirements

            ### Question Generation Rules
            - Maximum 3 questions total
            - Questions must use only existing columns
            - Each question must be verifiable using the dataset
            - No questions requiring unavailable data
            - No hypothetical or speculative questions
            - Questions should provide new insights
            - Questions must be specific and actionable

            ### Quality Standards
            - Direct relationship to original analysis
            - Clear business value in each question
            - Explicit connection to available data
            - No redundancy with original question
            - Feasible to answer with given columns

            ### Output Format

            1. [Precise question using available data] - [Brief business context and value]

            2. [Precise question using available data] - [Brief business context and value]

            3. [Precise question using available data] - [Brief business context and value]
            """
            
            response = self.openai_service.create_followup_generation(follow_up_prompt,available_columns=', '.join(self.df.columns),data_frame_preview=self.df.head(1).to_markdown())
            
            time.sleep(1)
            status.update(label="✅ Questions Generated!", state="complete")
            
            st.subheader("🔍 Follow-up Questions")
            st.markdown(response.choices[0].message.content)
            st.caption(f"Questions Token usage: {response.usage.total_tokens}")
            st.caption(f"Cached Token: {response.usage.prompt_tokens_details.cached_tokens}")
            st.session_state.question_complete = True
