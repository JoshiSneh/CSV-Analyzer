import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from openai import OpenAI
import time

st.set_page_config(
    page_title="CSV Analyzer",
    page_icon="📊",
    layout="wide"
)

if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None

with st.sidebar:
    st.subheader("OpenAI API Key")
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    
    if api_key:
        st.session_state.openai_api_key = api_key
        client = OpenAI(api_key=api_key)
    else:
        st.warning("Please enter your API key")
        st.stop()

# load_dotenv()
# client = OpenAI()


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

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'execution_complete' not in st.session_state:
    st.session_state.execution_complete = False
if 'summary_complete' not in st.session_state:
    st.session_state.summary_complete = False
if 'question_complete' not in st.session_state:
    st.session_state.question_complete = False

st.title("🎯 CSV Analyzer")
st.markdown("""
    Upload your CSV file, ask questions in natural language, and get instant insights!
    Perfect for quick data analysis and visualization.
""")

with st.container():
    st.subheader("📁 Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv","xlsx"], help="Upload your CSV/XLSX file here.")

if uploaded_file:
    try:
        if(uploaded_file.type == "text/csv"):
           df = pd.read_csv(uploaded_file)
        else:
           df = pd.read_excel(uploaded_file)
           
        with st.expander("🔍 Preview Your Data", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(df, use_container_width=True)
            with col2:
                st.write("📊 Data Overview")
                st.info(f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}")
                
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
                total_usage = 0
                with st.container():
                    st.subheader("📋 Analysis Pipeline")
                    
                    with st.status("Generating analysis plan...") as status:
                        task_planner_prompt = (
                        """
                        ### Task Planning System
                        You are a specialized task planning agent. Your role is to create precise, executable schema based task plans for analyzing DataFrame 'df'.

                        ### Input Context
                        - Available DataFrame: `df`
                        - Query: {user_query}
                        - Columns: {df_columns}
                        - DataFrame Preview: {df_str}
                        - Data Types: {df_types}

                        ### Core Requirements
                        1. Each task must be:
                        - Specific and directly executable with the `exec()` function of Python.
                        - Based solely on available columns. Donot assume additional data or columns
                        - Focused on DataFrame operations
                        - Contributing to the final solution
                        - Building logically on previous steps
                        - When doing the string extraction from the columns make sure to handle the case of the existance or not.
                        - Make sure to handle all edge cases and potential data issues gracefully. For example, missing values, incorrect data types etc.
                        - Donot generate task that can't be executed on the given dataframe and throw an error.
                        - At last, convert all the important operations into a dataframe and give the result.
                        - If there is a final dataframe then to DONOT convert that to the dictionary format. Keep the dataframes as it is.
                        
                        2. Variable Management:
                        - Store key intermediate results as DataFrame operations
                        - Use descriptive variable names related to their content
                        - Maintain data types appropriate for the analysis

                        3. Visualization Requirements (if needed):
                        - Use Plotly exclusively
                        - Make sure to generate the visualization based on the user query and the previous task. Look for the previous steps and then generate the visualization accordingly.
                        - Never generate the task with wrong x and y axis. Always look for the previous steps and then generate the visualization accordingly. 
                        - Store plot in variable 'fig' and if multiple plots are needed, then use suffix as `fig_`
                        - Specify exact chart type and columns
                        - Include all necessary parameters

                        4. Final Output Structure:
                        - Create output_dict containing:
                        - Key should be descriptive of the result
                        - Visualization should be start with 'fig' if applicable
                        - Meaningful, case-sensitive keys describing contents
                        - Final result should be stored in a variable named `output_dict`
                        - Inlucde all relevant dataframes and visualizations in `output_dict`. Identify based on the user query and then provide the output.
                        - If there are any important dataframes, like such dataframes which are important for the analysis, then include them as well in the output_dict. For example, final result dataframe, comparison dataframe etc.
                        - Keys of the final task `output_dict` should be a meaningful like "Number of Rows". Where each word starts with an uppercase letter and words are separated by a space. For example, "Number of Rows" instead of "number_of_rows", "Max Value" instead of "max_value" etc.
                        - Make sure no repetitive data is present in the output_dict

                        ### Output Format
                        Task-1: [Precise action description]
                        Task-2: [Precise action description]
                        [...]

                        ### Quality Standards
                        - No assumptions about unavailable data
                        - No skipped or redundant steps
                        - Clear progression toward solution
                        - Complete but concise descriptions
                        - Focus on DataFrame operations only.
                        - Always maintain the Keys formation in the `output_dict` as mentioned above. First word should start with uppercase with space separated words.

                        **Provide only the task plan. Do not include any additional explanations or commentary or python code or output or any other informations**
                        """
                        ).format(user_query=user_query,df_columns=', '.join(df.columns),df_str=df.head(2).to_markdown(),df_types="\n".join([f"- **{col}**: {dtype}" for col, dtype in df.items()]))
                        
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            temperature=0,
                            top_p=0.1,
                            messages=[
                                {"role": "system", "content": task_planner_prompt},
                                {"role": "user", "content": user_query}
                            ]
                        )
                        
                        time.sleep(1)
                        status.update(label="✅ Analysis plan generated!", state="complete")
                        st.code(response.choices[0].message.content)
                        st.caption(f"Task Planner Token usage: {response.usage.total_tokens}")
                        st.caption(f"Cached Token: {response.usage.prompt_tokens_details.cached_tokens}")
                        
                        st.session_state.analysis_complete = True

                # Task Execution Phase
                if st.session_state.analysis_complete:
                    with st.container():
                        with st.status("Executing analysis...") as status:
                            task_execution_prompt = (
                            """
                            ### Task Execution System

                            You are an expert data analysis assistant with deep expertise in pandas, numpy, and data visualization. Your role is to:
                            - Transform complex data analysis tasks into precise, executable Python code
                            - Ensure all operations maintain data integrity and type safety
                            - Follow best practices for DataFrame operations and memory efficiency
                            - Create clear, professional visualizations that effectively communicate insights
                            - Generate production-ready code that adheres to Python standards
                            - Handle edge cases and potential data issues gracefully
                            - Focus on accuracy and performance in all calculations

                            Your responses will be direct code implementations without explanations, focusing purely on executing the provided task plan with optimal efficiency.

                            ### Execution Plan
                            - {df_task_plan}

                            ### Context
                            - Available DataFrame: `df`
                            - Query: {user_query}
                            - Columns: {df_columns}
                            - DataFrame Preview: {df_str}
                            - Data Types: {df_types}

                            ### Core Requirements

                            #### Data Operations
                            - All operations must use exact column names from `df_columns`
                            - Intermediate results stored as pandas DataFrames
                            - Variables must have descriptive names reflecting their content
                            - All calculations must preserve data types specified in `df_types`

                            ### Function Generation:
                            - Handle all complex DataFrame operations based on the user's query
                            - For complex operations, create a separate function to perform the required tasks efficiently
                            - Handle data operations like filtering, grouping, and aggregations accurately.
                            
                            #### Instructions for Generating Python Code:
                            - Think step by step when generating the Python code based on the user query.
                            - Import all necessary libraries at the beginning of the code.
                            - Example: import pandas as pd, import plotly.express as px.
                            - Check if each value in the column matches the expected format (e.g., datetime format or other expected patterns). Only perform operations (such as parsing or calculations) on values that match the required format, and skip or ignore any non-matching values to avoid errors.
                            - Avoid using matplotlib. For plotting, use Plotly exclusively.
                            - Interpret user queries and generate functions as needed to fulfill task requirements.
                            - Use functions like pd.to_datetime() to convert columns when necessary.
                            - Add checks or use np.divide with where or np.errstate to handle division by zero safely.
                            - Use .str.strip() to remove leading and trailing spaces before comparisons or transformations.
                            - When using lambda function make sure the code is correct and should not throw any errors.
                            - Avoid overly complex lambda functions; use named functions for clarity if the logic is complex.
                            - If for a operation a extraction of part is required from a string value then handle that carefully.
                            - For string extraction (e.g., using .str.extract()), ensure the regex pattern matches correctly and handles edge cases.
                            - Always validate data structure before unpacking to ensure operations like string splitting or regex extraction return the expected elements. Use checks or defaults to handle missing elements.
                            - Use multiple functions if required to achieve the desired result
                            - Always final output should be stored in a variable named `output_dict` with all the necessary information.

                            #### Code Standards
                            Required imports:
                            - import pandas as pd
                            - import numpy as np
                            - import plotly.express as px
                            - import plotly.graph_objects as go

                            - Each operation follows task plan sequence
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
                            - Clean, readable format
                            - No print statements unless specified
                            - No markdown or text between code blocks
                            
                            ### Response Format

                            - # Imports
                            [import statements]

                            - # Task Execution 
                            [code implementing each task]
                            Step-by-Step implementation of the task plan based on the `df_task_plan`.
                            #Task-1, #Task2... with proper task description
                            """
                        ).format(df_task_plan=response.choices[0].message.content,user_query=user_query,df_columns=', '.join(df.columns),df_str=df.head(5).to_markdown(),df_types="\n".join([f"- **{col}**: {dtype}" for col, dtype in df.items()]))
                            
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                temperature=0,
                                top_p=0.1,
                                messages=[
                                    {"role": "system", "content": task_execution_prompt},
                                    {"role": "user", "content": user_query}
                                ]
                            )

                            time.sleep(1.5) 
                            status.update(label="✅ Analysis complete!", state="complete")
                            
                            task = response.choices[0].message.content
                            task = task.replace('`', '').replace("python", "").strip()
                            
                            st.code(task, language="python")
                            st.caption(f"Task Execution Token usage: {response.usage.total_tokens}")
                            st.caption(f"Cached Token: {response.usage.prompt_tokens_details.cached_tokens}")

                            # Execute the code
                            exec_globals = {"df": df, "pd": pd, "px": px, "io": io, "np": np}
                            exec_locals = {}
                            exec(task, exec_globals, exec_locals)
                            
                            graph_visual = {}

                            if "output_dict" in exec_locals:
                                for key, value in exec_locals["output_dict"].items():
                                    if isinstance(value, pd.DataFrame):
                                            st.write(f"📈 {key}")
                                            st.dataframe(value, use_container_width=True)
                                            
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
                                        graph_visual[key] = value.to_json()
                                    else:
                                        graph_visual["fig"] = None

                            st.session_state.execution_complete = True

                #Summary Generation Phase
                if st.session_state.execution_complete:
                    with st.container():
                        with st.status("Generating insights...") as status:
                            summary_prompt = (
                            """
                            ### Data Summary Assistant

                            ### Role
                            You are an expert data summary specialist who transforms technical analysis results into clear, actionable insights. Your expertise lies in:
                            - Extracting key findings from complex data analyses
                            - Translating technical results into business-friendly language
                            - Identifying meaningful patterns and relationships
                            - Creating concise, impactful summaries
                            - Explaining data visualizations effectively

                            ### Input Materials
                            - User Query: {user_question}
                            - Analysis Code: {task}
                            - Results Dictionary: {out_df}
                            - Visualization (if present): {fig}

                            ### Summary Structure

                            ### Content Requirements

                            1. Summary Section
                            - Direct answer to user query
                            - Overview of key findings
                            - Brief context if necessary

                            2. Key Insights Section
                            - Bullet points of significant findings
                            - Relevant metrics and numbers
                            - Important trends or patterns
                            - Statistical significance where applicable
                            - Business implications when clear

                            3. Data Visualization Section (Only if fig exists)
                            - Description of visualization type
                            - Main trends or patterns shown
                            - Specific data points of interest
                            - Relationship to user's question

                            ### Quality Standards
                            - Use clear, professional language
                            - Focus on facts present in output_dict
                            - Maintain logical flow of information
                            - Avoid technical jargon unless necessary
                            - Keep insights directly relevant to query
                            - No assumptions beyond provided data
                            - No placeholders or generic statements
                            - No explanations about missing visualizations

                            ### Format Guidelines
                            - Use markdown headers for sections
                            - Keep paragraphs concise (2-3 sentences)
                            - Use bullet points for key insights
                            - Present numbers with appropriate precision
                            - Include units where relevant

                            ### Output Format

                            ### Summary
                            [Concise overview of findings]

                            ### Key Insights
                            [Bullet points of main findings]

                            ### Data Visualization
                            [Only if figure exists - visualization analysis] Other wise, remove this section. Donot include this section if no visualization is present.
                            """
                            ).format(user_question=user_query,task=task,out_df=exec_locals["output_dict"],fig=str(graph_visual))
                            
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                temperature=0.1,
                                top_p=0.1,
                                messages=[
                                    {"role": "system", "content": summary_prompt},
                                    {"role": "user", "content": user_query}
                                ]
                            )

                            time.sleep(1)
                            status.update(label="✅ Insights generated!", state="complete")
                            
                            st.subheader("🎯 Key Insights")
                            st.markdown(response.choices[0].message.content)
                            st.caption(f"Summary Token usage: {response.usage.total_tokens}")
                            st.caption(f"Cached Token: {response.usage.prompt_tokens_details.cached_tokens}")
                            st.session_state.summary_complete = True

                #Question Generation Phase
                if st.session_state.summary_complete:
                    with st.container():
                        with st.status("Generating Questions...") as status:
                            follow_up_questions_prompt = (
                            """
                            # Follow-up Question Generator

                            ## Role
                            You are a precise data insight explorer who:
                            - Generates follow-up questions based strictly on available data
                            - Ensures questions are directly answerable using the dataset
                            - Maintains focus on business value and actionable insights
                            - Avoids assumptions or speculation beyond the data

                            ## Input
                            - Current Question: {user_question}
                            - Available Columns: {df_columns}
                            - Data Sample: {df_head}

                            ## Core Requirements

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

                            ## Output Format

                            1. [Precise question using available data] - [Brief business context and value]

                            2. [Precise question using available data] - [Brief business context and value]

                            3. [Precise question using available data] - [Brief business context and value]
                            """
                            ).format(user_question=user_query,df_columns=', '.join(df.columns),df_head=df.head(1).to_markdown())
                            
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                temperature=0.1,
                                top_p=0.1,
                                messages=[
                                    {"role": "system", "content": follow_up_questions_prompt},
                                    {"role": "user", "content": user_query}
                                ]
                            )

                            time.sleep(1)
                            status.update(label="✅ Questions generated!", state="complete")
                            
                            st.subheader("🔍 Follow-up Questions")
                            st.markdown(response.choices[0].message.content)
                            st.caption(f"Questions Token usage: {response.usage.total_tokens}")
                            st.caption(f"Cached Token: {response.usage.prompt_tokens_details.cached_tokens}")
                            st.session_state.question_complete = True
                   

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
