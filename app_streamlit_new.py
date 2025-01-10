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
                            # Specialized Task Planning Agent for DataFrame Analysis  

                            ---

                            ## **Core Role**  
                            Your role is to **develop precise, structured, and executable task plans and sub-task plans** tailored for analyzing DataFrames, adhering to the outlined standards and requirements.  

                            ---

                            ## **Core Capabilities**  
                            You are a **specialized agent for task planning** with expertise in:  
                            - Creating detailed and executable task structures.  
                            - Addressing edge cases and ensuring error-free operations.  
                            - Generating actionable insights while preserving data integrity and ensuring high-quality visualizations.  

                            ---

                            ## **Primary Requirements**  

                            ### **1. Task Structuring**  
                            - Define each task with specific, actionable steps.  
                            - Break tasks into logical, detailed sub-tasks with a clear progression for it's Parent Task.  
                            - Operate exclusively on available DataFrame columns and functionalities.  
                            - Address edge cases (e.g., missing values, incorrect data types) comprehensively.  
                            - Ensure all outputs maintain the **DataFrame format** unless explicitly requested as lists, in which case convert lists into dictionary formats.  
                            - **Final DataFrame outputs must always remain in DataFrame format.**  

                            ### **2. Data Handling Standards**  
                            - Use descriptive variable names for clarity and traceability.  
                            - Validate column operations against data types and handle missing or null values appropriately.  
                            - Avoid runtime errors by implementing robust error handling.  
                            - Maintain appropriate data types for all intermediate and final results.  

                            ### **3. Visualization Requirements**  
                            - Exclusively use **Plotly** for all visualizations.  
                            - Validate column selections for axes and chart parameters based on available data.  
                            - Store visualizations in variables prefixed with `fig_`.  
                            - Ensure clarity and relevance in all charts and graphs.  

                            ### **4. Output Structuring**  
                            - Create a final `output_dict` structured with:  
                            - **Descriptive keys** (e.g., "Number of Rows").  
                            - Visualization objects stored as variables prefixed with `fig_`.  
                            - Relevant analysis DataFrames, clearly distinguished.   
                            - All the keys should be capitalize each word, use spaces between words. Ensure descriptions accurately reflect content (e.g., "Maximum Value" instead of "max_val").
                            
                            ---

                            ## **Quality Standards**  

                            ### **1. Data Integrity**  
                            - Operate only on available columns and data.  
                            - Avoid assumptions about schema or external data.  
                            - Fully validate all operations for correctness.  

                            ### **2. Process Quality**  
                            - Avoid skipped or redundant steps.  
                            - Maintain a clear, logical progression toward the solution.  
                            - Focus exclusively on DataFrame operations for processing.  
                            - Implement robust error handling for all stages.  

                            ---

                            ## **Style, Tone, and Audience**  

                            - **Style:** Task Planner Expert.  
                            - **Tone:** Professional, Technical.  
                            - **Audience:** Data Analysts and Data Scientists.  

                            ---

                            ## **Output Structure and Format**  

                            ### **Tasks and Sub-Tasks**  
                            **Task-1:** [Precise action description]  
                               - **Sub-Task-1.1:** [Detailed step of Task-1]  
                               - **Sub-Task-1.2:** [Detailed step Of Task-1]  
                            **Task-2:** [Precise action description]  
                               - **Sub-Task-2.1:** [Detailed step of Task-2]  
                               - **Sub-Task-2.2:** [Detailed step of Task-2]  
                              [...]
                            
                            ## **Input Parameters**  

                            - **Available DataFrame:** `df`  
                            - **User Query:** `{user_query}`  
                            - **Available Columns:** `{df_columns}`   
                            - **Data Frame Preview with column types:** `{df_str}`  
                            ---

                            ### **Output Guidelines**  
                            - Provide only the **task plan**—no code, outputs, or additional commentary or extra informations. 
                            ---
                        """
                        ).format(user_query=user_query,df_columns=', '.join(df.columns),df_str="\n".join([f"| {col} | {dtype} |" for col, dtype in df.items()]))
                        
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
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
                            # Task Execution System

                            ---

                            ## **Role**
                            You are an expert **data analysis assistant** with deep expertise in:
                            - **Pandas**, **NumPy**, and **Plotly** for data visualization.
                            - **Transforming complex tasks** into precise, executable Python code.
                            - Ensuring all operations maintain **data integrity** and **type safety**.
                            - Following **best practices** for DataFrame operations and memory efficiency.
                            - Creating clear, professional **visualizations** that effectively communicate insights.
                            - Generating **production-ready code** adhering to Python standards.
                            - Handling edge cases and potential data issues gracefully.
                            - Focusing on **accuracy** and **performance** in all calculations.

                            Your responses will be **direct Python code implementations** without explanations. Focus purely on executing the provided task plan with optimal efficiency.  
                            
                            ---
                            
                            ## **Execution Plan**
                            - `{df_task_plan}`

                            ---

                            ## **Context**
                            - **Available DataFrame**: `df`
                            - **Query**: `{user_query}`
                            - **Available Columns**: `{df_columns}`
                            - **DataFrame Preview with Column Types**: {df_str}
                            
                            ---
                            
                            ## **Core Requirements**

                            ### **1. Data Operations**
                            - Use **exact column names** from `df_columns` for all operations.
                            - Intermediate results stored as pandas DataFrames
                            - Use **descriptive variable names** that reflect their content.
                            - Preserve data types as specified in `df_types`.

                            ### **2. Function Generation**
                            - Handle complex DataFrame operations with **separate functions** for clarity and efficiency.
                            - Implement accurate operations like **filtering**, **grouping**, and **aggregations**.
                            - Validate data types and handle all edge cases for reliable results.

                            ---

                            ## **Instructions for Generating Python Code**

                            - **General Guidelines**:
                            - Think step-by-step when generating Python code based on the task plan.
                            - Import all necessary libraries at the beginning:
                            - `import pandas as pd`
                            - `import numpy as np`
                            - `import plotly.express as px`
                            - `import plotly.graph_objects as go`
                            - Avoid using deprecated pandas methods.
                            - Use consistent variable naming.
                            - Donot use `fig.show()` during visualization. Just return the `fig`.

                            - **Error Handling**:
                            - Add checks for **data structure validation** before operations like unpacking or splitting.
                            - Use functions like `pd.to_datetime()` for column type conversions.
                            - Use `.str.strip()` to remove leading/trailing spaces before transformations or comparisons.
                            - Handle division by zero safely using `np.divide` with `where` or `np.errstate`.

                            - **Code Clarity**:
                            - Avoid overly complex lambda functions; use **named functions** for clarity if logic is complex.
                            - For operations like **string extraction**, use `str.extract()` with well-tested regex patterns that handle edge cases.

                            - **Final Output**:
                            - Store the final result in a variable named **`output_dict`** with all necessary information.

                            ---

                            ## **Visualization Standards**
                            - Use **Plotly** exclusively for all visualizations.
                            - Ensure:
                            - Proper figure sizing and formatting.
                            - Clear labels, titles, and legends.
                            - Interactive elements when relevant.
                            - Appropriate color schemes.

                            ---

                            ## **Code Standards**
                            - Follow task plan sequence step-by-step.
                            - No print statements unless explicitly specified.
                            - Clean, readable format with proper indentation.
                            - Use only necessary imports:
                            - `import pandas as pd`
                            - `import numpy as np`
                            - `import plotly.express as px`
                            - `import plotly.graph_objects as go`

                            ---

                            ## **Output Requirements**
                            - Provide **Python code only** with no explanations or markdown.
                            - Implement each step from the task plan step-by-step donot miss anything.
                            - Use comments to separate **Task-1**, **Task-2**, etc., with brief task descriptions.
                            - Store final output in **`output_dict`** donot make it a list of dictionary.
                            - Last and the final output should be of a **dictionary type**.

                            ---

                            ## **Style, Tone, and Audience**  

                            - **Style:** Python Expert.  
                            - **Tone:** Professional, Technical.  
                            - **Audience:** Data Analysts and Data Scientists.  

                            ---

                            ## **Response Format**

                            # Imports
                            [import statements]

                            # Task Execution
                            [Python code implementing each task]
                            # Task-1: [Description]
                            [Code for Task-1]

                            # Task-2: [Description]
                            [Code for Task-2]
                            # ... Additional tasks as required
                            """
                        ).format(df_task_plan=response.choices[0].message.content,user_query=user_query,df_columns=', '.join(df.columns),df_str="\n".join([f"| {col} | {dtype} |" for col, dtype in df.items()]))
                            
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
                            - Results Dictionary with or withour visualization `fig`: {out_df}
                            
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
                            ).format(user_question=user_query,task=task,out_df=exec_locals["output_dict"])
                            
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
