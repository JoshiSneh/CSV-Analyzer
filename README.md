# CSV Analyzer 📊

An intelligent CSV file analysis tool powered by OpenAI GPT models and Streamlit. Ask questions about your data in natural language and get instant insights, visualizations, and analysis.

## Features 🌟

- **Natural Language Queries**: Ask questions about your data in plain English
- **Automated Analysis**: Get instant insights and visualizations
- **Interactive Visualizations**: Dynamic charts and graphs using Plotly
- **Step-by-Step Analysis**: Watch the analysis process unfold in real-time
- **Downloadable Results**: Export processed data and analysis results
- **Smart Summaries**: AI-generated insights and key findings

## Installation 🚀

1. Clone the repository:
```bash
[git clone https://github.com/yourusername/smart-csv-analyzer.git](https://github.com/JoshiSneh/CSV-Analyzer)
cd scsv-analyzer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Requirements 📋

Create a `requirements.txt` file with the following dependencies:

```
streamlit
pandas
plotly
python-dotenv
openai
```

## Setup ⚙️

1. Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/account/api-keys)

2. Run the application:
```bash
streamlit run app_streamlit_new.py
```

## Usage 💡

1. Launch the application
2. Enter your OpenAI API key in the sidebar
3. Upload a CSV file using the file uploader
4. Enter your question about the data in natural language
5. Click "Analyze Data" to start the analysis
6. View the results in three phases:
   - Analysis Planning
   - Execution
   - Summary and Insights
   - 
## Application Structure 🏗️

```
smart-csv-analyzer/
├── app_streamlit_new.py # Main application file
├── requirements.txt    # Project dependencies
└── README.md          # Documentation
```

## Key Components 🔧

1. **Data Upload**: Supports CSV file upload with preview functionality
2. **Query Processing**: Natural language processing using OpenAI GPT models
3. **Analysis Pipeline**:
   - Task Planning: Breaks down user query into actionable steps
   - Task Execution: Performs the analysis using pandas and plotly
   - Summary Generation: Creates human-readable insights
4. **Visualization**: Interactive charts and graphs using Plotly

## Environment Variables 🔐

Create a `.env` file with the following variables (optional):
```
OPENAI_API_KEY=your_api_key_here
```
## Security Note 🔒

- The application requires an OpenAI API key
- API keys are stored only in session state and are not permanently saved
- Use environment variables for API keys in production

## Limitations ⚠️

- Limited to CSV file format
- Analysis complexity depends on OpenAI API response
- Processing time may vary based on data size and query complexity

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [OpenAI](https://openai.com/)
- Uses [Plotly](https://plotly.com/) for visualizations
- Pandas for data manipulation

---

