import pandas as pd

def load_data(uploaded_file):
    """Load data from uploaded file."""
    if uploaded_file.type == "text/csv":
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)