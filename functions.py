import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def fix_column_names(df):
    df.columns = (df.columns
            .str.replace(r'[\s\n\-’\'\"\‘\’\“\”\[\]\(\)\{\}\<\>\&\+]+', '_', regex=True)
            .str.replace(r'_{2,}', '_', regex=True) 
            .str.strip('_')
            .str.lower())
    


# streamlit
import streamlit as st

def navigation():
    """
    Function to customize navigation sidebar panel
    """
    st.sidebar.page_link("streamlit_app.py", label='👋 Welcome')
    st.sidebar.page_link("pages/01_external.py", label="📊 External")


    