import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from flask import Flask, render_template

# Load the crime data
data = pd.read_csv("data/rizalwitht.csv", encoding='latin1')

# Convert 'DATE COMMITTED' to datetime format
data['DATE COMMITTED'] = pd.to_datetime(data['DATE COMMITTED'], errors='coerce')

# Remove rows with missing LATITUDE or LONGITUDE
data = data.dropna(subset=['LATITUDE', 'LONGITUDE', 'DATE COMMITTED'])

# Perform KMeans clustering based on LATITUDE and LONGITUDE
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['LATITUDE', 'LONGITUDE']])

# Extract year, month, day for time-based filtering
data['Year'] = data['DATE COMMITTED'].dt.year
data['Month'] = data['DATE COMMITTED'].dt.month
data['Day'] = data['DATE COMMITTED'].dt.day

# Create a Flask instance (Dash uses this under the hood)
server = Flask(__name__)

# Initialize the Dash app
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)

# Layout of the app (Dash will render here)
app.layout = html.Div([
    html.Div([
        html.H1("Crime Data Analysis Dashboard", style={'textAlign': 'center'}),
        html.P("Explore crime patterns and clusters using data visualizations.",
               style={'textAlign': 'center', 'color': 'gray'})
    ], style={'padding': '20px', 'backgroundColor': '#f4f4f8'}),

    # Insert other Dash components (graphs, dropdowns, etc.)
])

# Serve the custom index.html
@server.route('/')
def home():
    return render_template('index.html')

# Callback and logic for interactivity

if __name__ == '__main__':
    app.run_server(debug=True)
