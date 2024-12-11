import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import base64
import io
import os

# Initialize Dash app
app = dash.Dash(__name__)

# Create 'uploaded_files' directory if it doesn't exist
if not os.path.exists('uploaded_files'):
    os.makedirs('uploaded_files')

# Default data loading (initial state before any file upload)
data = pd.read_csv("rizalwitht.csv", encoding='latin1')
data['DATE COMMITTED'] = pd.to_datetime(data['DATE COMMITTED'], errors='coerce')
data = data.dropna(subset=['LATITUDE', 'LONGITUDE', 'DATE COMMITTED'])
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['LATITUDE', 'LONGITUDE']])
data['Year'] = data['DATE COMMITTED'].dt.year
data['Month'] = data['DATE COMMITTED'].dt.month
data['Day'] = data['DATE COMMITTED'].dt.day

# App Layout
app.layout = html.Div([
    html.Div([
        html.H1("Crime Data Analysis Dashboard", style={'textAlign': 'center'}),
        html.P("Explore crime patterns and clusters using data visualizations.",
               style={'textAlign': 'center', 'color': 'gray'})
    ], style={'padding': '20px', 'backgroundColor': '#f4f4f8'}),

    # Filters Section
    html.Div([
        html.Div([
            html.Label("Select Cluster:"),
            dcc.Dropdown(
                id='cluster-filter',
                options=[{'label': f'Cluster {i}', 'value': i} for i in sorted(data['Cluster'].unique())],
                value=None,
                placeholder="Select a cluster",
                multi=True
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("Crime Type:"),
            dcc.Dropdown(
                id='crime-type-filter',
                options=[{'label': crime, 'value': crime} for crime in data['INCIDENT TYPE'].unique()],
                value=None,
                placeholder="Select crime types",
                multi=True
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("Date Range:"),
            dcc.DatePickerRange(
                id='date-range-filter',
                start_date=data['DATE COMMITTED'].min().date(),
                end_date=data['DATE COMMITTED'].max().date(),
                display_format='YYYY-MM-DD'
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'})
    ], style={'borderBottom': '2px solid #ccc', 'padding': '20px'}),

    # Visualizations Section
    html.Div([
        html.Div([ 
            dcc.Graph(id='map-view')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            dcc.Graph(id='crime-type-bar')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
    ]),

    html.Div([
        html.Div([ 
            dcc.Graph(id='time-trend')
        ], style={'width': '100%', 'padding': '10px'})
    ]),

    # Summary Metrics Section
    html.Div([
        html.H4("Summary Metrics", style={'textAlign': 'center'}),
        html.Div([
            html.Div([ 
                html.H5("Total Crimes"),
                html.P(id='total-crimes', style={'fontSize': '20px', 'fontWeight': 'bold'})
            ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([ 
                html.H5("Most Common Crime"),
                html.P(id='common-crime', style={'fontSize': '20px', 'fontWeight': 'bold'})
            ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([ 
                html.H5("Most Affected Area"),
                html.P(id='affected-area', style={'fontSize': '20px', 'fontWeight': 'bold'})
            ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center'})
        ])
    ], style={'backgroundColor': '#f9f9f9', 'padding': '20px', 'marginTop': '20px'}),

    # Updated File upload section
    html.Div([
        html.Label("Upload CSV or Excel File:", style={'fontWeight': 'bold'}),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                html.Button('Upload File', style={'backgroundColor': '#4CAF50', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'cursor': 'pointer'}),
                html.P("Drag and drop or click to select a file", style={'marginTop': '10px'})
            ]),
            multiple=False
        ),
        html.Div(id='output-data-upload'),
    ], style={'padding': '20px', 'textAlign': 'center'}),

    # Footer Section
    html.Div([
        html.P("Data Source: XYZ Crime Records | Dashboard by Your Name",
               style={'textAlign': 'center', 'color': 'gray'})
    ], style={'borderTop': '2px solid #ccc', 'padding': '10px', 'marginTop': '20px'})
])

# Callback to handle file upload and save it in 'uploaded_files' directory
@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def save_uploaded_file(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        try:
            if filename.endswith('.csv'):
                try:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                except UnicodeDecodeError:
                    df = pd.read_csv(io.StringIO(decoded.decode('latin1')))
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                return html.Div([html.H5("Invalid file type. Please upload a CSV or XLSX file.")])

            file_path = os.path.join('uploaded_files', f"uploaded_{filename}")
            df.to_csv(file_path, index=False)

            return html.Div([
                html.H5(f"File {filename} uploaded successfully."),
                html.Hr(),
                html.Div(f"File saved as {file_path}")
            ])
        except Exception as e:
            return html.Div([
                html.H5("There was an error processing this file."),
                html.Div(f"Error: {str(e)}")
            ])
    return html.Div("No file uploaded yet.")

# Callbacks for Interactive Filtering
@app.callback(
    [Output('map-view', 'figure'),
     Output('crime-type-bar', 'figure'),
     Output('time-trend', 'figure'),
     Output('total-crimes', 'children'),
     Output('common-crime', 'children'),
     Output('affected-area', 'children')],
    [Input('cluster-filter', 'value'),
     Input('crime-type-filter', 'value'),
     Input('date-range-filter', 'start_date'),
     Input('date-range-filter', 'end_date'),
     Input('crime-type-bar', 'clickData')]
)
def update_visuals(selected_clusters, selected_crime_types, start_date, end_date, bar_click):
    filtered_data = data.copy()
    if selected_clusters:
        filtered_data = filtered_data[filtered_data['Cluster'].isin(selected_clusters)]
    if selected_crime_types:
        filtered_data = filtered_data[filtered_data['INCIDENT TYPE'].isin(selected_crime_types)]
    if start_date and end_date:
        filtered_data = filtered_data[(filtered_data['DATE COMMITTED'] >= start_date) & (filtered_data['DATE COMMITTED'] <= end_date)]

    if bar_click:
        selected_crime = bar_click['points'][0]['y']
        filtered_data = filtered_data[filtered_data['INCIDENT TYPE'] == selected_crime]

    map_fig = px.scatter_mapbox(
        filtered_data, lat='LATITUDE', lon='LONGITUDE', color='Cluster',
        mapbox_style="carto-positron", zoom=10, title="Crime Clusters"
    )

    top_n = 10
    top_crimes = filtered_data.groupby('INCIDENT TYPE').size().nlargest(top_n).reset_index(name='Count')

    bar_fig = px.bar(
        top_crimes,
        y='INCIDENT TYPE', x='Count', 
        title=f"Top {top_n} Crime Type Distribution",
        orientation='h'
    )

    trend_fig = px.line(
        filtered_data.groupby('DATE COMMITTED').size().reset_index(name='Count'),
        x='DATE COMMITTED', y='Count', title="Crime Trends Over Time"
    )

    total_crimes = len(filtered_data)
    common_crime = filtered_data['INCIDENT TYPE'].mode()[0] if not filtered_data.empty else "N/A"
    affected_area = filtered_data['BARANGAY'].mode()[0] if not filtered_data.empty else "N/A"

    return map_fig, bar_fig, trend_fig, total_crimes, common_crime, affected_area

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
