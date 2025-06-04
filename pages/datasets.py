import os
import dash
import pandas as pd
from datetime import datetime
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
from app import app

# Base folder for projects.
projects_folder = "Projects"

# Helper to get subfolders of any folder.
def list_subfolders(folder):
    try:
        return sorted([f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))])
    except FileNotFoundError:
        return []

layout = html.Div([
    html.Div(
        dcc.Dropdown(
            id='project-dropdown',
            options=[{'label': 'Select project', 'value': ''}] +
                    [{'label': proj, 'value': proj} for proj in list_subfolders(projects_folder)],
            value='',
            placeholder='Select project',
            clearable=False,
            style={'width': '250px'}
        ),
        style={'display': 'flex', 'justifyContent': 'flex-end', 'marginBottom': '20px'}
    ),
    html.Div(id="processed-datasets-table", style={"marginRight": "50px"}),
])

@app.callback(
    Output("processed-datasets-table", "children"),
    [Input("project-dropdown", "value"), Input("url", "pathname")]
)
def show_processed_datasets(selected_project, pathname):
    """
    Lists files in the processed-datasets folder within the selected project.
    """
    # If no project has been selected, show an instructive message.
    if not selected_project:
        return html.Div("Select a project to show processed datasets", style={"fontFamily": "Arial"})

    # Build the base path for processed datasets in the selected project.
    base_path = os.path.join(projects_folder, selected_project, "Processed-datasets")

    if not os.path.exists(base_path):
        return html.Div(f"No '{base_path}' folder found.", style={"fontFamily": "Arial"})

    files = os.listdir(base_path)
    if not files:
        return html.Div(f"No files found in '{base_path}'.", style={"fontFamily": "Arial"})

    file_records = []
    for f in sorted(files):
        file_path = os.path.join(base_path, f)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(f)[1].lower()
        dataset_type = ext.replace('.', '').upper() if ext in ['.csv', '.tsv', '.parquet'] else 'UNKNOWN'
        rows, cols = 0, 0
        try:
            if ext == '.csv':
                df_temp = pd.read_csv(file_path)
                rows, cols = df_temp.shape
            elif ext == '.tsv':
                df_temp = pd.read_csv(file_path, sep='\t')
                rows, cols = df_temp.shape
            elif ext == '.parquet':
                df_temp = pd.read_parquet(file_path)
                rows, cols = df_temp.shape
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        last_updated = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%d/%m/%Y %I:%M %p")
        file_records.append({
            "Name": f,
            "Dataset type": dataset_type,
            "Cells (Columns x Rows)": f"{cols} x {rows}",
            "Last updated": last_updated
        })

    # Update title text: just use the project name with hyphens replaced by spaces.
    title_text = selected_project.replace("-", " ")
    header = html.H3(title_text, style={"fontFamily": "Arial"})

    columns = [
        {"name": col, "id": col} for col in ["Name", "Dataset type", "Cells (Columns x Rows)", "Last updated"]
    ]

    table = dash_table.DataTable(
        data=file_records,
        columns=columns,
        page_size=10,
        style_table={
            "overflowX": "auto",
            "marginRight": "50px",
            "border": "1px solid #ccc",
            "borderRadius": "5px",
            "boxShadow": "2px 2px 5px rgba(0, 0, 0, 0.1)"
        },
        style_header={
            "backgroundColor": "#f2f2f2",
            "fontFamily": "Arial",
            "fontSize": "16px",
            "fontWeight": "bold",
            "textAlign": "left",
            "border": "1px solid #ddd",
            "padding": "10px"
        },
        style_cell={
            "fontFamily": "Arial",
            "fontSize": "14px",
            "textOverflow": "ellipsis",
            "whiteSpace": "nowrap",
            "overflow": "hidden",
            "textAlign": "left",
            "border": "1px solid #ddd",
            "padding": "10px"
        },
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"}],
        style_as_list_view=True
    )

    return html.Div([header, table])
