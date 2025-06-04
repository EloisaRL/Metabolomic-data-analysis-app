# pages/multi_study_analysis.py
from memory_profiler import memory_usage
import os
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback, State, no_update


from .multi_study_analysis_page_tabs.upset_plots import layout    as upset_layout
from .multi_study_analysis_page_tabs.upset_plots import register_callbacks as register_upset_cb
from .multi_study_analysis_page_tabs.netowrk_plots     import layout    as network_layout
from .multi_study_analysis_page_tabs.netowrk_plots     import register_callbacks as register_network_cb


# Helper function to list project folders in the "Projects" folder.
def list_projects():
    projects_dir = "Projects"
    try:
        # List only directories
        return sorted([f for f in os.listdir(projects_dir) if os.path.isdir(os.path.join(projects_dir, f))])
    except FileNotFoundError:
        return []
    
# Helper function to list files in processed-datasets folder.
def list_processed_files(selected_project):
    # Modify the base path as needed for your project structure.
    folder_path = os.path.join("projects", selected_project, "processed-datasets")
    try:
        files = os.listdir(folder_path)
        # Only include files (exclude subdirectories)
        files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
        return files
    except Exception as e:
        print(f"Error listing files in {folder_path}: {e}")
        return []
    

# Define the modal popup that asks the user to "Choose a Project"
project_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Choose a Project")),
        dbc.ModalBody(
            dcc.Dropdown(
                id="project-dropdown-pop-msa",
                options=[
                    {'label': project.replace("-", " "), 'value': project} 
                    for project in list_projects()
                ],
                placeholder="Select a project",
                clearable=False,
                style={"width": "100%"}
            )
        ),
        dbc.ModalFooter(
            dbc.Button("Confirm", id="confirm-project-button-msa", n_clicks=0, color="primary")
        ),
    ],
    id="project-modal-msa",
    is_open=True,  # Open by default so the popup appears on page load.
    backdrop="static",  # Prevent clicking outside to close.
    centered=True
)

layout = html.Div([
    project_modal,
    dcc.Store(id="svg-store", storage_type="memory"),
    # Header that shows the selected project title.
    html.Div(
        [
            html.Div(
                id="selected-project-title-msa",
                style={
                    "fontSize": "1.75rem",
                    "textAlign": "left"
                }
            )
        ],
        style={"margin": "1rem"}
    ),
    # Main content: sidebar and tabs in a flex container.
    html.Div(
        [
            # Sidebar (left column): shows the processed files as a checklist.
            html.Div(
                [
                    html.H4("Processed Files"),
                    dcc.Checklist(
                        id="project-files-checklist-msa",
                        options=[],  # Options updated via a callback.
                        # Each label is forced to stay on one line.
                        labelStyle={'display': 'block', 'whiteSpace': 'nowrap'},
                        inputStyle={"margin-right": "10px"}
                    ),
                    dbc.Button(
                        "Process data",
                        id="process-data-button-msa",
                        color="primary",   # Blue color (Bootstrap primary)
                        size="lg",         # Large size
                        style={"marginTop": "20px", "width": "100%"}
                    )
                ],
                style={
                    "width": "20%",
                    "padding": "1rem",
                    "borderRight": "1px solid #ccc",
                    "overflowX": "auto"  # Enables horizontal scrolling if content is wider than the container.
                }
            ),
            # Main content (right column): Tabs that are visible regardless of sidebar.
            html.Div(
                [
                    dcc.Tabs(
                        id="multi-study-analysis-tabs",
                        value="upset_plots",
                        children=[
                            dcc.Tab(
                                label="Upset plots",
                                value="upset_plots",
                                children=upset_layout
                            ),
                            dcc.Tab(
                                label="Network graphs",
                                value="network-graphs",
                                children=network_layout
                            ),
                        ]
                    ),
                    html.Div(id="multi-analysis-content", style={"marginTop": "20px"})
                ],
                style={"width": "80%", "padding": "1rem"}
            )
        ],
        style={"display": "flex"}
    )
])

register_upset_cb()
register_network_cb()

@callback(
    [Output("selected-project-title-msa", "children"),
     Output("project-modal-msa", "is_open"),
     Output("project-dropdown-pop-msa", "value")],
    Input("confirm-project-button-msa", "n_clicks"),
    State("project-dropdown-pop-msa", "value")
)
def update_project_info(n_clicks, selected_project):
    if n_clicks and selected_project:
        display_project = selected_project.replace("-", " ")
        project_title = html.Div([
            display_project
        ])
        # Close modal (is_open False) and return the selected project value.
        return project_title, False, selected_project
    # If nothing is selected or the button isn't clicked, leave everything unchanged.
    return no_update, True, no_update


# Callback to update the checklist in the sidebar with the files from processed-datasets.
@callback(
    Output("project-files-checklist-msa", "options"),
    Input("project-dropdown-pop-msa", "value")
)
def update_files_checklist(selected_project):
    if selected_project:
        files = list_processed_files(selected_project)
        options = []
        for f in files:
            display_name = f
            # Remove leading 'processed_' if present.
            if display_name.startswith("processed_"):
                display_name = display_name[len("processed_"):]
            # Remove trailing '.csv' if present.
            if display_name.endswith(".csv"):
                display_name = display_name[:-len(".csv")]
            options.append({'label': display_name, 'value': f})
        return options
    return []

