# pages/single_study_analysis.py
import os
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback, State, no_update

from .single_study_analysis_page_tabs.differential_metabolites import layout    as differential_layout
from .single_study_analysis_page_tabs.differential_metabolites import register_callbacks as register_diff_cb
from .single_study_analysis_page_tabs.differential_pathways     import layout    as pathway_layout
from .single_study_analysis_page_tabs.differential_pathways     import register_callbacks as register_path_cb


# Helper function to list project folders in the "Projects" folder.
def list_projects():
    projects_dir = "Projects"
    try:
        # List only directories
        return sorted([f for f in os.listdir(projects_dir) if os.path.isdir(os.path.join(projects_dir, f))])
    except FileNotFoundError:
        return []

# Define the modal popup that asks the user to "Choose a Project"
project_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Choose a Project")),
        dbc.ModalBody(
            dcc.Dropdown(
                id="project-dropdown-pop",
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
            dbc.Button("Confirm", id="confirm-project-button", n_clicks=0, color="primary")
        ),
    ],
    id="project-modal",
    is_open=True,  # Open by default so the popup appears on page load.
    backdrop="static",  # Prevent clicking outside to close.
    centered=True
)

layout = html.Div([
    project_modal,
    # A container with flex styling that holds the files dropdown and the project title inline.
    html.Div(
        [
            # Project name on the left.
            html.Div(
                id="selected-project-title",
                style={
                    "fontSize": "1.75rem",  # Larger than the study title.
                    "textAlign": "left",
                    "display": "inline-block"
                }
            ),
            # Files dropdown on the right.
            dcc.Dropdown(
                id="project-files-dropdown",
                options=[],  # Will be populated based on the selected project.
                placeholder="Select a file",
                clearable=False,
                style={
                    "width": "300px",
                    "marginLeft": "auto"  # Pushes the dropdown to the right.
                }
            )
        ],
        style={
            "margin": "1rem",
            "display": "flex",
            "alignItems": "center"
        }
    ),
    html.Div(
        id="selected-study-title",
        style={
            "fontSize": "1.25rem",
            "textAlign": "left",       # Aligns the text to the left
            "marginTop": "1rem",
            "overflowX": "auto",       # Enables horizontal scrolling if needed
            "whiteSpace": "nowrap",    # Prevents the text from wrapping onto multiple lines
            "padding": "0 1rem"        # Optional: adds some padding for visual comfort
        }
    ),
    
    dcc.Tabs(
        id="single-study-analysis-tabs",
        value="differential",
        children=[
            dcc.Tab(
                label="Differential metabolite analysis",
                value="differential",
                children=differential_layout
            ),
            dcc.Tab(
                label="Differential pathway analysis",
                value="pathway",
                children=pathway_layout
            )
        ]
    ),
    html.Div(id="analysis-content", style={"marginTop": "20px"}),
    # IMPORTANT: include the modal (and its hidden stores) here!
    
])


register_diff_cb()
register_path_cb()

#-------------------------------------------
@callback(
    [Output("selected-project-title", "children"),
     Output("project-modal", "is_open"),
     Output("project-dropdown-pop", "value")],
    Input("confirm-project-button", "n_clicks"),
    State("project-dropdown-pop", "value")
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


@callback(
    Output("selected-study-title", "children"),
    Input("project-files-dropdown", "value")
)
def update_selected_study_title(selected_file):
    if selected_file:
        # Remove leading "processed_" if present.
        display_study = selected_file
        if display_study.startswith("processed_"):
            display_study = display_study[len("processed_"):]
        # Remove trailing ".csv" if present.
        if display_study.endswith(".csv"):
            display_study = display_study[:-len(".csv")]
        return html.Div([
            html.Strong("Selected Study: "),
            display_study
        ])
    return ""


# Callback to update the file list dropdown based on the selected project.
@callback(
    Output("project-files-dropdown", "options"),
    Input("project-dropdown-pop", "value")
)
def update_project_files(selected_project):
    if not selected_project:
        return []
    
    # Construct the path to the processed-datasets folder in the selected project.
    base_path = os.path.join("Projects", selected_project, "processed-datasets")
    
    # If the folder does not exist, return an empty options list.
    if not os.path.exists(base_path):
        return []
    
    # List only file names from the folder.
    file_names = sorted([
        f for f in os.listdir(base_path) 
        if os.path.isfile(os.path.join(base_path, f))
    ])
    
    # Create options for the dropdown.
    # Each option's label is rendered as an html.Span with a title attribute.
    options = [
        {
            "label": html.Span(
                f,
                title=f,  # Full file name shown when hovering.
                style={
                    "display": "inline-block",
                    "maxWidth": "280px",    # Adjust maximum width as needed.
                    "overflow": "hidden",
                    "textOverflow": "ellipsis"
                }
            ),
            "value": f
        } for f in file_names
    ]
    return options






