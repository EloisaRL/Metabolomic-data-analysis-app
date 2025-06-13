# pages/multi_study_analysis.py
import os
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback, State, no_update, callback_context
import os, base64
import plotly.io as pio

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
    dbc.Toast(
        id="save-toast-msa",
        header="",
        icon="",
        duration=3000,
        is_open=False,
        dismissable=True,
        style={
            "position": "fixed",
            "top": 10,
            "right": 10,
            "width": 250,
            "zIndex": 9999,
        },
        children="",
    ),
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

@callback(
    [
        Output("save-toast-msa", "is_open"),
        Output("save-toast-msa", "children"),
        Output("save-toast-msa", "header"),
        Output("save-toast-msa", "icon"),
    ],
    [
        Input("confirm-save-plot-button-upset-msa", "n_clicks"),
        Input("confirm-save-plot-button-diff-msa",  "n_clicks"),
    ],
    [
        State("project-dropdown-pop-msa",      "value"),
        State("plot-name-input-upset-msa",    "value"),
        State("upset-plot-store-msa",         "data"),
        State("plot-name-input-diff-msa",     "value"),
        State("diff-plot-store-msa",          "data"),
    ],
    prevent_initial_call=True
)
def show_save_toast_msa(n_upset, n_diff,
                        project,
                        upset_fn, upset_payload,
                        diff_fn,  diff_payload):
    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    def toast(open_, msg, hdr, icn):
        return open_, msg, hdr, icn

    def mk_msg(kind, name=None, relpath=None):
        if kind == "no_proj":
            return "Select a project before saving.", "Warning", "warning"
        if kind == "no_name":
            return "Enter a name before saving.",    "Warning", "warning"
        if kind == "no_data":
            return "No plot data available to save.", "Error",   "danger"
        # success
        if kind == "upset":
            return (
                f"Upset plot '{name}.svg' saved in '{relpath}'.",
                "Success",
                "success"
            )
        if kind == "diff":
            return (
                f"Differential plot '{name}.svg' saved in '{relpath}'.",
                "Success",
                "success"
            )

    # Common base directory
    base = os.path.join(
        "Projects", project or "",
        "Plots", "Multi-study-analysis"
    )

    # Validate project
    if trigger in (
        "confirm-save-plot-button-upset-msa",
        "confirm-save-plot-button-diff-msa"
    ) and not project:
        msg, hdr, icn = mk_msg("no_proj")
        return toast(True, msg, hdr, icn)

    # 1) Upset plot
    if trigger == "confirm-save-plot-button-upset-msa":
        if not upset_fn:
            msg, hdr, icn = mk_msg("no_name")
            return toast(True, msg, hdr, icn)
        if not upset_payload:
            msg, hdr, icn = mk_msg("no_data")
            return toast(True, msg, hdr, icn)

        subdir = "Co-occurring-metabolites-upset-plots"
        full_dir = os.path.join(base, subdir)
        if not os.path.isdir(full_dir):
            rel = os.path.relpath(full_dir)
            msg = (
                f"❌ Could not save '{upset_fn}.svg'; "
                f"folder '{rel}' not found."
            )
            return toast(True, msg, "Error", "danger")

        out_path = os.path.join(full_dir, f"{upset_fn}.svg")
        # Save as Plotly or raw SVG
        if upset_payload.get("type") == "plotly":
            fig = pio.from_json(upset_payload["data"])
            pio.write_image(fig, out_path, format="svg")
        else:
            svg_bytes = base64.b64decode(upset_payload["data"])
            with open(out_path, "wb") as f:
                f.write(svg_bytes)

        rel = os.path.relpath(full_dir)
        msg, hdr, icn = mk_msg("upset", upset_fn, rel)
        return toast(True, msg, hdr, icn)

    # 2) Differential upset plot
    if trigger == "confirm-save-plot-button-diff-msa":
        if not diff_fn:
            msg, hdr, icn = mk_msg("no_name")
            return toast(True, msg, hdr, icn)
        if not diff_payload:
            msg, hdr, icn = mk_msg("no_data")
            return toast(True, msg, hdr, icn)

        subdir = "Differential-co-occurring-metabolites-upset-plots"
        full_dir = os.path.join(base, subdir)
        if not os.path.isdir(full_dir):
            rel = os.path.relpath(full_dir)
            msg = (
                f"❌ Could not save '{diff_fn}.svg'; "
                f"folder '{rel}' not found."
            )
            return toast(True, msg, "Error", "danger")

        out_path = os.path.join(full_dir, f"{diff_fn}.svg")
        if diff_payload.get("type") == "plotly":
            fig = pio.from_json(diff_payload["data"])
            pio.write_image(fig, out_path, format="svg")
        else:
            svg_bytes = base64.b64decode(diff_payload["data"])
            with open(out_path, "wb") as f:
                f.write(svg_bytes)

        rel = os.path.relpath(full_dir)
        msg, hdr, icn = mk_msg("diff", diff_fn, rel)
        return toast(True, msg, hdr, icn)

    # default: no change
    return toast(False, no_update, no_update, no_update)
