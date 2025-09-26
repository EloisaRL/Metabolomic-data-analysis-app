# pages/single_study_analysis.py
import os, json
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback, State, no_update, callback_context
import plotly.io as pio
import base64

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
                id="project-dropdown-pop",
                options=[],
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
    dbc.Toast(
        id="save-toast-ssa",
        header="",    # set by callback
        icon="",      # set by callback
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
        children="",  # set by callback
    ),
    # A container with flex styling that holds the projects dropdown and the project title inline.
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
            )
        ],
        style={
            "margin": "1rem",
            "display": "flex",
            "alignItems": "center"
        }
    ),
    # Main content: sidebar and tabs in a flex container.
    html.Div(
        [
            # Sidebar (left column): shows the processed files as a checklist.
            html.Div(
                [
                    html.H4("Processed Files"),
                    dcc.RadioItems(
                        id="selected-file-radio-ssa",
                        options=[],  # Options updated via a callback.
                        # Each label is forced to stay on one line.
                        labelStyle={
                            "display": "block",
                            "whiteSpace": "nowrap",     # keep on one line
                            "overflow": "hidden",       # hide anything that overflows
                            "textOverflow": "ellipsis", # replace overflow with '...'
                            "maxWidth": "100%"          # ensure it respects container width
                        },
                        inputStyle={"margin-right": "10px"}
                    ),
                    html.Div(id="project-files-tooltips-ssa")  # holds tooltips
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
                    html.Div(id="analysis-content", style={"marginTop": "20px"})
                ],
                style={"width": "80%", "padding": "1rem"}
            )
        ],
        style={"display": "flex"}
    )
    # IMPORTANT: include the modal (and its hidden stores) here!
    
])

register_diff_cb()
register_path_cb()

#-------------------------------------------
@callback(
    Output("project-dropdown-pop", "options"),
    Input("project-modal", "is_open")
)
def update_project_list(is_open):
    if is_open:
        projects = list_projects()
        return [
            {"label": p.replace("-", " "), "value": p}
            for p in projects
        ]
    # if modal is closed, don’t change anything
    return no_update

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

# Callback to update the checklist in the sidebar with the files from processed-datasets.
@callback(
    Output("selected-file-radio-ssa", "options"),
    Output("project-files-tooltips-ssa", "children"),
    Input("project-dropdown-pop", "value"),
)
def update_files_checklist(selected_project):
    if not selected_project:
        return [], []

    files = list_processed_files(selected_project)

    # Load study metadata
    project_details_path = os.path.join("Projects", selected_project, "project_details_file.json")
    try:
        with open(project_details_path, "r", encoding="utf-8") as f:
            payload = json.load(f).get("studies", {})
    except Exception:
        payload = {}

    options, tooltips = [], []

    for idx, fpath in enumerate(files):
        basename = os.path.basename(fpath)
        display_name = os.path.splitext(basename)[0]
        if display_name.startswith("processed_"):
            display_name = display_name[len("processed_"):]

        # Extract study name from filename
        no_ext = os.path.splitext(basename)[0]
        study_name = no_ext[len("processed_"):].split("_")[0] if no_ext.startswith("processed_") else None

        info = payload.get(study_name, {}) if study_name else {}
        gf = info.get("group_filter", {}) or {}
        control_group = gf.get("Control") or []
        case_group    = gf.get("Case")    or []
        prep_options  = info.get("preprocessing") or []

        # Safe, simple, unique id based on display_name + index
        #label_id = f"option-{display_name}-{idx}"
        label_id = f"option-{study_name}-{idx}"

        # Truncated label with native-title fallback
        options.append({
                "label": html.Span(study_name, id=label_id),
                "value": fpath
            })
        # Rich tooltip to the right (no assets needed)
        tooltip_body = html.Div(
            [
                html.Div([html.Strong("Study: "), html.Span(study_name or "N/A")]),
                html.Div([html.Strong("Control: "), html.Span(", ".join(control_group) if control_group else "None")]),
                html.Div([html.Strong("Case: "), html.Span(", ".join(case_group) if case_group else "None")]),
                html.Div([html.Strong("Preprocessing: "), html.Span(", ".join(prep_options) if prep_options else "None")]),
                html.Hr(style={"margin": "6px 0"}),
                html.Div([html.Strong("File: "), html.Span(basename)]),
            ],
            style={"textAlign": "left", "whiteSpace": "normal"},
        )

        tooltips.append(
            dbc.Tooltip(
                tooltip_body,
                target=label_id,
                placement="right",
                trigger="hover",
                delay={"show": 100, "hide": 0},
            )
        )

    return options, tooltips

@callback(
    [
        Output("save-toast-ssa", "is_open"),
        Output("save-toast-ssa", "children"),
        Output("save-toast-ssa", "header"),
        Output("save-toast-ssa", "icon"),
    ],
    [
        Input("confirm-save-plot-chart",                "n_clicks"),
        Input("confirm-save-plot-table",                "n_clicks"),
        Input("confirm-save-plot-button-pathway-chart", "n_clicks"),
        Input("confirm-save-plot-button-pathway-table", "n_clicks"),
    ],
    [
        State("plot-name-input-chart",           "value"),
        State("diff-chart-store",                "data"),
        State("plot-name-input-table",           "value"),
        State("diff-table-store",                "data"),
        State("plot-name-input-pathway-chart",   "value"),
        State("pathway-chart-store",             "data"),
        State("plot-name-input-pathway-table",   "value"),
        State("pathway-table-store",             "data"),
        State("project-dropdown-pop",            "value"),
    ],
    prevent_initial_call=True,
)
def show_save_toast_ssa(
    n_chart, n_table, n_pw_chart, n_pw_table,
    chart_fn, chart_payload,
    table_fn, table_payload,
    pw_chart_fn, pw_chart_payload,
    pw_table_fn, pw_table_payload,
    project
):
    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    # helper to build outputs
    def toast(open_, msg, hdr, icn):
        return open_, msg, hdr, icn

    # validation helper
    def alert_msg(kind, name=None):
        if kind == "no_proj":
            return "Select a project before saving.", "Warning", "warning"
        if kind == "no_name":
            return "Enter a name before saving.", "Warning", "warning"
        if kind == "no_data":
            return "No data available to save.", "Error", "danger"
        # success kinds:
        if kind == "chart":
            return f"Chart '{name}.svg' saved.", "Success", "success"
        if kind == "table":
            return f"Table '{name}.csv' saved.", "Success", "success"
        if kind == "pw_chart":
            return f"Pathway chart '{name}.svg' saved.", "Success", "success"
        if kind == "pw_table":
            return f"Pathway table '{name}.csv' saved.", "Success", "success"

    base = os.path.join("Projects", project or "", "Plots", "Single-study-analysis")

    # Ensure project selected
    if trigger in (
        "confirm-save-plot-chart",
        "confirm-save-plot-table",
        "confirm-save-plot-button-pathway-chart",
        "confirm-save-plot-button-pathway-table"
    ) and not project:
        msg, hdr, icn = alert_msg("no_proj")
        return toast(True, msg, hdr, icn)

    # 1) Metabolite chart
    if trigger == "confirm-save-plot-chart":
        if not chart_fn:
            msg, hdr, icn = alert_msg("no_name")
        elif not chart_payload:
            msg, hdr, icn = alert_msg("no_data")
        else:
            fig = pio.from_json(chart_payload["data"])
            out_dir = os.path.join(base, "Differential-metabolites-box-plots")
            os.makedirs(out_dir, exist_ok=True)
            pio.write_image(
                fig,
                os.path.join(out_dir, f"{chart_fn}.svg"),
                format="svg",
                width=int(fig.layout.width or 700),
                height=int(fig.layout.height or 400),
            )
            msg, hdr, icn = alert_msg("chart", chart_fn)
        return toast(True, msg, hdr, icn)

    # 2) Metabolite table
    if trigger == "confirm-save-plot-table":
        if not table_fn:
            msg, hdr, icn = alert_msg("no_name")
        elif not table_payload:
            msg, hdr, icn = alert_msg("no_data")
        else:
            out_dir = os.path.join(base, "Differential-metabolites-table-plots")
            os.makedirs(out_dir, exist_ok=True)
            data = base64.b64decode(table_payload["data"])
            with open(os.path.join(out_dir, f"{table_fn}.csv"), "wb") as f:
                f.write(data)
            msg, hdr, icn = alert_msg("table", table_fn)
        return toast(True, msg, hdr, icn)

    # 3) Pathway chart
    if trigger == "confirm-save-plot-button-pathway-chart":
        if not pw_chart_fn:
            msg, hdr, icn = alert_msg("no_name")
        elif not pw_chart_payload:
            msg, hdr, icn = alert_msg("no_data")
        else:
            fig = pio.from_json(pw_chart_payload["data"])
            out_dir = os.path.join(base, "Differential-pathway-box-plots")
            os.makedirs(out_dir, exist_ok=True)
            pio.write_image(
                fig,
                os.path.join(out_dir, f"{pw_chart_fn}.svg"),
                format="svg",
                width=int(fig.layout.width or 700),
                height=int(fig.layout.height or 400),
            )
            msg, hdr, icn = alert_msg("pw_chart", pw_chart_fn)
        return toast(True, msg, hdr, icn)

    # 4) Pathway table
    if trigger == "confirm-save-plot-button-pathway-table":
        if not pw_table_fn:
            msg, hdr, icn = alert_msg("no_name")
        elif not pw_table_payload:
            msg, hdr, icn = alert_msg("no_data")
        else:
            out_dir = os.path.join(base, "Differential-pathway-table-plots")
            os.makedirs(out_dir, exist_ok=True)
            data = base64.b64decode(pw_table_payload["data"])
            with open(os.path.join(out_dir, f"{pw_table_fn}.csv"), "wb") as f:
                f.write(data)
            msg, hdr, icn = alert_msg("pw_table", pw_table_fn)
        return toast(True, msg, hdr, icn)

    # default: no update
    return toast(False, no_update, no_update, no_update)




