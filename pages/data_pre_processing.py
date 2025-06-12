from dash import html, dcc
import dash_bootstrap_components as dbc

from .data_pre_processing_page_tabs.select_studies import layout    as select_studies_layout
from .data_pre_processing_page_tabs.select_studies import register_callbacks as register_select_studies_cb
from .data_pre_processing_page_tabs.data_exploration     import layout    as data_exploration_layout
from .data_pre_processing_page_tabs.data_exploration     import register_callbacks as register_data_exploration_cb
from .data_pre_processing_page_tabs.data_summary     import layout    as data_summary_layout
from .data_pre_processing_page_tabs.data_summary     import register_callbacks as register_data_summary_cb

UPLOAD_FOLDER = "pre-processed-datasets"

# -----------------------------
# Helper functions
# -----------------------------

def side_panel_details_dpp():
    return html.Div(
        [
            html.H5("Additional Data Details", style={"marginBottom": "1rem"}),
            html.Div(
                [
                    dbc.Label("Study Name"),
                    dbc.Input(id="study-name_dpp", value="MTBLS6739", type="text")
                ],
                className="mb-3"
            ),
            html.Div(
                [
                    dbc.Label("Dataset Source"),
                    dbc.Select(
                        id="dataset-source_dpp",
                        options=[
                            {"label": "MetaboLights", "value": "metabolights"},
                            {"label": "Metabolomics Workbench", "value": "metabolomics_workbench"},
                            {"label": "Original data - Refmet ids", "value": "original_refmet_ids"},
                            {"label": "Original data - Chebi ids", "value": "original_Chebi_ids"}
                        ],
                        value="metabolights"
                    )
                ],
                className="mb-3"
            ),
        ],
        style={"padding": "1rem", "borderLeft": "1px solid #ccc"}
    )

# This view returns the file upload interface (to be used in the modal)
def select_studies_view_dpp():
    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("Upload Source"),
                    dbc.Input(value="Local upload", disabled=True, style={"marginBottom": "1rem"}),
                    html.H5("Select study to import"),
                    dcc.Upload(
                        id="upload-data_dpp",
                        multiple=True,
                        children=dbc.Button(
                            "Select files from your computer to be uploaded",
                            color="primary"
                        ),
                        style={
                            "display": "block",      # make the upload area only as big as the button
                            "marginTop": "1rem",     # space it down from the title
                            "marginLeft": "auto",    # center it horizontally
                            "marginRight": "auto"
                        }
                    ),
                    # Add a status div for upload messages
                    html.Div(id="upload-status_dpp", style={"marginTop": "1rem", "color": "green"}),
                    # ← placeholder for your checklist!
                    dbc.Checklist(
                        id="selected-files-checklist_dpp",
                        options=[],   # will be filled in by your callback
                        value=[],     # will be pre-checked by your callback
                        inline=False,
                        style={"marginTop": "1rem"},
                    ),
                ],
                width=6
            ),
            dbc.Col(side_panel_details_dpp(), width=6)
        ]
    )

# -----------------------------
# Main Layout for Data Pre-processing page
# -----------------------------
layout = dbc.Container(
        [
            # Hidden stores for state management.
            dcc.Store(id="uploaded-file-store_dpp", data={}),
            dcc.Store(id="selected-study-store_dpp", data=[]),
            dcc.Store(id="study-confirmed-store_dpp", data=False),
            dcc.Store(id="process-data-status_dpp", data=False),
            dcc.Location(id="data-summary-url_dpp", refresh=False),
            dcc.Store(id="active-input-store_dpp", data=""),
            dcc.Store(id="file-explorer-selected_dpp", data=""),
            html.Div(id="dummy-save-status_dpp", style={"display":"none"}),

            dcc.Store(id="tabs-prev_tab", data="details"),

            dcc.Store(id="processing-complete-store_dpp", data=False),
            dcc.Store(id="processing-store", data={"queue": [], "done": 0, "messages": []}),
            dcc.Interval(id="processing-interval", disabled=True, interval=500),
            dcc.Interval(id="hide-progress-interval", interval=2000, disabled=True),
            dcc.Interval(
                id="summary-check-interval_dpp",
                interval=2000,      # wait 2000ms = 2s
                n_intervals=0,
                max_intervals=1,    # only fire once
                disabled=True       # start off disabled
            ),

            dcc.Store(id="start-ts-store", data={}),
            dcc.Interval(id="folder-interval", interval=500, disabled=True),


            html.Div(id="dummy-output", style={"display": "none"}),
            dcc.Store(id="project-folder-store_dpp", data=""),  
            html.Div(
                id="project-name-display",
                style={
                    "fontSize": "24px",   # Medium font size
                    "padding": "10px",
                    "textAlign": "left"
                }
            ),
            # Modal for uploading a new study...
            html.Div(
                dcc.Loading(
                    id="loading-upload-modal",
                    type="circle",
                    fullscreen=False,                          # only cover this component
                    style={"position": "relative"},            # enable absolute overlay
                    overlay_style={                            # dims the modal
                        "position": "absolute",
                        "top": 0,
                        "left": 0,
                        "width": "100%",
                        "height": "100%",
                        "backgroundColor": "rgba(0, 0, 0, 0.3)",
                        "zIndex": 1060                          # above the modal content
                    },
                    children=dbc.Modal(
                        [
                            dbc.ModalHeader("Upload a new study"),
                            dbc.ModalBody(select_studies_view_dpp()),
                            dbc.ModalFooter(
                                [
                                    dbc.Button("Upload", id="upload-study-btn_dpp", color="primary"),
                                    dbc.Button("Close",  id="close-upload-study-btn_dpp", color="secondary"),
                                ],
                                className="d-flex justify-content-end"
                            ),
                        ],
                        id="upload-study-modal_dpp",
                        is_open=False,
                        backdrop="static",
                        size="xl",
                    )
                )
            ),
            # warning modal: shown when a folder with the same study name already exists
            dbc.Modal(
                [
                    dbc.ModalHeader("Folder already exists"),
                    dbc.ModalBody(
                        html.P(
                            [
                                "A study folder named ",
                                html.B(id="overwrite-study-name_dpp"),
                                " already exists. If you continue, new files will be added to it "
                                "and the Dataset Source in the study details file will be updated."
                            ]
                        )
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Cancel",
                                id="cancel-overwrite-btn_dpp",
                                color="secondary",
                                className="me-2",
                            ),
                            dbc.Button(
                                "Continue anyway",
                                id="confirm-overwrite-btn_dpp",
                                color="danger",
                            ),
                        ],
                        className="d-flex justify-content-end",
                    ),
                ],
                id="overwrite-warning-modal_dpp",
                is_open=False,
                backdrop="static",
            ),
            # Save New Data Flow Modal
            dbc.Modal(
                [
                    dbc.ModalHeader("Save New Data Flow", close_button=True),
                    dbc.ModalBody(
                        dbc.Input(
                            id="new-flow-name-input",
                            placeholder="Enter a name for this preprocessing flow",
                            type="text"
                        )
                    ),
                    dbc.ModalFooter(
                        dbc.Button("Save", id="confirm-save-flow-btn_dpp", color="primary")
                    ),
                ],
                id="save-flow-modal_dpp",
                is_open=False,
                centered=True,
            ),
            # Identify group in the datasets (pop up)
            dbc.Modal(
                [
                    dbc.ModalHeader("Identify the group label", close_button=False),
                    dbc.ModalBody(
                        [
                            # Dropdown at the top for group label selection.
                            dcc.Dropdown(
                                id="group-label-dropdown_dpp",
                                options=[],  # options will be populated dynamically
                                placeholder="Select a group label"
                            ),
                            # Medium-sized blank space with initial text 'Group labels'
                            html.Div(
                                "Group labels",
                                id="group-label-space",
                                style={
                                    "height": "150px",          # Adjust height as needed
                                    "marginTop": "15px",          # Spacing between the dropdown and the blank space
                                    "border": "1px dashed #ccc",  # Visual border to demarcate the area (optional)
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "fontSize": "16px",
                                    "color": "#666"
                                }
                            )
                        ]
                    ),
                    dbc.ModalFooter(
                        dbc.Button("Confirm", id="confirm-group-label-btn_dpp", color="primary")
                    ),
                ],
                id="group-identification-modal_dpp",
                is_open=False,
                centered=True,
            ),
            # Define the modal that will ask for the analysis project name
            dbc.Modal(
                [
                    dbc.ModalBody([
                        dbc.Label("Select an Existing Project"),
                        dcc.Dropdown(
                            id="dropdown-existing-projects",
                            placeholder="Choose existing project (optional)"
                        ),
                        html.Hr(),
                        dbc.Label("Or Enter a New Project Name"),
                        dbc.Input(
                            id="input-analysis-project",
                            placeholder="Enter the name of a new project",
                            type="text"
                        )
                    ]),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Confirm",
                            id="confirm-analysis-project-btn",
                            color="primary"
                        )
                    )
                ],
                id="modal-analysis-project",
                is_open=True,
                backdrop="static",
                keyboard=False
            ),
            dbc.Toast(
                id="save-toast",
                header="", # will be set by the callback
                icon="", # will be set by the callback
                duration=3000,           # auto-close after 3s
                is_open=False,           # start hidden
                dismissable=True,
                style={
                    "position": "fixed",
                    "top": 10,
                    "right": 10,
                    "width": 250,
                    "zIndex": 9999,
                },
                children="", # will be set by the callback
            ),
            dbc.Tabs(
                id="data_pre_process_tabs",
                active_tab="select-studies",
                children=[
                    # Select Studies Tab
                    dbc.Tab(
                        label="Select studies",
                        tab_id="select-studies",
                        children=select_studies_layout,
                        style={"padding": "1rem"}
                    ),
                    # Data Exploration Tab 
                    dbc.Tab(
                        label="Data Exploration",
                        tab_id="exploration",
                        id="exploration-tab_dpp",
                        disabled=True,
                        children=data_exploration_layout,
                        style={"padding": "1rem"}
                    ),
                    dbc.Tab(
                        label="Data Summary",
                        tab_id="summary",
                        id="summary-tab_dpp",
                        disabled=True,
                        children=data_summary_layout,
                        style={"padding": "1rem"}
                        
                    )
                ],
                style={
                    "width": "60%",
                    "margin": "20px 0 20px 20px",
                    "boxShadow": "0 0 5px rgba(0,0,0,0.1)"
                }
            )
        ],
        className="py-3",
        style={"textAlign": "center"}
    )

# -----------------------------
# Callbacks 
# -----------------------------

register_select_studies_cb()
register_data_exploration_cb()
register_data_summary_cb()

    














