# pages/data_pre_processing_page_tabs/data_exploration.py
from .shared_functions.helper import (read_study_details_dpp,
                                      get_flow_steps)
from .shared_functions.data_processing import (static_preprocess_workbench,
                                               static_preprocess)
from dash import html, dcc, callback, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import os
from dash.exceptions import PreventUpdate
import json
import pandas as pd
import glob
import requests
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import re
import numbers
import ast
import plotly.io as pio
import logging
from io import StringIO
logger = logging.getLogger(__name__)
from paths import UPLOAD_FOLDER, APP_ROOT, DATA_FLOWS_DIR

#UPLOAD_FOLDER = "pre-processed-datasets"
# Path to the temporary file that holds the selected studies
SELECTED_STUDIES_FILE = os.path.join(UPLOAD_FOLDER, "selected_studies_temp.txt")

###### REMOVE THE RELIANCE ON BELOW #######
default_metadata = pd.DataFrame({
    "Sample Name": ["Sample1", "Sample2", "Sample3"],
    "Group": ["Control", "Treatment", "Control"]
})

    
def ensure_list(val):
    """Ensure a value is a list."""
    if val is None:
        return []
    elif isinstance(val, list):
        return val
    else:
        return [val]


# ===================================== #
# PCA, Residual, and Box plot functions #
# ===================================== #

def pca_plot(data):
    """PCA and Residual plots"""
    # Separate Group
    if 'Group' in data.columns:
        groups = data['Group']
        data_for_pca = data.drop('Group', axis=1).fillna(0)
    else:
        groups = None
        data_for_pca = data.fillna(0)

    # Check that there are numeric columns available
    if data_for_pca.empty or data_for_pca.shape[1] == 0:
        raise ValueError("No numeric columns available for PCA")

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_for_pca)

    df_pca = pd.DataFrame(pca_result, columns=['PC1','PC2'], index=data.index)
    df_pca['Sample'] = data.index
    if groups is not None:
        df_pca['Group'] = groups

    title = f'Combined PCA Plot (PC1 {pca.explained_variance_ratio_[0]*100:.1f}%, PC2 {pca.explained_variance_ratio_[1]*100:.1f}%)'
    fig_pca = px.scatter(
        df_pca,
        x='PC1', y='PC2',
        hover_name='Sample',
        color='Group' if groups is not None else None,
        title=title
    )

    # Build colour map from PCA traces
    color_map = {}
    if groups is not None:
        for trace in fig_pca.data:
            color_map[trace.name] = trace.marker.color

    # Compute residuals
    recon = pca.inverse_transform(pca_result)
    df_pca['Residual'] = np.linalg.norm(data_for_pca.values - recon, axis=1)

    # Create residual bar chart using same colours
    if groups is not None:
        fig_residual = px.bar(
            df_pca.sort_values('Residual', ascending=False),
            x='Sample', y='Residual',
            color='Group',
            color_discrete_map=color_map,
            title="PCA Reconstruction Error (Residuals)"
        )
    else:
        fig_residual = px.bar(
            df_pca.sort_values('Residual', ascending=False),
            x='Sample', y='Residual',
            title="PCA Reconstruction Error (Residuals)"
        )

    return fig_pca, fig_residual


def box_plot(data):
    """Box plot"""
    if 'Group' in data.columns:
        groups = data['Group']
        data_numeric = data.drop('Group', axis=1)
        unique_groups = groups.unique()
        color_sequence = px.colors.qualitative.Plotly
        color_map = {grp: color_sequence[i % len(color_sequence)] for i, grp in enumerate(unique_groups)}
    else:
        data_numeric = data
    fig_box = go.Figure()
    if 'Group' in data.columns:
        for grp in unique_groups:
            fig_box.add_trace(
                go.Box(
                    x=[], 
                    y=[], 
                    name=grp,
                    marker=dict(color=color_map[grp]),
                    legendgroup=grp,
                    showlegend=True
                )
            )
        for sample in data_numeric.index:
            group = groups.loc[sample]
            fig_box.add_trace(
                go.Box(
                    y=data_numeric.loc[sample].values,
                    name=sample,
                    boxpoints='outliers',
                    marker=dict(size=4, color=color_map[group]),
                    whiskerwidth=0.2,
                    line=dict(width=1, color=color_map[group]),
                    legendgroup=group,
                    showlegend=False,
                    customdata=[sample]*len(data_numeric.loc[sample].values),
                    hovertemplate = f"Sample: %{{customdata}}<br>Group: {group}<extra></extra>"
                )
            )
    else:
        for sample in data_numeric.index:
            fig_box.add_trace(
                go.Box(
                    y=data_numeric.loc[sample].values,
                    name=sample,
                    boxpoints='outliers',
                    marker=dict(size=4),
                    whiskerwidth=0.2,
                    line=dict(width=1)
                )
            )
    fig_box.update_layout(
        title="Box and Whisker Plot of Samples",
        yaxis_title="Value",
        xaxis_title="Sample"
    )
    return fig_box

# ================================== #
# Layout of the Data exploration tab #
# ================================== #

layout = html.Div(
                    id="data-exploration-tab-content",  # Focusable container
                    tabIndex=0,  # Makes this container able to receive focus.
                    children=[
                        dcc.Dropdown(
                            id="selected-studies-dropdown_dpp",
                            placeholder="Select a study",
                            style={"width": "300px", "margin": "1rem auto"}
                        ),
                        dbc.Row([
                            # Left column: graphs (width 9)
                            dbc.Col(
                                [
                                    # ─── PCA graph + save button + feedback ────────────────────────────────
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Save PCA plot",
                                                id="save-pca-btn_dpp",
                                                color="info",
                                                size="sm",
                                                style={
                                                    "position": "absolute",
                                                    "top": "0.5rem",
                                                    "right": "0.5rem",
                                                    "zIndex": 1,
                                                    "backgroundColor": "white",
                                                    "color": "black"
                                                }
                                            ),
                                            dcc.Loading(
                                                dcc.Graph(id="pca-graph_dpp", figure={}, style={"height": "500px"}),
                                                type="circle",
                                                delay_show=0
                                            ),
                                            html.Div(id="save-feedback-pca_dpp", style={"marginTop": "0.5rem"})
                                        ],
                                        style={"position": "relative", "marginBottom": "2rem"}
                                    ),
                                    # PCA filename modal
                                    dbc.Modal(
                                        [
                                            dbc.ModalHeader(dbc.ModalTitle("Save PCA Plot")),
                                            dbc.ModalBody([
                                                dbc.Label("Filename"),
                                                dbc.Input(
                                                    id="plot-name-input-pca_dpp",
                                                    placeholder="Enter filename (no extension)"
                                                )
                                            ]),
                                            dbc.ModalFooter([
                                                dbc.Button("Cancel", id="cancel-save-pca_dpp", n_clicks=0, className="ms-auto"),
                                                dbc.Button("Save",   id="confirm-save-pca_dpp", n_clicks=0, color="primary")
                                            ])
                                        ],
                                        id="save-plot-modal-pca_dpp",
                                        is_open=False
                                    ),

                                    # ─── Residuals graph + save button + feedback ───────────────────────────
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Save Residuals plot",
                                                id="save-residual-btn_dpp",
                                                color="info",
                                                size="sm",
                                                style={
                                                    "position": "absolute",
                                                    "top": "0.5rem",
                                                    "right": "0.5rem",
                                                    "zIndex": 1,
                                                    "backgroundColor": "white",
                                                    "color": "black"
                                                }
                                            ),
                                            dcc.Loading(
                                                dcc.Graph(id="residual-graph_dpp", figure={}, style={"height": "500px"}),
                                                type="circle",
                                                delay_show=0
                                            ),
                                            html.Div(id="save-feedback-residual_dpp", style={"marginTop": "0.5rem"})
                                        ],
                                        style={"position": "relative", "marginBottom": "2rem"}
                                    ),
                                    # Residuals filename modal
                                    dbc.Modal(
                                        [
                                            dbc.ModalHeader(dbc.ModalTitle("Save Residuals Plot")),
                                            dbc.ModalBody([
                                                dbc.Label("Filename"),
                                                dbc.Input(
                                                    id="plot-name-input-residual_dpp",
                                                    placeholder="Enter filename (no extension)"
                                                )
                                            ]),
                                            dbc.ModalFooter([
                                                dbc.Button("Cancel", id="cancel-save-residual_dpp", n_clicks=0, className="ms-auto"),
                                                dbc.Button("Save",   id="confirm-save-residual_dpp", n_clicks=0, color="primary")
                                            ])
                                        ],
                                        id="save-plot-modal-residual_dpp",
                                        is_open=False
                                    ),

                                    # ─── Boxplot graph + save button + feedback ─────────────────────────────
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Save Boxplot",
                                                id="save-box-btn_dpp",
                                                color="info",
                                                size="sm",
                                                style={
                                                    "position": "absolute",
                                                    "top": "0.5rem",
                                                    "right": "0.5rem",
                                                    "zIndex": 1,
                                                    "backgroundColor": "white",
                                                    "color": "black"
                                                }
                                            ),
                                            dcc.Loading(
                                                dcc.Graph(id="box-graph_dpp", figure={}, style={"height": "500px"}),
                                                type="circle",
                                                delay_show=0
                                            ),
                                            html.Div(id="save-feedback-box_dpp", style={"marginTop": "0.5rem"})
                                        ],
                                        style={"position": "relative"}
                                    ),
                                    # Boxplot filename modal
                                    dbc.Modal(
                                        [
                                            dbc.ModalHeader(dbc.ModalTitle("Save Boxplot")),
                                            dbc.ModalBody([
                                                dbc.Label("Filename"),
                                                dbc.Input(
                                                    id="plot-name-input-box_dpp",
                                                    placeholder="Enter filename (no extension)"
                                                )
                                            ]),
                                            dbc.ModalFooter([
                                                dbc.Button("Cancel", id="cancel-save-box_dpp", n_clicks=0, className="ms-auto"),
                                                dbc.Button("Save",   id="confirm-save-box_dpp", n_clicks=0, color="primary")
                                            ])
                                        ],
                                        id="save-plot-modal-box_dpp",
                                        is_open=False
                                    ),
                                ],
                                width=9
                            ),
                            # Right column: sidebars (width 3)
                            dbc.Col(
                                html.Div(
                                    [
                                        # 1) Study details sidebar
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.H4("Study details", style={"margin": 0}),
                                                        dbc.Button("Confirm", id="confirm-study-details_dpp", color="primary")
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "justifyContent": "space-between",
                                                        "alignItems": "center",
                                                        "marginBottom": "1rem"
                                                    }
                                                ),
                                                html.Div(
                                                    [
                                                        dbc.Label("Outliers"),
                                                        dbc.Input(
                                                            id="side-outliers_dpp",
                                                            placeholder="Enter outliers",
                                                            type="text"
                                                        )
                                                    ],
                                                    id="div-outliers_dpp",
                                                    className="mb-3",
                                                    style={"cursor": "pointer"}
                                                ),
                                                html.Div(
                                                    [
                                                        dbc.Label("Control group"),
                                                        dcc.Dropdown(
                                                            id="side-control-group_dpp",
                                                            placeholder="Select control group",
                                                            multi=True
                                                        )
                                                    ],
                                                    className="mb-3"
                                                ),
                                                html.Div(
                                                    [
                                                        dbc.Label("Case group"),
                                                        dcc.Dropdown(
                                                            id="side-case-group_dpp",
                                                            placeholder="Select case group",
                                                            multi=True
                                                        )
                                                    ],
                                                    className="mb-3"
                                                )
                                            ],
                                            style={
                                                "padding": "1rem",
                                                "border": "1px solid #ccc",
                                                "borderRadius": "5px"
                                            }
                                        ),

                                        html.Br(),

                                        # 2) Data processing sidebar
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.H4("Data Processing", style={"margin": 0}),
                                                        dbc.Button("Confirm", id="confirm-data-processing_dpp", color="primary", size="sm")
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "justifyContent": "space-between",
                                                        "alignItems": "center",
                                                        "marginBottom": "1rem"
                                                    }
                                                ),
                                                html.H5("Pre-built data processing", className="mt-3"),
                                                dbc.Checklist(
                                                    id="prebuilt-flows_dpp",
                                                    options=[],          # populated by callback
                                                    value=[],
                                                    inline=False,
                                                    style={"paddingLeft": "1rem"}
                                                ),
                                                html.Div(id="prebuilt-tooltips-container_dpp"),
                                                dcc.Interval(
                                                    id="update-prebuilt-flows-interval_dpp", 
                                                    interval=1000, 
                                                    max_intervals=1
                                                ),
                                                html.H5("New data processing", className="mt-4"),
                                                html.Div(
                                                    [
                                                        html.H6("Missing Values Imputation"),
                                                        dbc.Checklist(
                                                            id="missing-values-checklist_dpp",
                                                            options=[
                                                                {"label": "KNN Imputer", "value": "knn_imputer"},
                                                                {"label": "Mean Imputer", "value": "mean_imputer"},
                                                                {"label": "Iterative Imputer", "value": "iterative_imputer"}
                                                            ],
                                                            value=[],
                                                            inline=False,
                                                            style={"paddingLeft": "1rem"}
                                                        ),
                                                        dcc.Store(id="prev-transformation_dpp",  data=[]),
                                                        dcc.Store(id="prev-standardisation_dpp", data=[]),
                                                        html.H6("Transformation", className="mt-3"),
                                                        dbc.Checklist(
                                                            id="transformation-checklist_dpp",
                                                            options=[
                                                                {"label": "Log Transform", "value": "log_transform"},
                                                                {"label": "Cube Root Transform", "value": "cube_root"}
                                                            ],
                                                            value=[],
                                                            inline=False,
                                                            style={"paddingLeft": "1rem"}
                                                        ),
                                                        html.H6("Standardisation", className="mt-3"),
                                                        dbc.Checklist(
                                                            id="standardisation-checklist_dpp",
                                                            options=[
                                                                {"label": "Standard Scaler", "value": "standard_scaler"},
                                                                {"label": "Min-Max Scaler", "value": "min_max_scaler"},
                                                                {"label": "Robust Scaler", "value": "robust_scaler"},
                                                                {"label": "Max Abs Scaler", "value": "max_abs_scaler"}
                                                            ],
                                                            value=[],
                                                            inline=False,
                                                            style={"paddingLeft": "1rem"}
                                                        )
                                                    ]
                                                ),
                                                dbc.Button(
                                                    "Save options",
                                                    id="save-options-btn_dpp",
                                                    color="secondary",
                                                    size="sm",
                                                    className="mt-3",
                                                    style={"width": "50%", "fontSize": "0.8rem", "padding": "0.2rem 0.5rem"}
                                                ),
                                            ],
                                            style={
                                                "padding": "1rem",
                                                "border": "1px solid #ccc",
                                                "borderRadius": "5px",
                                                "marginTop": "1.5rem"
                                            }
                                        ),

                                        # 3) Move Refresh button here, below both boxes
                                        dbc.Button(
                                            "Refresh graphs",
                                            id="refresh-graphs-sidebar-btn_dpp",
                                            color="primary",
                                            size="lg",              # make it larger
                                            className="mt-4",
                                            style={
                                                "width": "100%",
                                                "fontSize": "1.1rem",
                                                "padding": "0.75rem"
                                            }
                                        )
                                    ],
                                    style={"display": "flex", "flexDirection": "column"}
                                ),
                                width=3
                            )

                        ])
                    ],
                    style={"padding": "1rem"}
                )

def register_callbacks(): 
    # Callback to let the user know when all the details nescessary have been provided for a study in the dropdown   
    @callback(
        [
            Output("selected-studies-dropdown_dpp", "options"),
            Output("selected-studies-dropdown_dpp", "value"),
        ],
        [
            Input("selected-study-store_dpp",       "data"),
            Input("selected-studies-dropdown_dpp",  "value"),
        ],
    )
    def update_selected_studies_dropdown(selected_studies, current_value):
        if not selected_studies:
            return [], None

        # Sanitize: strip leading "✓ " if present
        sanitized_studies = [
            s[2:] if isinstance(s, str) and s.startswith("✓ ") else s
            for s in selected_studies
        ]
        if isinstance(current_value, str) and current_value.startswith("✓ "):
            current_value = current_value[2:]

        # Attempt to load the JSON payload once
        try:
            with open(SELECTED_STUDIES_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f).get("studies", {})
        except Exception:
            logger.exception("Data exploration tab - Error reading SELECTED_STUDIES_FILE")
            payload = {}

        options = []
        for study in selected_studies:
            info = payload.get(study, {})
            gf      = info.get("group_filter", {})
            control = gf.get("Control") or []
            case    = gf.get("Case")    or []
            prep    = info.get("preprocessing") or []

            # A study is "ready" if it has nonempty Control, Case, and preprocessing
            ready = bool(control and case and isinstance(prep, list) and prep)
            label = f"{'✓ ' if ready else ''}{study}"
            options.append({"label": label, "value": study})

        # Preserve the current selection if still valid; otherwise pick the first
        if current_value in selected_studies:
            return options, current_value
        else:
            return options, options[0]["value"]

    # Callback controls the confirmation of study selection and confirmation of group selection
    @callback(
        [Output("selected-study-store_dpp", "data"),
        Output("study-confirmed-store_dpp", "data")],
        [Input("confirm-study-selection-btn_dpp", "n_clicks"),
        Input("confirm-group-label-btn_dpp", "n_clicks")],
        [State("studies-table_dpp", "data"),
        State("studies-table_dpp", "selected_rows"),
        State("group-label-dropdown_dpp", "value"),
        State("selected-study-store_dpp", "data"),
        State("selected-studies-dropdown_dpp", "value")],
        prevent_initial_call=True
    )
    def combined_callback(confirm_study_clicks, confirm_group_clicks,
                        table_data, selected_rows, group_selection, stored_study, dropdown_study):

        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        # Sanitize: strip leading "✓ " if present
        if isinstance(dropdown_study, str) and dropdown_study.startswith("✓ "):
            dropdown_study = dropdown_study[2:]

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if triggered_id == "confirm-study-selection-btn_dpp":
            # When the "Confirm study selection" button is clicked:
            if table_data and selected_rows is not None and len(selected_rows) > 0:
                selected_studies = [table_data[i]["Study Name"] for i in selected_rows]
                # Build the payload such that each study has its own dict for 'group' and 'preprocessing'
                # with additional keys for 'outliers' and 'group_filter'
                payload = {
                    "studies": {
                        study: {
                            "filename": "",
                            "group_type": None,
                            "preprocessing": {},
                            "outliers": "",          
                            "group_filter": {}         
                        } for study in selected_studies
                    }
                }
                try:
                    os.makedirs(os.path.dirname(SELECTED_STUDIES_FILE), exist_ok=True)
                    with open(SELECTED_STUDIES_FILE, "w") as f:
                        json.dump(payload, f, indent=2)
                except Exception:
                    logger.exception("Data exploration tab - Error writing SELECTED_STUDIES_FILE")
                logger.info(f"Data exploration tab - Studies selected for processing: {selected_studies}")
                # Return the full list of selected studies.
                return selected_studies, True
            else:
                raise PreventUpdate

        elif triggered_id == "confirm-group-label-btn_dpp":
            # When the "Confirm group" button is clicked, use the currently selected study from the dropdown.
            if group_selection and dropdown_study:
                current_study = dropdown_study[0] if isinstance(dropdown_study, list) else dropdown_study
                try:
                    with open(SELECTED_STUDIES_FILE, "r") as f:
                        payload = json.load(f)
                except Exception:
                    logger.exception("Data exploration tab - Error reading SELECTED_STUDIES_FILE")
                    payload = {"studies": {}}

                # Update the payload with the group selection for the current study.
                payload["studies"][current_study]["group_type"] = group_selection
                try:
                    with open(SELECTED_STUDIES_FILE, "w") as f:
                        json.dump(payload, f, indent=2)
                except Exception:
                    logger.exception("Data exploration tab - Error writing group selection to SELECTED_STUDIES_FILE")
                # Return the original full list of selected studies.
                return stored_study, True
            else:
                raise PreventUpdate

        else:
            raise PreventUpdate

    # Callback controls showing the different group types for a study
    @callback(
        [Output("group-identification-modal_dpp", "is_open"),
        Output("group-label-dropdown_dpp", "options")],
        [Input("selected-studies-dropdown_dpp", "value"),
        Input("data_pre_process_tabs", "active_tab")],
        State("selected-study-store_dpp", "data"),
        prevent_initial_call=True
    )
    def show_group_popup(selected_study, active_tab, stored_study):

        # Only show the group pop-up when on the "exploration" tab.
        if active_tab != "exploration":
            return False, []

        if not selected_study:
            raise PreventUpdate

        # Sanitize: strip leading "✓ " if present
        if isinstance(selected_study, str) and selected_study.startswith("✓ "):
            selected_study = selected_study[2:]

        # If selected_study is a list, take the first element.
        if isinstance(selected_study, list):
            if not selected_study:
                raise PreventUpdate
            selected_study = selected_study[0]

        folder = os.path.join(UPLOAD_FOLDER, selected_study)
        if not os.path.exists(folder):
            return False, []

        details = read_study_details_dpp(folder)
        dataset_source = details.get("Dataset Source", "").lower()

        # Only proceed for metabolomics workbench or metabolights datasets.
        if dataset_source not in ["metabolomics workbench", "metabolights", "original data - refmet ids", "original data - chebi ids"]:
            return False, []

        # Check the SELECTED_STUDIES_FILE for a saved group.
        try:
            with open(SELECTED_STUDIES_FILE, "r") as f:
                payload = json.load(f)
        except Exception:
            logger.exception("Data exploration tab - Error reading SELECTED_STUDIES_FILE")
            payload = {"selected_studies": [], "preprocessing": {}}

        saved_group = payload.get("studies", {}).get(selected_study, {}).get("group_type")
        if saved_group:
            # A group is already saved for this study, so no need to open the modal.
            return False, []

        # Set group options based on the dataset source.
        group_options = []
        if dataset_source in (
            "metabolomics workbench",
            "original data - refmet ids",
            "original data - chebi ids",
        ):
            # Extract group options from the first CSV.
            csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
            if csv_files:
                csv_path = os.path.join(folder, csv_files[0])
                try:
                    df_temp = pd.read_csv(csv_path)
                    if "Class" in df_temp.columns and not df_temp.empty:
                        class_value = df_temp.iloc[0]["Class"]
                        groups = [grp.strip() for grp in class_value.split("|") if grp.strip()]
                        group_options = [{"label": grp, "value": grp} for grp in groups]
                except Exception:
                    logger.exception("Data exploration tab - Error reading CSV for group extraction")
        elif dataset_source == "metabolights":
            # 1) build the pattern
            pattern = os.path.join(folder, "s_*.txt")

            # 2) expand the pattern into actual files
            matches = glob.glob(pattern)

            # 3) handle zero or many matches, and pick one
            if not matches:
                logger.error(f"Data exploration tab - No metadata file found matching pattern: {pattern!r}")
                raise PreventUpdate
            elif len(matches) > 1:
                # you could choose the newest, the first, or raise an error
                matches.sort()  # alphabetical; or sort by os.path.getmtime for newest
            meta_filepath = matches[0]
            #meta_filepath = os.path.join(folder, "s_*.txt")
            metadata_df = (pd.read_csv(meta_filepath, sep="\t", encoding="unicode_escape")
                if os.path.exists(meta_filepath) else default_metadata)

            group_options = [{"label": col, "value": col} for col in metadata_df.columns if "Factor Value" in col]

        # If no group options were determined, do not open the modal.
        if not group_options:
            return False, []

        # Open the modal with the appropriate group options.
        return True, group_options

    # Callback controls showing the group options for selected group type for a study
    @callback(
        Output("group-label-space", "children"),
        [Input("group-label-dropdown_dpp", "value"),
        Input("selected-studies-dropdown_dpp", "value")]
    )
    def update_group_label_space(selected_group, selected_study):
        # If no group is selected, show the default text.
        if not selected_group:
            return "Group labels"
        
        # Use the selected study from the study dropdown.
        if not selected_study:
            return "No study selected."
        
        # If the selected study comes as a list, take the first element.
        if isinstance(selected_study, list):
            if not selected_study:
                return "No study selected."
            selected_study = selected_study[0]
        
        folder = os.path.join(UPLOAD_FOLDER, selected_study)
        if not os.path.exists(folder):
            return "Data folder not found."
        
        # Read the study details (assumed to return at least "Dataset Source" and optionally "ID")
        details = read_study_details_dpp(folder)
        dataset_source = details.get("Dataset Source", "").lower()
        
        # Define a style dict to be used when returning the list of labels.
        list_style = {
            "height": "150px",
            "overflowY": "auto",
            "padding": "10px",
            "textAlign": "left",
            "border": "1px dashed #ccc"  # Optional: to visually demarcate the area
        }
        
        if dataset_source in (
            "metabolomics workbench",
            "original data - refmet ids",
            "original data - chebi ids",
        ):
            # For workbench studies, extract group labels from the first CSV file’s "Class" column.
            csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
            if not csv_files:
                return "No CSV files found."
            csv_path = os.path.join(folder, csv_files[0])
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                logger.exception(f"Data exploration tab - Error reading CSV file: {csv_path}")
                return "Error reading CSV file"
            
            if "Class" not in df.columns or df.empty:
                return "CSV missing 'Class' column or is empty."
            
            # Use the first row to determine group order (assumes pipe-separated values)
            class_value = df.iloc[0]["Class"]
            groups = [grp.strip() for grp in str(class_value).split("|") if grp.strip()]
            try:
                group_index = groups.index(selected_group)
            except ValueError:
                return "Selected group not found in CSV header."
            
            # Collect unique labels for the selected group index across all rows.
            unique_labels = set()
            for val in df["Class"]:
                parts = [p.strip() for p in str(val).split("|")]
                if len(parts) > group_index:
                    unique_labels.add(parts[group_index])
            unique_labels = sorted(unique_labels)
            
            if unique_labels:
                return html.Div(
                    html.Ul([html.Li(label) for label in unique_labels]),
                    style=list_style
                )
            else:
                return "No unique labels found in CSV."
        
        elif dataset_source == "metabolights":
            # For metabolights studies, assume the dropdown value corresponds to a metadata column name.

            # 1) build the pattern
            pattern = os.path.join(folder, "s_*.txt")

            # 2) expand the pattern into actual files
            matches = glob.glob(pattern)

            # 3) handle zero or many matches, and pick one
            if not matches:
                logger.error(f"Data exploration tab - No metadata file found matching pattern: {pattern!r}")
                raise PreventUpdate
            elif len(matches) > 1:
                # you could choose the newest, the first, or raise an error
                matches.sort()  # alphabetical; or sort by os.path.getmtime for newest
            meta_filepath = matches[0]
            if not os.path.exists(meta_filepath):
                return "Metadata file not found."
            try:
                metadata_df = pd.read_csv(meta_filepath, sep="\t", encoding="unicode_escape")
            except Exception:
                logger.exception("Data exploration tab - Error reading metadata file")
                return "Error reading metadata file"
            
            if selected_group not in metadata_df.columns:
                return f"Selected column '{selected_group}' not found in metadata."
            
            unique_vals = metadata_df[selected_group].dropna().unique()
            unique_vals = sorted(unique_vals)  # sorted returns a list
            
            if len(unique_vals) > 0:
                return html.Div(
                    html.Ul([html.Li(str(val)) for val in unique_vals]),
                    style=list_style
                )
            else:
                return "No unique labels found in metadata."
        
        else:
            return "Unsupported dataset source."
    
    # Callback to produce the PCA, Residual, and box graphs based on the options chosen in the sidebar
    @callback(
        [
            Output("pca-graph_dpp", "figure"),
            Output("residual-graph_dpp", "figure"),
            Output("box-graph_dpp", "figure"),
        ],
        [
            Input("selected-studies-dropdown_dpp", "value"),
            Input("refresh-graphs-sidebar-btn_dpp", "n_clicks"),
            Input("data_pre_process_tabs", "active_tab"),
            Input("confirm-group-label-btn_dpp", "n_clicks"),
            Input("study-confirmed-store_dpp", "data"),
        ],
        [
            State("group-label-dropdown_dpp", "value"),
            State("selected-study-store_dpp", "data"),
            State("prebuilt-flows_dpp", "value"),
            State("missing-values-checklist_dpp", "value"),
            State("transformation-checklist_dpp", "value"),
            State("standardisation-checklist_dpp", "value"),
            # Sidebar inputs
            State("side-outliers_dpp", "value"),
            State("side-control-group_dpp", "value"),
            State("side-case-group_dpp", "value"),
        ],
        prevent_initial_call=True
    )
    def produce_graphs(
        selected_study, refresh_clicks, active_tab, confirm_clicks, study_confirmed,
        group_selection, stored_study,
        prebuilt_flow_state, missing_values_opts, transformation_opts, standardisation_opts,
        side_outliers, side_control_group, side_case_group
    ):
        """
        Produces PCA, residual, and box plots for the selected study.
        Applies sidebar outliers and control/case filters (or None if unset).
        """
        
        # Early exits
        if active_tab != "exploration" or not selected_study:
            raise PreventUpdate
        if isinstance(selected_study, list):
            if not selected_study:
                raise PreventUpdate
            selected_study = selected_study[0]

        # Sanitize: strip leading "✓ " if present
        if isinstance(selected_study, str) and selected_study.startswith("✓ "):
            selected_study = selected_study[2:]

        folder = os.path.join(UPLOAD_FOLDER, selected_study)
        if not os.path.isdir(folder):
            return {}, {}, {}
        
        # Unpack single-element lists to strings (or None)
        prebuilt_flow   = prebuilt_flow_state[0]   if prebuilt_flow_state   else None
        missing_value   = missing_values_opts[0]   if missing_values_opts   else None
        transformation  = transformation_opts[0]   if transformation_opts   else None
        standardisation = standardisation_opts[0]  if standardisation_opts  else None

        details = read_study_details_dpp(folder)
        dataset_source = details.get("Dataset Source", "").lower()

        # Read the selected studies file and look for the confirmed group.
        try:
            with open(SELECTED_STUDIES_FILE, "r") as f:
                payload = json.load(f)
        except Exception:
            logger.exception("Data exploration tab - Error reading SELECTED_STUDIES_FILE")
            payload = {"studies": {}}

        saved_group = payload.get("studies", {}).get(selected_study, {}).get("group_type")
        if not saved_group:
            # Group has not been confirmed.
            return {}, {}, {}

        group_selection = saved_group

        outliers = payload.get("studies", {}).get(selected_study, {}).get("outliers")
        if isinstance(outliers, str):
            # Split by comma and remove any extra whitespace from each entry
            outliers = [value.strip() for value in outliers.split(",") if value.strip()]
        filters = payload.get("studies", {}).get(selected_study, {}).get("group_filter")

        # Determine trigger
        ctx = callback_context
        triggered = ctx.triggered and ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Helper for preprocessing
        def do_preprocess():
            if prebuilt_flow:
                steps = get_flow_steps(prebuilt_flow)
            else:
                # custom → preserve ordering: missing → transform → standardise
                steps = (
                    ensure_list(missing_value) +
                    ensure_list(transformation) +
                    ensure_list(standardisation)
                )

            common_kwargs = dict(
                preprocessing_steps=steps,
                selected_group=group_selection,
                outliers=outliers,
                filter=filters,
            )
            if dataset_source in (
                "metabolomics workbench",
                "original data - refmet ids",
                "original data - chebi ids",
            ):
                return static_preprocess_workbench(
                    folder,
                    database_source=dataset_source,
                    **common_kwargs
                )
            else:
                # Load metadata
                pattern = os.path.join(folder, "s_*.txt")
                matches = glob.glob(pattern)
                if not matches:
                    logger.error(f"Data exploration tab - No metadata file found matching pattern: {pattern!r}")
                    raise PreventUpdate
                matches.sort()
                meta_fp = matches[0]
                try:
                    metadata_df = pd.read_csv(meta_fp, sep="\t", encoding="unicode_escape")
                except Exception:
                    logger.exception("Data exploration tab - Error reading metadata file")
                    #metadata_df = default_metadata
                    raise PreventUpdate

                return static_preprocess(
                    folder,
                    metadata_df,
                    **common_kwargs
                )

        # On Refresh → always re-generate
        if triggered == "refresh-graphs-sidebar-btn_dpp":
            # Parse outliers
            outliers = (
                [o.strip() for o in side_outliers.split(",") if o.strip()]
                if side_outliers else
                None
            )

            # Parse control/case; only use if both lists non-empty
            cg = side_control_group or []
            eg = side_case_group    or []
            filters = {"Control": cg, "Case": eg} if cg and eg else None
            try:
                payload = json.load(open(SELECTED_STUDIES_FILE))
                group_selection = payload["studies"][selected_study]["group_type"]
            except Exception:
                logger.exception("Data exploration tab - Error reading group type in SELECTED_STUDIES_FILE")
                return {}, {}, {}

            try:
                df = do_preprocess()
                print('data on page')
                print(df)
                if df.empty:
                    return {}, {}, {}
                fig_pca, fig_residual = pca_plot(df)
                fig_box          = box_plot(df)
                return fig_pca, fig_residual, fig_box
            except Exception:
                logger.exception("Data exploration tab - Error on refresh preprocess, triggered by refresh button")
                return {}, {}, {}

        # Otherwise only after confirm…
        try:
            payload = json.load(open(SELECTED_STUDIES_FILE))
            group_selection = payload["studies"].get(selected_study, {}).get("group_type")
        except Exception:
            logger.exception("Data exploration tab - Error reading group type in SELECTED_STUDIES_FILE")
            #group_selection = None
            return {}, {}, {}

        #if not group_selection:
        #    return {}, {}, {}

        try:
            df = do_preprocess()
            print('data on page')
            print(df)
            if df.empty:
                return {}, {}, {}
            fig_pca, fig_residual = pca_plot(df)
            fig_box          = box_plot(df)
            return fig_pca, fig_residual, fig_box
        except Exception:
            logger.exception("Data exploration tab - Error in preprocess, not triggered by refresh button")
            return {}, {}, {}
    
    # Save only the study details
    @callback(
        Output("confirm-study-details_dpp", "children"),
        Input("confirm-study-details_dpp", "n_clicks"),
        [
            State("selected-studies-dropdown_dpp", "value"),
            State("side-control-group_dpp",        "value"),
            State("side-case-group_dpp",           "value"),
            State("side-outliers_dpp",             "value"),
        ],
        prevent_initial_call=True
    )
    def save_study_details(n_clicks, selected_study, control, case, outliers):
        if not n_clicks or not selected_study:
            raise PreventUpdate

        # Sanitize: strip leading "✓ " if present
        if isinstance(selected_study, str) and selected_study.startswith("✓ "):
            selected_study = selected_study[2:]

        # collapse list → single study
        if isinstance(selected_study, list):
            selected_study = selected_study[0] if selected_study else None
        if not selected_study:
            raise PreventUpdate

        # Load or init the JSON payload
        try:
            with open(SELECTED_STUDIES_FILE, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}

        data.setdefault("studies", {})
        study_dict = data["studies"].setdefault(selected_study, {})

        # Persist only the group_filter and outliers
        study_dict["group_filter"] = {
            "Control": control or [],
            "Case":    case    or []
        }
        study_dict["outliers"] = outliers or ""

        # Write back to disk
        with open(SELECTED_STUDIES_FILE, "w") as f:
            json.dump(data, f, indent=2)

        # Leave the button text unchanged
        return no_update

    # Callback to control clicks on graphs for outlier selection
    @callback(
        Output("active-input-store_dpp", "data"),
        Input("div-outliers_dpp", "n_clicks")
    )
    def update_active_input_store_dpp(n_outliers):
        if not n_outliers:
            raise PreventUpdate
        return "side-outliers_dpp"
    
    # Callback to control values for study details based on clicks and values already saved in the file
    @callback(
        [
            Output("side-control-group_dpp", "options"),
            Output("side-case-group_dpp",    "options"),
            Output("side-outliers_dpp",      "value"),
            Output("side-control-group_dpp", "value"),
            Output("side-case-group_dpp",    "value"),
        ],
        [
            Input("selected-studies-dropdown_dpp", "value"),
            Input("pca-graph_dpp",       "clickData"),
            Input("residual-graph_dpp",  "clickData"),
            Input("box-graph_dpp",       "clickData"),
        ],
        [
            State("side-outliers_dpp",      "value"),
            State("side-control-group_dpp", "value"),
            State("side-case-group_dpp",    "value"),
        ],
        prevent_initial_call=True
    )
    def study_details_sidebar_update_by_saved_and_clicks(
        selected_study, pca_click, res_click, box_click,
        current_outliers, current_control, current_case
    ):
        if not selected_study:
            raise PreventUpdate
        
        # Sanitize: strip leading "✓ " if present
        if isinstance(selected_study, str) and selected_study.startswith("✓ "):
            selected_study = selected_study[2:]

        if isinstance(selected_study, list):
            selected_study = selected_study[0] if selected_study else None
        if not selected_study:
            raise PreventUpdate

        folder = os.path.join(UPLOAD_FOLDER, selected_study)
        if not os.path.isdir(folder):
            return [], [], "", [], []

        # Load saved info
        try:
            payload = json.load(open(SELECTED_STUDIES_FILE))
            study_info = payload.get("studies", {}).get(selected_study, {})
        except Exception:
            logger.exception("Data exploration tab - Error reading SELECTED_STUDIES_FILE")
            study_info = {}

        saved_group   = study_info.get("group_type", "")
        file_outliers = study_info.get("outliers", "")

        # Build Control/Case options (same as before)…
        details = read_study_details_dpp(folder)
        src     = details.get("Dataset Source", "").lower()
        options = []
        if saved_group:
            if src in ("metabolomics workbench", "original data - refmet ids", "original data - chebi ids"):
                csvs = [f for f in os.listdir(folder) if f.endswith(".csv")]
                if csvs:
                    df = pd.read_csv(os.path.join(folder, csvs[0]))
                    if "Class" in df.columns and not df.empty:
                        classes = [c.strip() for c in str(df.iloc[0]["Class"]).split("|")]
                        if saved_group in classes:
                            idx    = classes.index(saved_group)
                            labels = sorted({str(r).split("|")[idx].strip() for r in df["Class"]})
                            options = [{"label": l, "value": l} for l in labels]
            elif src == "metabolights":
                files = sorted(glob.glob(os.path.join(folder, "s_*.txt")))
                if files:
                    md = pd.read_csv(files[0], sep="\t", encoding="unicode_escape")
                    if saved_group in md.columns:
                        vals    = sorted(md[saved_group].dropna().unique())
                        options = [{"label": str(v), "value": str(v)} for v in vals]

        # Parse saved group_filter into lists
        raw_filter = study_info.get("group_filter", {})
        if isinstance(raw_filter, str):
            s = raw_filter.strip()
            try:
                raw_filter = json.loads(s)
            except Exception:
                try:
                    raw_filter = ast.literal_eval(s)
                except Exception:
                    logger.exception("Data exploration tab - Error parsing saved group_filter into lists")
                    raw_filter = {}
        if not isinstance(raw_filter, dict):
            raw_filter = {}

        def ensure_list(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str) and x:
                return [x]
            return []

        file_control = ensure_list(raw_filter.get("Control", []))
        file_case    = ensure_list(raw_filter.get("Case",   []))

        # Now figure out what triggered us
        ctx  = callback_context
        trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        # 1) Study changed → load everything from file
        if trig == "selected-studies-dropdown_dpp" or trig is None:
            return options, options, file_outliers, file_control, file_case

        # 2) Graph clicked → update outliers, keep whatever was in the dropdowns
        if trig in ("pca-graph_dpp", "residual-graph_dpp", "box-graph_dpp"):
            # figure out new label
            new_label = None
            if trig == "pca-graph_dpp" and pca_click:
                new_label = pca_click["points"][0].get("hovertext", "")
            elif trig == "residual-graph_dpp" and res_click:
                new_label = res_click["points"][0].get("hovertext", "")
            elif trig == "box-graph_dpp" and box_click:
                pt = box_click["points"][0]
                new_label = pt.get("customdata") or pt.get("hovertext") or pt.get("x")

            if new_label:
                base = current_outliers or file_outliers or ""
                labels = [l.strip() for l in base.split(",") if l.strip()]
                if new_label not in labels:
                    labels.append(new_label)
                updated = ", ".join(labels)
            else:
                updated = current_outliers or file_outliers or ""

            # return: (options, options, outliers_value, control_value, case_value)
            return options, options, updated, current_control, current_case

        # Fallback (shouldn’t happen)
        return options, options, file_outliers, file_control, file_case

    # Callback to toggle activation of exploration tab
    @callback(
        Output("no-data-modal_dpp", "is_open"),
        [Input("data_pre_process_tabs", "active_tab"),
        Input("new-study-submit_dpp", "n_clicks")],
        [State("no-data-modal_dpp", "is_open"),
        State("uploaded-file-store_dpp", "data")]
    )
    def toggle_no_data_modal_dpp(active_tab, submit_clicks, is_open, uploaded_data):
        ctx = callback_context
        if active_tab == "exploration" and (not uploaded_data or len(uploaded_data) == 0):
            return True
        if ctx.triggered and "new-study-submit_dpp" in ctx.triggered[0]['prop_id']:
            return False
        return is_open

    # Callback to control mutual exclusion of preprocessing options and preprocessing flow
    @callback(
        [
            Output("prebuilt-flows_dpp",            "value"),
            Output("missing-values-checklist_dpp",  "value"),
            Output("transformation-checklist_dpp",  "value"),
            Output("standardisation-checklist_dpp", "value"),
            Output("prev-transformation_dpp",       "data"),
            Output("prev-standardisation_dpp",      "data"),
        ],
        [
            Input("data_pre_process_tabs",          "active_tab"),
            Input("selected-studies-dropdown_dpp",  "value"),
            Input("prebuilt-flows_dpp",             "value"),   
            Input("missing-values-checklist_dpp",   "value"),
            Input("transformation-checklist_dpp",   "value"),
            Input("standardisation-checklist_dpp",  "value"),
        ],
        [
            State("prev-transformation_dpp",        "data"),
            State("prev-standardisation_dpp",       "data"),
        ],
        prevent_initial_call=True
    )
    def merged_preprocessing_options(
        active_tab, selected_study,
        prebuilt_vals, missing_vals,
        trans_vals, std_vals,
        prev_trans, prev_std
    ):
        ctx = callback_context
        trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else None

        # Helpers to normalize/ensure lists
        def to_list(v):
            return v if isinstance(v, list) else []
        def ensure_mv(v):
            lst = to_list(v)
            return lst if lst else ["knn_imputer"]

        # 1) Normalize prebuilt flow to at most one selection
        prebuilt = to_list(prebuilt_vals)
        if len(prebuilt) > 1:
            prebuilt = [prebuilt[-1]]

        # 2) Reset everything if not on exploration tab or no study selected
        if active_tab != "exploration" or not selected_study:
            return [], [], [], [], [], []
        
        # Sanitize: strip leading "✓ " if present
        if isinstance(selected_study, str) and selected_study.startswith("✓ "):
            selected_study = selected_study[2:]

        # 3) Populate from saved settings on tab or study change
        if trigger in ["data_pre_process_tabs.active_tab", "selected-studies-dropdown_dpp.value"]:
            try:
                with open(SELECTED_STUDIES_FILE) as f:
                    data = json.load(f)
                saved = data.get("studies", {}) \
                            .get(selected_study, {}) \
                            .get("preprocessing", None)
            except Exception:
                logger.exception("Data exploration tab - Error reading SELECTED_STUDIES_FILE")
                saved = None

            flows = [
                os.path.splitext(fn)[0]
                for fn in os.listdir(DATA_FLOWS_DIR)
                if fn.endswith(".txt")
            ]

            # If it's a single prebuilt flow
            if isinstance(saved, list) and len(saved) == 1 and saved[0] in flows:
                return [saved[0]], [], [], [], [], []

            # Default custom steps
            if not saved:
                saved = ["knn_imputer", "log_transform", "standard_scaler"]

            mv_list = [s for s in saved if s in ['knn_imputer','mean_imputer','iterative_imputer']]
            tr_list = [s for s in saved if s in ['log_transform','cube_root']]
            st_list = [s for s in saved if s in ['standard_scaler','min_max_scaler','robust_scaler','max_abs_scaler']]

            mv_val = mv_list[0] if mv_list else None
            tr_val = tr_list[0] if tr_list else None
            st_val = st_list[0] if st_list else None

            return (
                [],                                      # prebuilt
                [mv_val] if mv_val else ["knn_imputer"], # missing
                [tr_val] if tr_val else [],              # transformation
                [st_val] if st_val else [],              # standardisation
                [tr_val] if tr_val else [],              # prev_trans
                [st_val] if st_val else []               # prev_std
            )

        # 4) Prebuilt flow changed
        if trigger == "prebuilt-flows_dpp.value":
            if prebuilt:
                # Selected a flow → clear custom options
                return prebuilt, [], [], [], [], []
            else:
                # Cleared flow → keep customs, enforce missing default
                mv = ensure_mv(missing_vals)
                tv = to_list(trans_vals)
                sv = to_list(std_vals)
                return [], mv, tv, sv, tv, sv

        # 5) Missing-values changed → clear flow & enforce default
        if trigger == "missing-values-checklist_dpp.value":
            mv = ensure_mv(missing_vals)
            tv = to_list(trans_vals)
            sv = to_list(std_vals)
            return [], mv, tv, sv, tv, sv

        # 6) Transformation toggled
        if trigger == "transformation-checklist_dpp.value":
            curr = to_list(trans_vals)
            # If >1 selected, pick the newly added one
            if len(curr) > 1:
                new = [x for x in curr if x not in to_list(prev_trans)]
                chosen = new[0] if new else curr[-1]
                new_tr = [chosen]
            else:
                new_tr = curr
            # Click-again to deselect
            if new_tr == to_list(prev_trans):
                new_tr = []
            mv = ensure_mv(missing_vals)
            sv = to_list(std_vals)
            return [], mv, new_tr, sv, new_tr, sv

        # 7) Standardisation toggled
        if trigger == "standardisation-checklist_dpp.value":
            curr = to_list(std_vals)
            if len(curr) > 1:
                new = [x for x in curr if x not in to_list(prev_std)]
                chosen = new[0] if new else curr[-1]
                new_st = [chosen]
            else:
                new_st = curr
            if new_st == to_list(prev_std):
                new_st = []
            mv = ensure_mv(missing_vals)
            tv = to_list(trans_vals)
            return [], mv, tv, new_st, tv, new_st

        # 8) Fallback
        return (
            prebuilt,
            ensure_mv(missing_vals),
            to_list(trans_vals),
            to_list(std_vals),
            to_list(prev_trans),
            to_list(prev_std)
        )
    
    # Callback to show pop up to save preprocessing options as a pre-built flow
    @callback(
        Output("save-flow-modal_dpp", "is_open"),
        [
            Input("save-options-btn_dpp", "n_clicks"),
            Input("confirm-save-flow-btn_dpp", "n_clicks")
        ],
        [State("save-flow-modal_dpp", "is_open")]
    )
    def toggle_save_flow_modal(n_save_options, n_confirm, is_open):
        ctx = callback_context
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "save-options-btn_dpp":
            return True
        elif button_id == "confirm-save-flow-btn_dpp":
            return False
        return is_open
    
    # Callback to save preprocessing options as pre-built flow and save preprocessing for study
    @callback(
        Output("dummy-save-status_dpp", "children"),
        [
            Input("confirm-save-flow-btn_dpp", "n_clicks"),
            Input("confirm-data-processing_dpp", "n_clicks"),
        ],
        [
            State("new-flow-name-input", "value"),
            State("selected-studies-dropdown_dpp", "value"),
            State("prebuilt-flows_dpp", "value"),
            State("missing-values-checklist_dpp", "value"),
            State("transformation-checklist_dpp", "value"),
            State("standardisation-checklist_dpp", "value"),
        ],
        prevent_initial_call=True
    )
    def handle_multiple_save_buttons(n_save_clicks, n_confirm_clicks,
                                    flow_name, selected_study, prebuilt_flow_opt,
                                    missing_value_opt, transformation_opt, standardisation_opt):
        ctx = callback_context
        triggered = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # helper to pull a single string out of a list (or None)
        def unpack(opt_list):
            return opt_list[0] if isinstance(opt_list, list) and opt_list else None

        prebuilt_flow   = unpack(prebuilt_flow_opt)
        missing_value   = unpack(missing_value_opt)
        transformation  = unpack(transformation_opt)
        standardisation = unpack(standardisation_opt)

        # Sanitize: strip leading "✓ " if present
        if isinstance(selected_study, str) and selected_study.startswith("✓ "):
            selected_study = selected_study[2:]

        # If saving a prebuilt flow to a file.
        if triggered == "confirm-save-flow-btn_dpp":
            if not flow_name:
                return "Please provide a name for the preprocessing flow."
            filename = flow_name.replace(" ", "-") + ".txt"
            folder = DATA_FLOWS_DIR
            os.makedirs(folder, exist_ok=True)
            file_path = os.path.join(folder, filename)
            data = {
                "missing_values": missing_value,
                "transformation": transformation,
                "standardisation": standardisation
            }
            with open(file_path, "w") as f:
                json.dump(data, f)
            return f"Flow saved as {filename}"

        # If saving data processing options for a study.
        elif triggered == "confirm-data-processing_dpp":
            if not selected_study:
                raise PreventUpdate

            # Determine the preprocessing steps.
            if prebuilt_flow:
                steps = get_flow_steps(prebuilt_flow)
            else:
                # Combine steps ensuring each input is a list element.
                custom_steps = (
                    ensure_list(missing_value) +
                    ensure_list(transformation) +
                    ensure_list(standardisation)
                )
                steps = custom_steps
            # Load the selected studies file.
            data = {}
            if os.path.exists(SELECTED_STUDIES_FILE):
                try:
                    with open(SELECTED_STUDIES_FILE, "r") as f:
                        data = json.load(f)
                except Exception:
                    data = {}
            # Make sure the "studies" key exists.
            if "studies" not in data:
                data["studies"] = {}
            # Ensure there is an entry for the selected study.
            if selected_study not in data["studies"]:
                # Initialize with default group_type and an empty preprocessing list.
                data["studies"][selected_study] = {"group_type": None, "preprocessing": []}
                print('selected study not in studies')
            # Save the steps under the "preprocessing" key for the selected study.
            data["studies"][selected_study]["preprocessing"] = steps
            try:
                with open(SELECTED_STUDIES_FILE, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                logger.exception("Data exploration tab - Error saving preprocessing steps")

            return ""
        
        raise PreventUpdate

    # Callback to populate the pre-built preprocessing flow list and the preprocessing steps shown in the tooltip when hoving over a pre-bulit flow
    @callback(
        [Output("prebuilt-flows_dpp", "options"),
        Output("prebuilt-tooltips-container_dpp", "children")],
        [Input("confirm-save-flow-btn_dpp", "n_clicks"),
        Input("update-prebuilt-flows-interval_dpp", "n_intervals")]
    )
    def update_prebuilt_flows(n_save, n_intervals):
        # Mapping from processing option values to display label names.
        option_labels = {
            "knn_imputer": "KNN Imputer",
            "mean_imputer": "Mean Imputer",
            "iterative_imputer": "Iterative Imputer",
            "log_transform": "Log Transform",
            "cube_root": "Cube Root Transform",
            "standard_scaler": "Standard Scaler",
            "min_max_scaler": "Min-Max Scaler",
            "robust_scaler": "Robust Scaler",
            "max_abs_scaler": "Max Abs Scaler"
        }
        
        folder = DATA_FLOWS_DIR
        options = []
        tooltips = []
        
        if os.path.exists(folder):
            # Loop through each JSON file in the folder.
            for flow in sorted(os.listdir(folder)):
                if flow.endswith(".txt"):
                    flow_id = os.path.splitext(flow)[0]
                    display_name = flow_id.replace("-", " ")
                    # Create an option with a Span that has a unique id for the tooltip target.
                    options.append({
                        "label": html.Span(display_name, id=f"prebuilt-{flow_id}"),
                        "value": flow_id
                    })
                    file_path = os.path.join(folder, flow)
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                    except Exception:
                        data = {}
                    # Combine processing steps from the different sections.
                    steps = []
                    for key in ["missing_values", "transformation", "standardisation"]:
                        file_steps = data.get(key, [])
                        # Ensure file_steps is a list (if a single value is saved, wrap it in a list).
                        if isinstance(file_steps, str):
                            file_steps = [file_steps]
                        steps.extend(file_steps)
                    # Build the tooltip content using the display labels.
                    if steps:
                        labels = [option_labels.get(step, step) for step in steps]
                        # Create an unordered list: each list item is a complete label.
                        tooltip_content = html.Ul([html.Li(item) for item in labels])
                    else:
                        tooltip_content = "No processing options saved."
                    tooltips.append(
                        dbc.Tooltip(
                            tooltip_content,
                            target=f"prebuilt-{flow_id}",
                            placement="top",
                            delay={"show": 500, "hide": 100}
                        )
                    )
        
        return options, tooltips

    # ─── PCA Modal Toggle ────────────────────────────────────────────────────────────
    @callback(
        Output("save-plot-modal-pca_dpp", "is_open"),
        [
            Input("save-pca-btn_dpp",     "n_clicks"),
            Input("confirm-save-pca_dpp", "n_clicks"),
            Input("cancel-save-pca_dpp",  "n_clicks"),
        ],
        [
            State("save-plot-modal-pca_dpp", "is_open"),
            State("plot-name-input-pca_dpp", "value"),
        ],
        prevent_initial_call=True
    )
    def toggle_pca_modal(open_n, confirm_n, cancel_n, is_open, filename):
        ctx = callback_context
        if not ctx.triggered:
            # no trigger → no change
            return is_open

        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger == "save-pca-btn_dpp":
            # always open on initial save button click
            return True

        if trigger == "cancel-save-pca_dpp":
            # always close on cancel
            return False

        if trigger == "confirm-save-pca_dpp":
            # only close if a filename is provided
            if filename and filename.strip():
                return False
            else:
                # keep open if no filename
                return True

        # fallback
        return is_open


    # ─── PCA Save Callback ───────────────────────────────────────────────────────────
    @callback(
        Output("save-feedback-pca_dpp", "children"),
        Input("confirm-save-pca_dpp", "n_clicks"),
        [
            State("dropdown-existing-projects", "value"),
            State("input-analysis-project", "value"),
            State("selected-studies-dropdown_dpp", "value"),
            State("plot-name-input-pca_dpp",        "value"),
            State("pca-graph_dpp",                  "figure"),
        ],
        prevent_initial_call=True
    )
    def save_pca_plot(n_clicks, selected_project, new_project, study, filename, fig_json):    
        project_name = new_project if new_project else selected_project  
        # Sanitize: strip leading "✓ " if present
        if isinstance(study, str) and study.startswith("✓ "):
            study = study[2:]
        try:
            # Rehydrate from JSON
            fig = pio.from_json(json.dumps(fig_json))
            w   = fig.layout.width  or 700
            h   = fig.layout.height or 400

            # Build path
            project = project_name.replace(" ", "-")
            out_dir = os.path.join(
                APP_ROOT,
                "Projects",
                project,
                "Plots",
                "Preprocessing-analysis",
                "PCA-plots"
            )
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{filename}.svg")

            # Write out
            pio.write_image(fig, path, format="svg", width=int(w), height=int(h))

        except Exception:
            logger.exception(f"Data exploration tab - Error saving PCA plot '{filename}.svg'")
            # keep the modal open, no UI‐change
            return no_update

        else:
            # only runs if no exception was raised
            return no_update


    # ─── Residuals Modal Toggle ──────────────────────────────────────────────────────
    @callback(
        Output("save-plot-modal-residual_dpp", "is_open"),
        [
            Input("save-residual-btn_dpp",     "n_clicks"),
            Input("confirm-save-residual_dpp", "n_clicks"),
            Input("cancel-save-residual_dpp",  "n_clicks"),
        ],
        [
            State("save-plot-modal-residual_dpp", "is_open"),
            State("plot-name-input-residual_dpp", "value"),  
        ],
        prevent_initial_call=True
    )
    def toggle_residual_modal(open_n, confirm_n, cancel_n, is_open, filename):
        ctx = callback_context
        if not ctx.triggered:
            # no trigger → no change
            return is_open

        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger == "save-residual-btn_dpp":
            # always open on initial save button click
            return True

        if trigger == "cancel-save-residual_dpp":
            # always close on cancel
            return False

        if trigger == "confirm-save-residual_dpp":
            # only close if a filename is provided
            if filename and filename.strip():
                return False
            else:
                # keep open if no filename
                return True

        # fallback
        return is_open


    # ─── Residuals Save Callback ─────────────────────────────────────────────────────
    @callback(
        Output("save-feedback-residual_dpp", "children"),
        Input("confirm-save-residual_dpp", "n_clicks"),
        [
            State("dropdown-existing-projects", "value"),
            State("input-analysis-project", "value"),
            State("selected-studies-dropdown_dpp", "value"),
            State("plot-name-input-residual_dpp", "value"),
            State("residual-graph_dpp", "figure"),
        ],
        prevent_initial_call=True
    )
    def save_residual_plot(n_clicks, selected_project, new_project, study, filename, fig_json):
        project_name = new_project if new_project else selected_project 
        # Sanitize: strip leading "✓ " if present
        if isinstance(study, str) and study.startswith("✓ "):
            study = study[2:]
        try:
            # Rehydrate from JSON
            fig = pio.from_json(json.dumps(fig_json))
            w   = fig.layout.width  or 700
            h   = fig.layout.height or 400

            # Build path
            project = project_name.replace(" ", "-")
            out_dir = os.path.join(
                APP_ROOT,
                "Projects",
                project,
                "Plots",
                "Preprocessing-analysis",
                "Residual-plots"
            )
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{filename}.svg")

            # Write out
            pio.write_image(fig, path, format="svg", width=int(w), height=int(h))

        except Exception:
            logger.exception(f"Data exploration tab - Error saving residual plot '{filename}.svg'")
            return no_update

        else:
            # On success, likewise leave modal & feedback alone
            return no_update


    # ─── Boxplot Modal Toggle ────────────────────────────────────────────────────────
    @callback(
        Output("save-plot-modal-box_dpp", "is_open"),
        [
            Input("save-box-btn_dpp",     "n_clicks"),
            Input("confirm-save-box_dpp", "n_clicks"),
            Input("cancel-save-box_dpp",  "n_clicks"),
        ],
        [
            State("save-plot-modal-box_dpp", "is_open"),
            State("plot-name-input-box_dpp", "value"),  # your filename input
        ],
        prevent_initial_call=True
    )
    def toggle_box_modal(open_n, confirm_n, cancel_n, is_open, filename):
        ctx = callback_context
        if not ctx.triggered:
            return is_open

        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger == "save-box-btn_dpp":
            return True

        if trigger == "cancel-save-box_dpp":
            return False

        if trigger == "confirm-save-box_dpp":
            # only close if a filename is provided
            if filename and filename.strip():
                return False
            return True

        return is_open


    # ─── Boxplot Save Callback ───────────────────────────────────────────────────────
    @callback(
        Output("save-feedback-box_dpp", "children"),
        Input("confirm-save-box_dpp", "n_clicks"),
        [
            State("dropdown-existing-projects", "value"),
            State("input-analysis-project", "value"),
            State("selected-studies-dropdown_dpp", "value"),
            State("plot-name-input-box_dpp",        "value"),
            State("box-graph_dpp",                  "figure"),
        ],
        prevent_initial_call=True
    )
    def save_box_plot(n_clicks, selected_project, new_project, study, filename, fig_json):
        project_name = new_project if new_project else selected_project 
        # Sanitize: strip leading "✓ " if present
        if isinstance(study, str) and study.startswith("✓ "):
            study = study[2:]
        try:
            # Rehydrate from JSON
            fig = pio.from_json(json.dumps(fig_json))
            w   = fig.layout.width  or 700
            h   = fig.layout.height or 400

            # Build path
            project = project_name.replace(" ", "-")
            out_dir = os.path.join(
                APP_ROOT,
                "Projects",
                project,
                "Plots",
                "Preprocessing-analysis",
                "Box-plots"
            )
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{filename}.svg")

            # Write out
            pio.write_image(fig, path, format="svg", width=int(w), height=int(h))

        except Exception:
            logger.exception(f"Data exploration tab - Error saving box plot '{filename}.svg'")
            return no_update

        else:
            # On success, likewise leave modal & feedback alone
            return no_update
        
    # ─── Success/Error Message Handling Callback ───────────────────────────────────────────────────────
    @callback(
        [
            Output("save-toast", "is_open"),
            Output("save-toast", "children"),
            Output("save-toast", "header"),
            Output("save-toast", "icon"),
        ],
        [
            Input("confirm-study-details_dpp",      "n_clicks"),
            Input("confirm-data-processing_dpp",    "n_clicks"),
            Input("confirm-save-pca_dpp",           "n_clicks"),
            Input("confirm-save-residual_dpp",      "n_clicks"),
            Input("confirm-save-box_dpp",           "n_clicks"),
            Input("summary-tab_dpp",                "n_clicks"),  
        ],
        [
            State("summary-tab_dpp",                "disabled"),   
            State("selected-studies-dropdown_dpp",  "value"),      
            State("selected-study-store_dpp",       "data"),      
            # PCA
            State("plot-name-input-pca_dpp",       "value"),
            State("pca-graph_dpp",                 "figure"),
            # Residual
            State("plot-name-input-residual_dpp",  "value"),
            State("residual-graph_dpp",            "figure"),
            # Box
            State("plot-name-input-box_dpp",       "value"),
            State("box-graph_dpp",                 "figure"),
        ],
        prevent_initial_call=True
    )
    def show_save_toast(
        n_click_details, n_click_options,
        n_click_pca, n_click_residual, n_click_box,
        n_click_summary_tab,
        summary_disabled,
        selected_dropdown, selected_store,
        pca_filename,      pca_fig,
        res_filename,      res_fig,
        box_filename,      box_fig
    ):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        btn = ctx.triggered[0]["prop_id"].split(".")[0]

        # Default toast values
        header = "Error"
        icon   = "danger"
        message = ""

        # Sanitize: strip leading "✓ " if present
        if isinstance(selected_dropdown, str) and selected_dropdown.startswith("✓ "):
            selected_dropdown = selected_dropdown[2:]

        # Normalize study name for single-study buttons
        study_name = None
        if isinstance(selected_dropdown, list):
            study_name = selected_dropdown[0] if selected_dropdown else None
        else:
            study_name = selected_dropdown
        study_label = study_name or "the study"

        if btn == "confirm-study-details_dpp":
            header, icon = "Success", "success"
            message = f"Study details have been saved for {study_label}."
            logger.info(f"Data exploration tab - Study details have been saved for {study_label}")

        elif btn == "confirm-data-processing_dpp":
            header, icon = "Success", "success"
            message = f"Pre-processing options have been saved for {study_label}."
            logger.info(f"Data exploration tab - Pre-processing options have been saved for {study_label}")

        elif btn == "confirm-save-pca_dpp":
            if not pca_filename:
                message = f"Please enter a filename for the PCA plot for {study_label}."
                logger.warning(f"Data exploration tab - No filename given for the PCA plot for {study_label}")
            elif not pca_fig:
                message = f"No PCA plot data available to save for {study_label}."
                logger.warning(f"Data exploration tab - No PCA plot data available to save for {study_label}")
            else:
                header, icon = "Success", "success"
                message = f"PCA plot '{pca_filename}.svg' has been saved for {study_label}."
                logger.info(f"Data exploration tab - PCA plot '{pca_filename}.svg' has been saved for {study_label}")

        elif btn == "confirm-save-residual_dpp":
            if not res_filename:
                message = f"Please enter a filename for the residuals plot for {study_label}."
                logger.warning(f"Data exploration tab - No filename given for the residuals plot for {study_label}")
            elif not res_fig:
                message = f"No residuals plot data available to save for {study_label}."
                logger.warning(f"Data exploration tab - No residuals plot data available to save for {study_label}")
            else:
                header, icon = "Success", "success"
                message = f"Residuals plot '{res_filename}.svg' has been saved for {study_label}."
                logger.info(f"Data exploration tab - Residuals plot '{res_filename}.svg' has been saved for {study_label}")

        elif btn == "confirm-save-box_dpp":
            if not box_filename:
                message = f"Please enter a filename for the box plot for {study_label}."
                logger.warning(f"Data exploration tab - No filename given for the box plot for {study_label}")
            elif not box_fig:
                message = f"No box plot data available to save for {study_label}."
                logger.warning(f"Data exploration tab - No box plot data available to save for {study_label}")
            else:
                header, icon = "Success", "success"
                message = f"Box plot '{box_filename}.svg' has been saved for {study_label}."
                logger.info(f"Data exploration tab - Box plot '{res_filename}.svg' has been saved for {study_label}")

        else:
            # unknown trigger
            raise PreventUpdate

        return True, message, header, icon 
    