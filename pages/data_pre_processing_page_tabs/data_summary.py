# pages/data_pre_processing_page_tabs/data_summary.py
from .shared_functions.helper import (read_study_details_dpp,
                                      get_flow_steps)
from .shared_functions.data_processing import (static_preprocess_workbench,
                                               static_preprocess)
from dash import html, dcc, callback, Input, Output, State, dash_table, no_update, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import os
import json
import glob
import requests
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import numpy as np
import re
import numbers
import ast
import pandas as pd
import time
import logging
from io import StringIO
logger = logging.getLogger(__name__)


UPLOAD_FOLDER = "pre-processed-datasets"
# Path to the temporary file that holds the selected studies
SELECTED_STUDIES_FILE = os.path.join(UPLOAD_FOLDER, "selected_studies_temp.txt")

###### REMOVE THE RELIANCE ON BELOW #######
default_metadata = pd.DataFrame({
    "Sample Name": ["Sample1", "Sample2", "Sample3"],
    "Group": ["Control", "Treatment", "Control"]
})
default_md_filter = {"Group": ["Control", "Treatment"]}

# ================================== #
# Layout of the Data summary tab #
# ================================== #

layout = html.Div([
                    html.H2("Data Summary", style={"fontFamily": "Arial"}),
                    dcc.Interval(
                        id="processed-file-check-interval_dpp",
                        interval=2000,  # 2000ms = 2 seconds
                        n_intervals=0,
                        disabled=True  # We'll enable it when processing is complete
                    ),
                    # Dropdown to select a study
                    dcc.Dropdown(
                        id="selected-studies-dropdown-summary_dpp",
                        placeholder="Select a study",
                        options=[],  # Updated by a callback
                        value=None,
                        style={"width": "300px", "margin": "1rem auto"}
                    ),
                    dbc.Row([
                        # Left column: processed data preview wrapped in Collapse
                        dbc.Col(
                            [
                                # 1) smaller, left-aligned button
                                dbc.Button(
                                    "Process Data",
                                    id="process-data-btn_dpp",
                                    color="primary",
                                    size="md",              # md or sm will shrink the padding/font
                                    className="mt-3 mb-3",  # spacing
                                    style={"width": "180px"}  # or "auto" if you prefer
                                ),

                                # 2) your collapse
                                dbc.Collapse(
                                    [
                                        html.H4("Processed Data Preview"),
                                        html.Div(id="processed-data-table_dpp"),
                                        html.Div(
                                            id="process-data-progress-bar_dpp",
                                            style={
                                                "display": "flex",
                                                "justifyContent": "center",
                                                "alignItems": "center",
                                            },
                                        ),
                                    ],
                                    id="processed-data-collapse_dpp",
                                    is_open=False,
                                ),
                            ],
                            width=9,
                        ),
                        # Right column: Preprocessing sidebar
                        dbc.Col(
                            html.Div(
                                [
                                    # 1) Study details sidebar (read-only)
                                    html.Div(
                                        [
                                            # header (no button here, since it’s summary only)
                                            html.H4("Study details", style={"margin": 0, "marginBottom": "1rem"}),

                                            # Outliers as a disabled text input
                                            html.Div(
                                                [
                                                    dbc.Label("Outliers"),
                                                    dbc.Input(
                                                        id="summary-side-outliers_dpp",
                                                        type="text",
                                                        disabled=True,
                                                        style={"backgroundColor": "#f9f9f9"}
                                                    ),
                                                ],
                                                className="mb-3"
                                            ),

                                            # Control group as a disabled multi-dropdown
                                            html.Div(
                                                [
                                                    dbc.Label("Control group"),
                                                    dcc.Dropdown(
                                                        id="summary-side-control-group_dpp",
                                                        multi=True,
                                                        disabled=True,
                                                        style={"backgroundColor": "#f9f9f9"}
                                                    ),
                                                ],
                                                className="mb-3"
                                            ),

                                            # Case group as a disabled multi-dropdown
                                            html.Div(
                                                [
                                                    dbc.Label("Case group"),
                                                    dcc.Dropdown(
                                                        id="summary-side-case-group_dpp",
                                                        multi=True,
                                                        disabled=True,
                                                        style={"backgroundColor": "#f9f9f9"}
                                                    ),
                                                ],
                                                className="mb-3"
                                            ),
                                        ],
                                        style={
                                            "padding": "1rem",
                                            "border": "1px solid #ccc",
                                            "borderRadius": "5px",
                                        },
                                    ),

                                    html.Br(),

                                    # 2) Data processing sidebar (read-only)
                                    html.Div(
                                        [
                                            html.H4("Data Processing", style={"margin": 0, "marginBottom": "1rem"}),

                                            html.H6("Missing Values Imputation", className="mt-4"),
                                            dbc.Checklist(
                                                id="summary-missing-values-checklist_dpp",
                                                options=[
                                                    {"label": "KNN Imputer",      "value": "knn_imputer"},
                                                    {"label": "Mean Imputer",     "value": "mean_imputer"},
                                                    {"label": "Iterative Imputer","value": "iterative_imputer"},
                                                ],
                                                value=[],  # will be set by callback
                                                inline=False,
                                                style={"pointerEvents": "none", "paddingLeft": "1rem"}
                                            ),

                                            html.H6("Transformation", className="mt-3"),
                                            dbc.Checklist(
                                                id="summary-transformation-checklist_dpp",
                                                options=[
                                                    {"label": "Log Transform",       "value": "log_transform"},
                                                    {"label": "Cube Root Transform", "value": "cube_root"},
                                                ],
                                                value=[],  # will be set by callback
                                                inline=False,
                                                style={"pointerEvents": "none", "paddingLeft": "1rem"}
                                            ),

                                            html.H6("Standardisation", className="mt-3"),
                                            dbc.Checklist(
                                                id="summary-standardisation-checklist_dpp",
                                                options=[
                                                    {"label": "Standard Scaler", "value": "standard_scaler"},
                                                    {"label": "Min-Max Scaler",  "value": "min_max_scaler"},
                                                    {"label": "Robust Scaler",   "value": "robust_scaler"},
                                                    {"label": "Max Abs Scaler",  "value": "max_abs_scaler"},
                                                ],
                                                value=[],  # will be set by callback
                                                inline=False,
                                                style={"pointerEvents": "none", "paddingLeft": "1rem"}
                                            ),
                                        ],
                                        style={
                                            "padding": "1rem",
                                            "border": "1px solid #ccc",
                                            "borderRadius": "5px",
                                            "marginTop": "1.5rem",
                                        },
                                    ),
                                ],
                                style={"display": "flex", "flexDirection": "column"},
                            ),
                            width=3,
                        )
                    ])
                ], style={"padding": "1rem"})

def register_callbacks():
    # Callback to enable the data summary tab if all details have been given for all studies
    @callback(
        [
            Output("summary-tab_dpp",             "disabled"),
            Output("summary-check-interval_dpp",  "disabled"),
            Output("summary-check-interval_dpp",  "n_intervals"),
        ],
        [
            Input("confirm-study-details_dpp",   "n_clicks"),
            Input("confirm-data-processing_dpp", "n_clicks"),
            Input("summary-check-interval_dpp",  "n_intervals"),
        ],
        State("selected-study-store_dpp", "data"),
        prevent_initial_call=True
    )
    def delayed_enable_summary(confirm_details, confirm_processing, n_intervals, selected_studies):
        triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]

        # 1) if user just clicked *either* confirm button ➞ kick off the 2s timer
        if triggered_id in ("confirm-study-details_dpp", "confirm-data-processing_dpp"):
            # leave the tab as-is (still disabled), enable the interval, reset its counter
            return no_update, False, 0

        # 2) if the interval just fired (n_intervals == 1) ➞ do the real check
        if triggered_id == "summary-check-interval_dpp" and n_intervals == 1:
            # default: keep disabled unless all studies are good
            disabled = True

            # no studies? bail
            if selected_studies:
                try:
                    with open(SELECTED_STUDIES_FILE, "r", encoding="utf-8") as f:
                        payload = json.load(f).get("studies", {})
                except Exception:
                    logger.exception("Data summary tab - Error reading SELECTED_STUDIES_FILE")
                    payload = {}

                # check every selected study
                ok = True
                for study in selected_studies:
                    info = payload.get(study, {})
                    gf      = info.get("group_filter", {})
                    control = gf.get("Control") or []
                    case    = gf.get("Case")    or []
                    prep    = info.get("preprocessing") or []
                    if not (control and case and isinstance(prep, list) and prep):
                        ok = False
                        break
                disabled = not ok

                if ok:
                    logger.info("Data summary tab - All selected studies have complete details — enabling Data Summary tab")

            # after checking, disable the interval so it doesn't fire again
            return disabled, True, no_update

        # fallback — do nothing
        raise PreventUpdate

    # Callback to hide the process data button once clicked so the data isn't processed twice
    @callback(
        Output("process-data-btn_dpp", "style"),
        Input("process-data-btn_dpp", "n_clicks"),
        prevent_initial_call=True
    )
    def hide_process_button(n_clicks):
        if n_clicks:
            # Hide the button entirely
            return {"display": "none"}
        # Should never get here, but just in case
        return no_update

    # Callback to process the data 
    @callback(
        [Output("process-data-status_dpp", "children"),
        Output("processing-complete-store_dpp", "data")],
        [Input("process-data-btn_dpp", "n_clicks")],
        [State("selected-study-store_dpp", "data"),
        State("project-folder-store_dpp", "data")],
        prevent_initial_call=True
    )
    def process_data_for_all_studies(n_clicks, selected_studies, project_folder):
        if not n_clicks:
            raise PreventUpdate

        if not selected_studies:
            raise PreventUpdate

        # Use project_folder to build the save path, or error.
        if project_folder:
            base_save = os.path.join(project_folder, "Processed-datasets")
        else:
            logger.error("Data summary tab - project folder chosen doesn't exist")
            return html.Div("Project folder chosen doesn't exist"), False

        final_save_folder = base_save
        os.makedirs(final_save_folder, exist_ok=True)

        # Load preprocessing steps and other study-specific parameters from the selected studies file.
        steps_map = {}
        temp = {}
        if os.path.exists(SELECTED_STUDIES_FILE):
            try:
                with open(SELECTED_STUDIES_FILE) as f:
                    temp = json.load(f)
            except Exception:
                logger.exception("Data summary tab - Error reading SELECTED_STUDIES_FILE")

        # Iterate over all selected studies.
        for study in selected_studies:
            folder = os.path.join(UPLOAD_FOLDER, study)
            if not os.path.isdir(folder):
                logger.error(f"Data summary tab - No folder for study: {study}")
                continue

            details = read_study_details_dpp(folder)
            dataset_source = details.get("Dataset Source", "").lower()
            study_name = details.get("Study Name", "")
            # Retrieve outliers from the SELECTED_STUDIES_FILE payload.
            outliers = temp.get("studies", {}).get(study, {}).get("outliers")

            # If outliers is a string, convert it to a list by splitting on commas.
            if isinstance(outliers, str):
                # Split by comma and remove any extra whitespace from each entry
                outliers = [value.strip() for value in outliers.split(",") if value.strip()]
            try:
                md_filter_local = temp.get("studies", {}).get(study, {}).get("group_filter")
            except Exception:
                logger.exception(f"Data summary tab - Error parsing metadata filter for study {study}")
                #md_filter_local = default_md_filter
                continue

            # Retrieve preprocessing steps.
            preprocessing_steps = temp.get("studies", {}).get(study, {}).get("preprocessing")
            flows = [os.path.splitext(f)[0] for f in os.listdir("data_preprocessing_flows") if f.endswith(".txt")]
            if isinstance(preprocessing_steps, list) and len(preprocessing_steps) == 1 and preprocessing_steps[0] in flows:
                flow_name_for_file = preprocessing_steps[0]
                preprocessing_steps = []  # Replace with actual flow steps as needed.
            else:
                flow_name_for_file = "_".join(preprocessing_steps) if preprocessing_steps else "Untitled"

            # Build a safe filename.
            filename = f"processed_{details.get('Study Name', study)}_{flow_name_for_file}.csv"
            path = os.path.join(final_save_folder, filename)

            # ↪ record the filename back into the SELECTED_STUDIES_FILE payload
            temp["studies"][study]["filename"] = filename
            try:
                with open(SELECTED_STUDIES_FILE, "w", encoding="utf-8") as f:
                    json.dump(temp, f, indent=2)
            except Exception:
                logger.exception("Data summary tab - Error writing filename back into SELECTED_STUDIES_FILE")

            # Retrieve the confirmed group type from the temp file.
            saved_group = temp.get("studies", {}).get(study, {}).get("group_type")
            if not saved_group:
                logger.error(f"Data summary tab - Group not confirmed for study: {study}")
                return html.Div("Group not confirmed for one or more studies."), False

            group_selection = saved_group

            try:
                if dataset_source in (
                    "metabolomics workbench",
                    "original data - refmet ids",
                    "original data - chebi ids",
                ):
                    processed_df = static_preprocess_workbench(
                        folder,
                        preprocessing_steps=preprocessing_steps,
                        outliers=outliers,
                        filter=md_filter_local,
                        selected_group=group_selection, 
                        database_source=dataset_source
                    )
                    group_mapping = {g: group_type for group_type, groups in md_filter_local.items() for g in groups}
                    processed_df['group_type'] = processed_df['Group'].map(group_mapping)
                    if processed_df['group_type'].isnull().any():
                        missing_groups = processed_df.loc[processed_df['group_type'].isnull(), 'Group'].unique()
                        logger.error(f"Data summary tab - The following group names were not found in the metadata filter: {missing_groups}")

                    # 1) Drop the old Group column
                    processed_df = processed_df.drop(columns=["Group"], errors="ignore")

                    # 2) Reset index (this may create “Identifier” if your old index was named that)
                    processed_df = processed_df.reset_index()

                    # 3) Rename or drop whatever columns you need
                    processed_df = (
                        processed_df
                        .rename(columns={"index": "database_identifier"})      # in case you have an unnamed index
                        .drop(columns=["Identifier"], errors="ignore")         # drop the unwanted one
                    )

                    # 4) Reorder
                    cols = processed_df.columns.tolist()
                    rest = [c for c in cols if c not in ("database_identifier", "group_type")]
                    processed_df = processed_df[["database_identifier", "group_type"] + rest]

                    # 5) Save
                    processed_df.to_csv(path, index=False)
                else:
                    # 1) build the pattern
                    pattern = os.path.join(folder, "s_*.txt")

                    # 2) expand the pattern into actual files
                    matches = glob.glob(pattern)

                    # 3) handle zero or many matches, and pick one
                    if not matches:
                        logger.error(f"Data summary tab - No metadata file found matching pattern: {pattern!r}")
                        raise PreventUpdate
                    elif len(matches) > 1:
                        # you could choose the newest, the first, or raise an error
                        matches.sort()  # alphabetical; or sort by os.path.getmtime for newest
                    meta_filepath = matches[0]
                    #meta_filepath = os.path.join(folder, "s_*.txt")
                    if os.path.exists(meta_filepath):
                        try:
                            metadata_df = pd.read_csv(meta_filepath, sep="\t", encoding="unicode_escape")
                        except Exception:
                            logger.exception(f"Data summary tab - Error reading metadata file for study {study}")
                            #metadata_df = default_metadata
                            continue
                    else:
                        metadata_df = default_metadata

                    processed_df = static_preprocess(
                        folder, metadata_df,
                        preprocessing_steps, outliers, md_filter_local,
                        selected_group=group_selection
                    )
                    group_mapping = {g: group_type for group_type, groups in md_filter_local.items() for g in groups}
                    processed_df['group_type'] = processed_df['Group'].map(group_mapping)
                    if processed_df['group_type'].isnull().any():
                        missing_groups = processed_df.loc[processed_df['group_type'].isnull(), 'Group'].unique()
                        logger.error(f"Data summary tab - The following group names were not found in the metadata filter: {missing_groups}")
                    
                    # 1) Drop the old Group column
                    processed_df = processed_df.drop(columns=["Group"], errors="ignore")

                    # 2) Reset index (this may create “Identifier” if your old index was named that)
                    processed_df = processed_df.reset_index()

                    # 3) Rename or drop whatever columns you need
                    processed_df = (
                        processed_df
                        .rename(columns={"index": "database_identifier"})      # in case you have an unnamed index
                        .drop(columns=["Identifier"], errors="ignore")         # drop the unwanted one
                    )

                    # 4) Reorder
                    cols = processed_df.columns.tolist()
                    rest = [c for c in cols if c not in ("database_identifier", "group_type")]
                    processed_df = processed_df[["database_identifier", "group_type"] + rest]

                    # 5) Save
                    processed_df.to_csv(path, index=False)

                logger.info(f"Data summary tab - Saved {filename} -> {final_save_folder}")
            except Exception:
                logger.exception(f"Data summary tab - Error processing study {study}")
                continue

        # After processing all studies, save the SELECTED_STUDIES_FILE into the project folder 
        # under the new name "project_details_file.json".

        # Determine destination path
        if project_folder:
            dest_path = os.path.join(project_folder, "project_details_file.json")
        else:
            dest_path = "project_details_file.json"

        try:
            # 1) Load the incoming payload
            with open(SELECTED_STUDIES_FILE, "r", encoding="utf-8") as f:
                new_payload = json.load(f)
            new_studies = new_payload.get("studies", {})

            # 2) Load existing details if present, otherwise start fresh
            if os.path.exists(dest_path):
                with open(dest_path, "r", encoding="utf-8") as f:
                    existing_payload = json.load(f)
                existing_studies = existing_payload.get("studies", {})
            else:
                existing_payload = {}
                existing_studies = {}

            # 3) Merge: replace or append each new study
            for study_name, details in new_studies.items():
                if study_name in existing_studies:
                    logger.info(f"Data summary tab - Updated details for study {study_name} in {project_folder}")
                else:
                    logger.info(f"Data summary tab - Added new study {study_name} in {project_folder}")
                existing_studies[study_name] = details

            # 4) Write back the merged payload
            merged = {"studies": existing_studies}
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)


        except Exception:
            logger.exception(f"Data summary tab - Error saving details for study {study_name} in {project_folder}")


        processing_complete = True
        logger.info("Data summary tab - All studies have been pre-processed")
        return None, processing_complete 

    # Callback to delay check of if files have being processed
    @callback(
        Output("processed-file-check-interval_dpp", "disabled"),
        Input("processing-complete-store_dpp", "data")
    )
    def toggle_interval(processing_complete):
        # If processing is complete, enable the interval (disabled=False)
        return not processing_complete

    # Callback to display the processed dataset for the study selected in the dropdown
    @callback(
        Output("processed-data-table_dpp", "children"),
        [
        Input("selected-studies-dropdown-summary_dpp", "value"),
        Input("processing-complete-store_dpp", "data")],
        State("project-folder-store_dpp", "data"),  # Added project folder store state
        prevent_initial_call=True
    )
    def display_processed_data_from_file(selected_study, processing_complete, project_folder):
        if not processing_complete:
            return no_update

        # Build the base path using the project folder if available, else error
        if project_folder:
            base = os.path.join(project_folder, "Processed-datasets")
        else:
            logger.error("Data summary tab - project folder chosen doesn't exist")
            return html.Div("Project folder chosen doesn't exist")

        # Determine the final save folder based on the folder choice.
        save_folder = base

        # Load preprocessing steps and determine the flow for the file name.
        steps_map = {}
        if os.path.exists(SELECTED_STUDIES_FILE):
            try:
                with open(SELECTED_STUDIES_FILE) as f:
                    steps_map = json.load(f)
            except Exception:
                logger.exception("Data summary tab - Error reading SELECTED_STUDIES_FILE")

        preprocessing_steps = steps_map.get("studies", {}).get(selected_study, {}).get("preprocessing")
        flows = [os.path.splitext(f)[0] for f in os.listdir("data_preprocessing_flows") if f.endswith(".txt")]
        if isinstance(preprocessing_steps, list) and len(preprocessing_steps) == 1 and preprocessing_steps[0] in flows:
            flow_name_for_file = preprocessing_steps[0]
            # Optionally, assign the actual flow steps if needed, e.g., preprocessing_steps = get_flow_steps(flow_name_for_file)
        else:
            flow_name_for_file = "_".join(preprocessing_steps) if preprocessing_steps else "Untitled"

        # Build the filename and file path.
        filename = f"processed_{selected_study}_{flow_name_for_file}.csv"
        filepath = os.path.join(save_folder, filename)

        """ if not os.path.exists(filepath):
            return html.Div("Processing data, please wait...") """

        try:
            df = pd.read_csv(filepath)
        except Exception:
            logger.exception("Data summary tab - Error reading processed file")
            return html.Div("Error reading processed file")

        #df_head = df.head(100).reset_index().rename(columns={"index": "database_identifier"})
        df_head = df.head(100)
        # now just reflect the DataFrame’s columns *in order*
        columns = [{"name": col, "id": col} for col in df_head.columns]
        fixed_width = "150px"
        processed_table = dash_table.DataTable(
            data=df_head.to_dict("records"),
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
                "padding": "10px",
                "minWidth": fixed_width,
                "width": fixed_width,
                "maxWidth": fixed_width
            },
            markdown_options={"html": True},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"}
            ]
        )
        return processed_table
    
    # Callback that controls the progress bar updating
    @callback(
        Output("start-ts-store",             "data"),
        Output("folder-interval",            "disabled"),
        Output("hide-progress-interval",     "disabled"),
        Output("process-data-progress-bar_dpp","children"),
        Input("process-data-btn_dpp",        "n_clicks"),
        Input("folder-interval",             "n_intervals"),
        Input("hide-progress-interval",      "n_intervals"),
        State("start-ts-store",              "data"),
        State("project-folder-store_dpp",    "data"),
        State("selected-study-store_dpp",    "data"),
        prevent_initial_call=True
    )
    def progress_bar_control(btn, folder_ticks, hide_ticks, start_store, project_folder, selected_studies):
        # figure out which Input fired
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        total = len(selected_studies or [])
        if project_folder:
            base = os.path.join(project_folder, "Processed-datasets")
        else:
            logger.error("Data summary tab - project folder chosen doesn't exist")
            raise PreventUpdate
        #base  = os.path.join(project_folder, "Processed-datasets") if project_folder else "processed-datasets"

        # 1) Button‐click: seed timestamp, enable folder‐interval, disable hide‐interval, show 0/total
        if trigger == "process-data-btn_dpp":
            if total == 0:
                raise PreventUpdate
            ts = time.time()
            bar0 = dbc.Progress(
                value=0,
                label=f"0/{total} studies",
                striped=True, animated=True,
                style={"width":"250px","height":"20px","marginBottom":"1rem"}
            )
            return {"start": ts}, False, True, bar0

        # 2) Folder‐interval tick: count new files → update bar
        if trigger == "folder-interval":
            if not start_store or "start" not in start_store:
                raise PreventUpdate
            cutoff = start_store["start"]

            # count only .csv created after click, matching our studies
            count = 0
            if os.path.isdir(base):
                seen = {
                    fn for fn in os.listdir(base)
                    if fn.endswith(".csv")
                    and os.path.getmtime(os.path.join(base, fn)) > cutoff
                    and any(fn.startswith(f"processed_{s}_") for s in selected_studies)
                }
                count = len(seen)

            pct = int(count/total*100) if total else 0
            bar = dbc.Progress(
                value=pct,
                label=f"{count}/{total} studies",
                striped=True, animated=True,
                style={"width":"250px","height":"20px","marginBottom":"1rem"}
            )

            done = (count >= total)
            # when done: stop polling (disable folder), start hide timer (enable hide)
            return start_store, done, not done, bar

        # 3) Hide‐interval tick: clear bar & disable hide
        if trigger == "hide-progress-interval":
            # simply clear out the progress bar and stop this timer
            return start_store, True, True, ""

        # fallback
        raise PreventUpdate

    # Callback to display the processed data
    @callback(
        Output("processed-data-collapse_dpp", "is_open"),
        Input("process-data-btn_dpp", "n_clicks"),
        prevent_initial_call=True
    )
    def show_processed_data(n_clicks):
        # As soon as the button is clicked once, open the collapse
        return bool(n_clicks)

    # Callback to ensure that the correct studies are displayed in the dropdown
    @callback(
        [Output("selected-studies-dropdown-summary_dpp", "options"),
        Output("selected-studies-dropdown-summary_dpp", "value")],
        Input("selected-study-store_dpp", "data")
    )
    def update_summary_dropdown(selected_studies):
        if selected_studies:
            options = [{"label": study, "value": study} for study in selected_studies]
            return options, options[0]["value"]
        return [], None
    
    # Callback to populate the sidebars with the options chosen for the study in the dropdown
    @callback(
        [
            # study‐details sidebar
            Output("summary-side-outliers_dpp",      "value"),
            Output("summary-side-control-group_dpp", "options"),
            Output("summary-side-control-group_dpp", "value"),
            Output("summary-side-case-group_dpp",    "options"),
            Output("summary-side-case-group_dpp",    "value"),
            # preprocessing summary
            Output("summary-missing-values-checklist_dpp", "value"),
            Output("summary-transformation-checklist_dpp", "value"),
            Output("summary-standardisation-checklist_dpp","value"),
        ],
        [
            Input("data_pre_process_tabs",                 "active_tab"),
            Input("selected-studies-dropdown-summary_dpp", "value"),
        ],
        prevent_initial_call=True
    )
    def populate_summary_sidebars(active_tab, selected_study):
        if active_tab != "summary" or not selected_study:
            raise PreventUpdate

        # 1) load JSON
        try:
            with open(SELECTED_STUDIES_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            logger.exception("Data summary tab - Error reading SELECTED_STUDIES_FILE")
            raise PreventUpdate

        study = payload.get("studies", {}).get(selected_study, {})

        # ─── study details ─────────────────────────────
        outliers = study.get("outliers") or []

        group_filter = study.get("group_filter", {})
        control = group_filter.get("Control", []) or []
        case    = group_filter.get("Case",   []) or []

        control_options = [{"label": g, "value": g} for g in control]
        case_options    = [{"label": g, "value": g} for g in case]

        # ─── preprocessing summary ─────────────────────
        saved = study.get("preprocessing") or []

        # detect single‐flow case
        flows = [
            os.path.splitext(f)[0]
            for f in os.listdir("data_preprocessing_flows")
            if f.endswith(".txt")
        ]
        if isinstance(saved, list) and len(saved) == 1 and saved[0] in flows:
            steps = get_flow_steps(saved[0])
        else:
            steps = saved

        missing_vals   = [s for s in steps if s in ["knn_imputer", "mean_imputer", "iterative_imputer"]]
        transformation = [s for s in steps if s in ["log_transform", "cube_root"]]
        standardisation= [s for s in steps if s in ["standard_scaler", "min_max_scaler", "robust_scaler", "max_abs_scaler"]]

        return (
            # study details
            outliers,
            control_options, control,
            case_options,    case,
            # preprocessing
            missing_vals,
            transformation,
            standardisation,
        )
    
    
