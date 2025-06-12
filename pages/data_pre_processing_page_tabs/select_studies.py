# pages/data_pre_processing_page_tabs/select_studies.py
from dash import html, dcc, callback, Input, Output, State, callback_context, dash_table, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import os
import base64


UPLOAD_FOLDER = "pre-processed-datasets"
DATASET_SOURCE_LABELS = {
    "metabolights": "MetaboLights",
    "metabolomics_workbench": "Metabolomics Workbench",
    "original_refmet_ids": "Original data - Refmet ids",
    "original_Chebi_ids": "Original data - Chebi ids"
}

def get_study_names_dpp():
    if os.path.exists(UPLOAD_FOLDER):
        return [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isdir(os.path.join(UPLOAD_FOLDER, f))]
    return []

layout = [
            html.Div([
                html.H3("Available Studies", style={"display": "inline-block", "margin": "0", "textAlign": "left"}),
                html.Div([
                    dbc.Button("Confirm study selection", id="confirm-study-selection-btn_dpp", color="primary", size="sm", disabled=True, style={"marginRight": "10px"}),
                    dbc.Button("Upload a new study", id="open-upload-study-btn_dpp", color="primary", size="sm")
                ], style={"float": "right"})
            ], style={"width": "100%", "position": "relative", "marginBottom": "1rem"}),
            # DataTable remains unchanged
            html.Div(id="studies-table-container_dpp", style={"marginRight": "50px"}),
        ]

def register_callbacks():
    # Callback to populate the studies table with folder names from UPLOAD_FOLDER
    @callback(
        Output("studies-table-container_dpp", "children"),
        [Input("data-summary-url_dpp", "href"),
        Input("upload-study-modal_dpp", "is_open")]
    )
    def update_studies_table_dpp(href, modal_is_open):
        # Only refresh table when modal is closed.
        if modal_is_open:
            raise PreventUpdate

        study_names = get_study_names_dpp()
        if not study_names:
            return html.Div("No studies found.", style={"textAlign": "center"})
        
        records = [{"Study Name": name} for name in study_names]
        table = dash_table.DataTable(
            id="studies-table_dpp",  # Set the ID so we can access selected_rows
            data=records,
            columns=[{"name": "Study Name", "id": "Study Name"}],
            row_selectable="multi",
            selected_rows=[],  # initialize with no rows selected
            page_size=10,
            style_table={"overflowX": "auto", "marginRight": "50px"},
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
            }
        )
        return table

    @callback(
        Output("confirm-study-selection-btn_dpp", "disabled"),
        Input("studies-table_dpp", "selected_rows")
    )
    def toggle_confirm_button(selected_rows):
        # Enable button only if at least one row is selected.
        if selected_rows and len(selected_rows) > 0:
            return False
        return True

    @callback(
        Output("exploration-tab_dpp", "disabled"),
        Input("study-confirmed-store_dpp", "data")
    )
    def toggle_exploration_tab(confirmed):
        # Enable the tab if confirmed is True, otherwise disable it.
        if confirmed:
            return False
        return True

    # Callback to open/close the "Upload a new study" modal
    @callback(
        Output("upload-study-modal_dpp", "is_open"),
        [Input("open-upload-study-btn_dpp", "n_clicks"),
        Input("close-upload-study-btn_dpp", "n_clicks")],
        [State("upload-study-modal_dpp", "is_open")],
    )
    def toggle_upload_study_modal_dpp(open_click, close_click, is_open):
        ctx = callback_context
        if ctx.triggered:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if button_id == "open-upload-study-btn_dpp":
                return True
            elif button_id == "close-upload-study-btn_dpp":
                return False
        return is_open
    
    @callback(
        Output("upload-status_dpp", "children"),
        Output("uploaded-file-store_dpp", "data"),
        Output("selected-files-checklist_dpp", "options"),
        Output("selected-files-checklist_dpp", "value"),
        Output("overwrite-warning-modal_dpp", "is_open"),
        Output("overwrite-study-name_dpp", "children"),

        Input("upload-data_dpp",            "contents"),
        Input("upload-study-btn_dpp",       "n_clicks"),
        Input("confirm-overwrite-btn_dpp",  "n_clicks"),
        Input("cancel-overwrite-btn_dpp",   "n_clicks"),

        State("upload-data_dpp",            "filename"),
        State("uploaded-file-store_dpp",    "data"),
        State("selected-files-checklist_dpp","value"),
        State("study-name_dpp",             "value"),
        State("dataset-source_dpp",         "value"),
        prevent_initial_call=True,
    )
    def handle_upload_all(contents,
                        upload_clicks,
                        confirm_clicks,
                        cancel_clicks,
                        filenames,
                        store,           # now a Dict[str,str]
                        checked_names,
                        study_name,
                        dataset_source):

        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        def do_write(files_to_write):
            folder = os.path.join(UPLOAD_FOLDER, study_name)
            os.makedirs(folder, exist_ok=True)
            saved = []
            for f in files_to_write:
                name = f["name"]
                _, b64 = f["content"].split(",", 1)
                data = base64.b64decode(b64)
                with open(os.path.join(folder, name), "wb") as fd:
                    fd.write(data)
                saved.append(name)
            # rewrite details file
            with open(os.path.join(folder, "study_details.txt"), "w") as fd:
                fd.write(f"Study Name: {study_name}\n")
                fd.write(f"Dataset Source: {DATASET_SOURCE_LABELS.get(dataset_source, dataset_source)}\n")
            return saved

        # 1) File‐picker: stage & checklist update
        if trigger == "upload-data_dpp":
            if not contents:
                raise PreventUpdate

            # Merge new files into our map
            staged_map = store or {}
            prev_checked = set(checked_names or [])

            for name, content in zip(filenames, contents):
                if name not in staged_map:
                    staged_map[name] = content

            options = [{"label": n, "value": n} for n in staged_map.keys()]
            # carry forward any previously checked + newly added
            value = list(prev_checked.union(staged_map.keys()))

            return (
                no_update,     # upload-status unchanged
                staged_map,    # new store: Dict[name->content]
                options,
                value,
                False,         # overwrite-warning closed
                no_update
            )

        # 2) Upload button clicked: either write or show warning
        if trigger == "upload-study-btn_dpp":
            staged_map = store or {}
            checked = set(checked_names or [])

            # which names to upload
            to_upload_names = [n for n in staged_map if n in checked]
            if not to_upload_names:
                return (
                    "⚠️ No files ticked for upload.",
                    no_update, no_update, no_update,
                    False, no_update
                )

            folder = os.path.join(UPLOAD_FOLDER, study_name)
            if os.path.exists(folder):
                # prompt overwrite
                return (
                    no_update, no_update, no_update, no_update,
                    True, study_name
                )
            else:
                # build full list of dicts
                to_upload = [
                    {"name": n, "content": staged_map[n]}
                    for n in to_upload_names
                ]
                saved = do_write(to_upload)
                status = f"✅ Uploaded: {', '.join(saved)}"
                opts   = [{"label": n, "value": n} for n in saved]

                # remove written files from the staged map
                for n in saved:
                    staged_map.pop(n, None)

                return (
                    status,
                    staged_map,   # keep any remaining staged files
                    opts,
                    saved,        # mark these as checked
                    False,        # close overwrite-warning
                    no_update
                )

        # 3) Confirm‐overwrite clicked: force write into existing
        if trigger == "confirm-overwrite-btn_dpp":
            staged_map = store or {}
            checked = set(checked_names or [])

            to_upload_names = [n for n in staged_map if n in checked]
            if not to_upload_names:
                return (
                    "⚠️ No files ticked for upload.",
                    no_update, no_update, no_update,
                    False, no_update
                )

            to_upload = [
                {"name": n, "content": staged_map[n]}
                for n in to_upload_names
            ]
            saved = do_write(to_upload)
            status = f"✅ Appended to '{study_name}': {', '.join(saved)}"

            # clear them out
            for n in saved:
                staged_map.pop(n, None)

            options = [{"label": n, "value": n} for n in staged_map.keys()]
            return (
                status,
                staged_map,
                options,
                [],      # nothing checked after overwrite
                False,
                no_update
            )

        # 4) Cancel‐overwrite clicked: just close the warning
        if trigger == "cancel-overwrite-btn_dpp":
            return (
                no_update, no_update, no_update, no_update,
                False, no_update
            )

        # fallback
        return no_update, no_update, no_update, no_update, False, no_update

    
    @callback(
        [Output("dropdown-existing-projects", "options"),
        Output("input-analysis-project", "value"),
        Output("dropdown-existing-projects", "value")],
        [Input("modal-analysis-project", "is_open"),
        Input("dropdown-existing-projects", "value"),
        Input("input-analysis-project", "value")]
    )
    def manage_project_inputs(is_open, dropdown_value, input_value):
        ctx = callback_context
        trigger_id = ctx.triggered_id  # Get which input triggered the callback

        # Default outputs: no change
        dropdown_options = no_update
        input_val = no_update
        dropdown_val = no_update

        # Populate dropdown when modal opens
        if trigger_id == "modal-analysis-project" and is_open:
            if os.path.exists("Projects"):
                folders = os.listdir("Projects")
                dropdown_options = [
                    {"label": folder.replace("-", " "), "value": folder}
                    for folder in folders if os.path.isdir(os.path.join("Projects", folder))
                ]
            else:
                dropdown_options = []

        # If user selected a dropdown item, clear the input
        elif trigger_id == "dropdown-existing-projects" and dropdown_value:
            input_val = ""

        # If user typed in the input, clear the dropdown selection
        elif trigger_id == "input-analysis-project" and input_value:
            dropdown_val = None

        return dropdown_options, input_val, dropdown_val
    
    @callback(
        [Output("modal-analysis-project", "is_open"),
        Output("project-name-display", "children"),
        Output("dummy-output", "children"),
        Output("project-folder-store_dpp", "data")],
        Input("confirm-analysis-project-btn", "n_clicks"),
        State("input-analysis-project", "value"),
        State("dropdown-existing-projects", "value"),
        State("modal-analysis-project", "is_open")
    )
    def update_project_name_and_create_folders(n_clicks, new_project_name, selected_project_folder, is_open):
        if n_clicks:
            # Determine the project name and folder
            if selected_project_folder:
                sanitized_name = selected_project_folder  # Already in hyphen format
                display_text = sanitized_name.replace("-", " ")
            elif new_project_name:
                display_text = new_project_name
                sanitized_name = new_project_name.replace(" ", "-")
            else:
                return is_open, "Project Name Not Provided", "", ""

            projects_dir = "Projects"
            project_folder_path = os.path.join(projects_dir, sanitized_name)

            # Ensure the Projects folder exists
            os.makedirs(projects_dir, exist_ok=True)

            # Create main project folder
            os.makedirs(project_folder_path, exist_ok=True)

            # Create subfolders
            processed_datasets_path = os.path.join(project_folder_path, "Processed-datasets")
            plots_path = os.path.join(project_folder_path, "Plots")
            os.makedirs(processed_datasets_path, exist_ok=True)
            os.makedirs(plots_path, exist_ok=True)

            # Subfolders under Plots
            single_study_path = os.path.join(plots_path, "Single-study-analysis")
            preprocessing_analysis_path = os.path.join(plots_path, "Preprocessing-analysis")
            multi_study_path = os.path.join(plots_path, "Multi-study-analysis")
            os.makedirs(single_study_path, exist_ok=True)
            os.makedirs(preprocessing_analysis_path, exist_ok=True)
            os.makedirs(multi_study_path, exist_ok=True)

            # Single-study-analysis subfolders
            os.makedirs(os.path.join(single_study_path, "Differential-metabolites-box-plots"), exist_ok=True)
            os.makedirs(os.path.join(single_study_path, "Differential-metabolites-table-plots"), exist_ok=True)
            os.makedirs(os.path.join(single_study_path, "Differential-pathway-box-plots"), exist_ok=True)
            os.makedirs(os.path.join(single_study_path, "Differential-pathway-table-plots"), exist_ok=True)

            # Preprocessing-analysis subfolders
            os.makedirs(os.path.join(preprocessing_analysis_path, "PCA-plots"), exist_ok=True)
            os.makedirs(os.path.join(preprocessing_analysis_path, "Box-plots"), exist_ok=True)
            os.makedirs(os.path.join(preprocessing_analysis_path, "Residual-plots"), exist_ok=True)

            # Multi-study-analysis subfolders
            os.makedirs(os.path.join(multi_study_path, "Co-occurring-metabolites-upset-plots"), exist_ok=True)
            os.makedirs(os.path.join(multi_study_path, "Differential-co-occurring-metabolites-upset-plots"), exist_ok=True)
            os.makedirs(os.path.join(multi_study_path, "Differential-metabolites-network-plots"), exist_ok=True)
            os.makedirs(os.path.join(multi_study_path, "Differential-pathway-network-plots"), exist_ok=True)

            # Optional debug message
            folder_structure_message = (
                f"Created/verified folder: {project_folder_path}\n"
                f"In 'Plots': {single_study_path}, {preprocessing_analysis_path}, {multi_study_path}"
            )

            return False, display_text, folder_structure_message, project_folder_path

        return is_open, "", "", ""
