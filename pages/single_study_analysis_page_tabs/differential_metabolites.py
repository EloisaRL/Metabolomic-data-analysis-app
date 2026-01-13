# pages/single_study_analysis_page_tabs/differential_metabolites.py
from .shared_functions.data_processing import da_testing
from dash import html, dcc, callback, Input, Output, dash_table, State, no_update
import dash_bootstrap_components as dbc
import os, re, gzip, csv, requests
import pandas as pd
import plotly.express as px
import plotly.io as pio
import base64
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

import requests

import sqlite3

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = CACHE_DIR / "chebi_name_map.db"

# ========================================== #
# Layout of the Differential metabolites tab #
# ========================================== #

layout = html.Div([
                    # Title
                    html.H2("Differential Metabolite Analysis"),

                    # Background processing description
                    html.Div(
                        [
                            html.H4("Background processing", style={"marginBottom": "0.75rem"}),

                            html.P(
                                "Differential testing separates metabolites into Case and Control groups and performs "
                                "an independent two-sided t-test for each metabolite. Metabolites are labelled as "
                                "“Up” or “Down” based on the sign of the test statistic, followed by Benjamini–Hochberg "
                                "FDR correction of p-values. Metabolites with an adjusted p-value below 0.05 are "
                                "reported as differentially abundant.",
                                style={"marginBottom": "0.75rem"}
                            ),

                            html.Div(
                                [
                                    html.B("Important note"),
                                    html.P(
                                        [
                                            "For studies using ChEBI identifiers, metabolite names are retrieved via the "
                                            "ChEBI 2.0 API. This step takes approximately ",
                                            html.B("15–20 seconds"),
                                            " to convert 10 metabolites. "
                                            "Once completed, names are cached locally so future analyses run faster."
                                        ],
                                        style={"marginTop": "0.5rem"}
                                    ),
                                    html.P(
                                        "Please keep the app open until the results are displayed. Closing or refreshing "
                                        "the app will interrupt name retrieval and prevent caching. "
                                        "Advanced users can consult the app log file to see how many metabolites are being processed.",
                                        style={"marginBottom": 0}
                                    ),
                                ],
                                style={
                                    "backgroundColor": "#ffffff",
                                    "borderLeft": "4px solid #0074D9",
                                    "padding": "0.75rem 1rem",
                                    "borderRadius": "3px",
                                },
                            ),
                        ],
                        style={
                            "backgroundColor": "#f5f5f5",
                            "padding": "1.25rem",
                            "borderRadius": "6px",
                            "marginBottom": "1.5rem",
                        },
                    ),

                    # always‐visible “num metabolites” input
                    html.Div(
                        [
                            dbc.Label("Number of metabolites to plot:"),
                            dcc.Input(
                                id="num-top-metabolites",
                                type="number", min=1, max=50, step=1, value=10,
                                style={"width": "100px", "marginBottom": "1rem"}
                            ),
                        ],
                        id="num-met-wrapper",
                        style={"display": "block", "textAlign": "center"},
                    ),

                    # always‐visible Save button for the **chart**
                    html.Div(
                        dbc.Button(
                            "Save chart",
                            id="open-save-modal-chart",
                            n_clicks=0,
                            style={"backgroundColor": "white", "color": "black"},
                        ),
                        id="save-chart-wrapper",
                        className="d-flex justify-content-end mb-2",
                        style={"display": "flex"},
                    ),

                    # chart placeholder (blank until callback fills it)
                    dcc.Loading(
                        id="loading-differential-chart",
                        children=html.Div(
                            id="differential-chart-content",
                            style={
                                "display":        "flex",
                                "justifyContent": "center",
                                "width":          "100%",
                                "minHeight":      "300px",
                            }
                        )
                    ),

                    # Store, Modal, feedback for **chart**
                    dcc.Store(id="diff-chart-store"),
                    dbc.Modal(
                        [
                            dbc.ModalHeader("Name your chart file"),
                            dbc.ModalBody(
                                dcc.Input(
                                    id="plot-name-input-chart",
                                    type="text",
                                    placeholder="Enter filename…",
                                    style={"width": "100%"},
                                )
                            ),
                            dbc.ModalFooter(
                                dbc.Button(
                                    "Save",
                                    id="confirm-save-plot-chart",
                                    color="primary",
                                    className="ms-auto",
                                    n_clicks=0,
                                )
                            ),
                        ],
                        id="save-plot-modal-chart",
                        is_open=False,
                        size="sm",
                    ),


                    # always‐visible Save button for the **table**
                    html.Div(
                        dbc.Button(
                            "Save table",
                            id="open-save-modal-table",
                            n_clicks=0,
                            style={"backgroundColor": "white", "color": "black"},
                        ),
                        id="save-table-wrapper",
                        className="d-flex justify-content-end mb-2",
                        style={"display": "flex"},
                    ),

                    # table placeholder (blank until callback fills it)
                    dcc.Loading(
                        id="loading-differential-table",
                        children=html.Div(
                            id="differential-table-content",
                            style={
                                "display":        "flex",
                                "justifyContent": "center",
                                "width":          "100%",
                                "minHeight":      "300px",
                            }
                        )
                    ),

                    # Store, Modal, feedback for **table**
                    dcc.Store(id="diff-table-store"),
                    dbc.Modal(
                        [
                            dbc.ModalHeader("Name your table file"),
                            dbc.ModalBody(
                                dcc.Input(
                                    id="plot-name-input-table",
                                    type="text",
                                    placeholder="Enter filename…",
                                    style={"width": "100%"},
                                )
                            ),
                            dbc.ModalFooter(
                                dbc.Button(
                                    "Save",
                                    id="confirm-save-plot-table",
                                    color="primary",
                                    className="ms-auto",
                                    n_clicks=0,
                                )
                            ),
                        ],
                        id="save-plot-modal-table",
                        is_open=False,
                        size="sm",
                    ),

                ], style={"padding": "1rem"})

# Register this tab’s callbacks
def register_callbacks():
    # Callback performs the differential testing on the selected study and produces the box plot and the table
    @callback(
        # 4 outputs: chart‐DIV, chart‐store, table‐DIV, table‐store
        Output("differential-chart-content", "children"),
        Output("diff-chart-store",        "data"),
        Output("differential-table-content","children"),
        Output("diff-table-store",        "data"),
        [
            Input("project-dropdown-pop",       "value"),
            Input("selected-file-radio-ssa",     "value"),
            Input("num-top-metabolites",        "value"),
        ]
    )
    def update_differential_analysis(selected_project, selected_file, top_n):
        # if nothing selected, show a warning in the chart area and clear everything else
        if not selected_project or not selected_file:
            return html.Div("Please select a project and a file for differential metabolite analysis."), None, None, None

        # build path
        filepath = os.path.join(
            "Projects", selected_project, "processed-datasets", selected_file
        )
        if not os.path.exists(filepath):
            error = dbc.Alert(f"Processed file not found: {filepath}", color="danger")
            return error, None, None, None

        # load and index
        df = pd.read_csv(filepath).set_index("database_identifier")

        # run your da_testing as before...
        class DA: pass
        da = DA()
        da.processed_data = df
        da.pathway_data   = None
        da.md_filter      = None
        da.node_name      = f"{selected_project}/{selected_file}"
        da.pathway_level  = False
        da.da_testing     = da_testing.__get__(da, DA)

        try:
            da.da_testing()
        except Exception:
            logger.exception(f"Differential metabolite tab - Error running differential analysis for: {selected_file}")
            err = dbc.Alert("Error running differential analysis", color="danger")
            return err, None, None, None

        sig = da.pval_df[da.pval_df["FDR_P-value"] < 0.05]
        if sig.empty:
            empty = html.Div("No significant metabolites (FDR < 0.05) found.")
            return empty, None, None, None

        # prepare sorted table
        sig_sorted = sig.sort_values("FDR_P-value").copy()
        sig_sorted["P-value"]     = sig_sorted["P-value"].apply(lambda x: f"{x:.3e}")
        sig_sorted["FDR_P-value"] = sig_sorted["FDR_P-value"].apply(lambda x: f"{x:.3e}")
        sig_sorted["Stat"]        = sig_sorted["Stat"].round(3)

        # Filtering the abundance table to only contain the diff metabolites
        # Columns that should always be retained
        always_keep = ['Group', 'group_type']

        # Filter df to include differential metabolites + always_keep columns
        keep_cols = list(df.columns.intersection(sig_sorted.index)) + [col for col in always_keep if col in df.columns]

        # Reorder columns so that 'Group' and 'group_type' stay at the end
        df_diff = df[keep_cols].copy()


        # ChEBI 2.0 - based name lookup   
        _CHEBI_PATTERN = re.compile(r"^(CHEBI:\d+|chebi:\d+|\d+)$")

        def is_valid_chebi_id(value: str) -> bool:
            return isinstance(value, str) and bool(_CHEBI_PATTERN.match(value))
        def normalise_chebi_id(value: str) -> str:
            value = value.strip()
            if value.isdigit():
                return f"CHEBI:{value}"
            return value.upper()
             
        def _safe_chebi_name(value, cursor):
            # Skip non-ChEBI columns
            if not is_valid_chebi_id(value):
                return value

            chebi_id = normalise_chebi_id(value)

            # 1️⃣ SQLite lookup
            cursor.execute(
                "SELECT name FROM chebi_name_map WHERE chebi_id = ?",
                (chebi_id,)
            )
            row = cursor.fetchone()

            if row and row[0] != chebi_id:
                return row[0]

            # 2️⃣ API fallback (UNCHANGED LOGIC)
            response = requests.get(
                "https://www.ebi.ac.uk/chebi/backend/api/public/compounds/",
                params={"chebi_ids": chebi_id},
                timeout=10
            )

            try:
                data = response.json()
            except ValueError:
                return value

            # CASE 1: dict keyed by CHEBI ID
            if isinstance(data, dict):
                entry = data.get(chebi_id) or data.get(value)
                if entry and entry.get("exists"):
                    name = entry["data"].get("name", value)
                    cursor.execute(
                        "INSERT OR REPLACE INTO chebi_name_map VALUES (?, ?)",
                        (chebi_id, name)
                    )
                    return name

            # CASE 2: list response
            if isinstance(data, list) and data and isinstance(data[0], dict):
                name = data[0].get("name", value)
                cursor.execute(
                    "INSERT OR REPLACE INTO chebi_name_map VALUES (?, ?)",
                    (chebi_id, name)
                )
                return name

            return value


        num_of_cols = len(df_diff.columns)
        logger.info(f"Upset plots tab - Starting ChEBI id to metabolite name conversion for {selected_file} which has {num_of_cols} ids.")
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            # Apply to columns
            df_diff.columns = [
                _safe_chebi_name(c, cursor)
                for c in df_diff.columns
            ]

            # Apply to index
            sig_sorted.index = [
                _safe_chebi_name(i, cursor)
                for i in sig_sorted.index
            ]

            conn.commit()  # commit ONCE


        #df_diff.columns = df_diff.columns.map(_safe_chebi_name)
        #sig_sorted = sig_sorted.rename(index=_safe_chebi_name)


        # Save the KPCA scores.
        results_folder = os.path.join("Projects", selected_project, "raw_results_data", "differential metabolites")
        os.makedirs(results_folder, exist_ok=True)
        base_filename = selected_file.replace('.csv', '')
        save_filename = f"Diff_Metabolite_results{base_filename}.csv"
        save_filepath = os.path.join(results_folder, save_filename)
        df_diff.to_csv(save_filepath)

        metabolite_table = dash_table.DataTable(
                data=sig_sorted.reset_index().to_dict('records'),
                columns=[{"name": c, "id": c} for c in sig_sorted.reset_index().columns],
                sort_action="native",
                page_size=10,
                style_table={"overflowX": "auto", "marginRight": "50px", "border": "1px solid #ccc",
                            "borderRadius": "5px", "boxShadow": "2px 2px 5px rgba(0, 0, 0, 0.1)"},
                style_header={"backgroundColor": "#f2f2f2", "fontFamily": "Arial", "fontSize": "16px",
                            "fontWeight": "bold", "textAlign": "left", "border": "1px solid #ddd",
                            "padding": "10px"},
                style_cell={"fontFamily": "Arial", "fontSize": "14px", "textOverflow": "ellipsis",
                            "whiteSpace": "nowrap", "overflow": "hidden", "textAlign": "left",
                            "border": "1px solid #ddd", "padding": "10px"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"}],
                style_as_list_view=True
            )

        # pick top N for the box plot
        top_mets    = list(sig_sorted.index)[: (top_n or 10)]
        #ordered_mets = top_mets
        ordered_mets = list(sig_sorted.loc[top_mets].index)
        title       = f"Box Plot of Top {len(top_mets)} Differentially Abundant Metabolites"

        box_df = df_diff[top_mets + ["group_type"]].reset_index(drop=True)
        box_long = pd.melt(
            box_df,
            id_vars=["group_type"],
            value_vars=top_mets,
            var_name="Metabolite",
            value_name="Value"
        )

        # Extract the study_name from selected_file:
        basename = os.path.basename(selected_file)           # "processed_MTBLS1866_knn_imputer_log_transform.csv"
        no_ext   = os.path.splitext(basename)[0]             # "processed_MTBLS1866_knn_imputer_log_transform"
        if no_ext.startswith("processed_"):
            remainder  = no_ext[len("processed_"):]           # "MTBLS1866_knn_imputer_log_transform"
            study_name = remainder.split("_")[0]              # "MTBLS1866"
        else:
            study_name = None

        project_details_path = os.path.join("Projects", selected_project, "project_details_file.json")

        with open(project_details_path, "r", encoding="utf-8") as f:
            payload = json.load(f).get("studies", {})

        group_filter = payload[study_name].get("group_filter", {})
        group_labels = {
            gt: ", ".join(labels)
            for gt, labels in group_filter.items()
        }

        box_long["Group_Label"] = box_long["group_type"].map(group_labels)

        fig_box = px.box(
            box_long,
            x="Metabolite",
            y="Value",
            color="Group_Label",
            title=title,
            labels={"Value":"Metabolite Intensity"},
            category_orders={"Metabolite":ordered_mets}
        )

        NEW_H = 400
        orig_w = fig_box.layout.width  or 700
        orig_h = fig_box.layout.height or 450
        aspect = orig_w / orig_h
        BASE_W = int(aspect * NEW_H) + 200

        # width driven by requested count (but never more than actual)
        requested = top_n or 10
        top_mets = sig_sorted.index.tolist()[:requested]
        actual    = len(top_mets)
        n_for_width = requested if requested <= actual else actual
        BAR_PX      = 50
        bar_needed  = n_for_width * BAR_PX
        NEW_W       = max(BASE_W, bar_needed)

        #  (1) Compute the longest label length in characters
        max_label_len = max(len(str(lbl)) for lbl in ordered_mets)
        #  (2) Turn that into an estimated pixel height needed for rotated labels
        PX_PER_CHAR         = 5   # px of vertical space per character (45°‐rotated)
        estimated_label_px  = max_label_len * PX_PER_CHAR
        NEW_H = 220 + estimated_label_px
        NEW_W = max(BASE_W, bar_needed)
        fig_box.update_layout(
            width = NEW_W,
            height = NEW_H,
            margin = dict(
                l = 40,
                r = 40,
                t = 40,
                b = 40
            ),
            title = {
                "text": title,
                "x": 0.5,
                "xanchor": "center"
            }
        )

        # serialize for the two stores
        fig_json  = pio.to_json(fig_box)
        csv_bytes = sig_sorted.reset_index().to_csv(index=False).encode()
        table_b64 = base64.b64encode(csv_bytes).decode()

        # the 2 “children” outputs go straight into your placeholders;
        # the Loading spinners live in the layout around them
        chart_child = html.Div(
            dcc.Graph(
                figure=fig_box,
                style={
                    "width":  f"{NEW_W}px",
                    "height": f"{NEW_H}px"
                }
            ),
            style={
                "display": "flex",           # make it a flex container
                "justifyContent": "center",  # center children horizontally
                "padding": "0 1rem",
                "boxSizing": "border-box"
            }
        )


        table_child = html.Div(
            metabolite_table,
            style={"width":"100%","padding":"0 1rem","boxSizing":"border-box"}
        )

        # ✅ Log success
        logger.info(f"Differential metabolite tab - Successfully produced differential metabolites summary chart and table for: {selected_file}")

        return (
            chart_child,
            {"type":"plotly","data":fig_json},
            table_child,
            {"type":"csv","data":table_b64},
        )
    
    # Callback controls the save box plot pop up from opening
    @callback(
        Output("save-plot-modal-chart","is_open"),
        [ Input("open-save-modal-chart","n_clicks"),
        Input("confirm-save-plot-chart","n_clicks") ],
        [ State("save-plot-modal-chart","is_open") ]
    )
    def toggle_chart_modal(open_n, save_n, is_open):
        if open_n or save_n:
            return not is_open
        return is_open

    # Callback controls the save table pop up from opening
    @callback(
        Output("save-plot-modal-table","is_open"),
        [ Input("open-save-modal-table","n_clicks"),
        Input("confirm-save-plot-table","n_clicks") ],
        [ State("save-plot-modal-table","is_open") ]
    )
    def toggle_table_modal(open_n, save_n, is_open):
        if open_n or save_n:
            return not is_open
        return is_open


    # Callback which saves **chart** as SVG
    @callback(
        Input("confirm-save-plot-chart","n_clicks"),
        [
        State("plot-name-input-chart","value"),
        State("diff-chart-store","data"),
        State("project-dropdown-pop","value"),
        ]
    )
    def save_chart(n_clicks, filename, payload, project):
       # no n_clicks yet → do nothing
        if not n_clicks:
            return
        # validation errors printed to console
        if not project:
            logger.error("Differential metabolite tab - No project selected for chart")
            return
        if not filename:
            logger.error("Differential metabolite tab - No filename provided for chart")
            return
        if not payload:
            logger.error("Differential metabolite tab - No chart data to save")
            return

        # Rebuild the figure from JSON
        fig = pio.from_json(payload["data"])

        # Grab the on-screen dimensions (assumes you set them on the figure)
        width  = fig.layout.width  or 700   # fallback if unset
        height = fig.layout.height or 400   # fallback if unset

        # Build your output directory
        out_dir = os.path.join(
            "Projects",
            project,
            "Plots",
            "Single-study-analysis",
            "Differential-metabolites-box-plots"
        )
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{filename}.svg")

        # Write using the exact same pixels
        pio.write_image(
            fig,
            out_path,
            format="svg",
            width=int(width),
            height=int(height),
        )

        # ✅ Log success
        logger.info(f"Differential metabolite tab - Successfully saved chart: {out_path}")

    # Callback which saves **table** as CSV
    @callback(
        Input("confirm-save-plot-table","n_clicks"),
        [
        State("plot-name-input-table","value"),
        State("diff-table-store","data"),
        State("project-dropdown-pop","value"),
        ]
    )
    def save_table(n_clicks, filename, payload, project):
        # no n_clicks yet → do nothing
        if not n_clicks:
            return
        # validation errors printed to console
        if not project:
            logger.error("Differential metabolite tab - No project selected for table")
            return
        if not filename:
            logger.error("Differential metabolite tab - No filename provided for table")
            return
        if not payload:
            logger.error("Differential metabolite tab - No table data to save")
            return

        out_dir = os.path.join(
            "Projects",
            project,
            "Plots",
            "Single-study-analysis",
            "Differential-metabolites-table-plots"
        )
        os.makedirs(out_dir, exist_ok=True)

        csv_data = base64.b64decode(payload["data"])
        out_path = os.path.join(out_dir, f"{filename}.csv")
        with open(out_path, "wb") as f:
            f.write(csv_data)

        # ✅ Log success
        logger.info(f"Differential metabolite tab - Successfully saved table: {out_path}")


