# pages/single_study_analysis_page_tabs/differential_pathways.py
from dash import html, dcc, callback, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
import os
import sspa
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import plotly.express as px
import plotly.io as pio
import base64
import json
import logging
logger = logging.getLogger(__name__)

refmet = pd.read_csv("refmet.csv", dtype=object)
refmet.columns = refmet.columns.str.strip() 
name2pubchem = dict(zip(refmet['refmet_name'], refmet['chebi_id']))

def read_study_details_dpp2(folder):
    """Reads study details for a given study, contains info of the study name and dataset source"""
    details_path = os.path.join(folder, "study_details.txt")
    details = {}
    if os.path.exists(details_path):
        with open(details_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key, value = parts
                        details[key.strip()] = value.strip()
    return details

###### NEED TO ADD THE USE OF THIS FUNCTION ########
def da_testing(self):
        '''
        Performs differential analysis testing, adds pval_df attribute containing results.
        '''
        if self.pathway_level == True:
            dat = self.pathway_data
        else:
            dat = self.processed_data
        print('starting da test')

        # Directly filter using the 'group_type' column
        X_case = dat.loc[dat['group_type'] == 'Case'].select_dtypes(include='number')
        X_ctrl = dat.loc[dat['group_type'] == 'Control'].select_dtypes(include='number')

        
        # Convert to DataFrame if filtering returned a Series
        if isinstance(X_case, pd.Series):
            X_case = X_case.to_frame().T
        if isinstance(X_ctrl, pd.Series):
            X_ctrl = X_ctrl.to_frame().T
        
        # Restrict to common numeric columns
        common_cols = X_case.columns.intersection(X_ctrl.columns)
        X_case = X_case[common_cols]
        X_ctrl = X_ctrl[common_cols]

        stat, pvals = stats.ttest_ind(X_case, X_ctrl,
                                    alternative='two-sided',
                                    nan_policy='raise')
        pval_df = pd.DataFrame({
            'P-value': pvals,
            'Stat': stat,
            'Direction': ['Up' if s > 0 else 'Down' for s in stat]
        }, index=X_case.columns)
        
        pval_df['Stat'] = stat
        pval_df['Direction'] = ['Up' if x > 0 else 'Down' for x in stat]
        self.pval_df = pval_df

        # fdr correction 
        pval_df['FDR_P-value'] = multipletests(pvals, method='fdr_bh')[1]

        # return significant metabolites
        self.DA_metabolites = pval_df[pval_df['FDR_P-value'] < 0.05].index.tolist()
        print(f"Number of differentially abundant metabolites: {len(self.DA_metabolites)}") 

        # generate tuples for nx links
        self.connection = [(self.node_name, met) for met in self.DA_metabolites]
        self.full_connection = [(self.node_name, met) for met in self.processed_data.columns[:-1]]

# ======================================= #
# Layout of the Differential pathways tab #
# ======================================= #

layout = html.Div([

                        # Title
                        html.H2("Differential Pathway Analysis"),

                        # Background processing in a grey box
                        html.Div(
                            [
                                html.H4("Background processing description", style={"marginBottom": "0.5rem"}),

                                html.P(
                                    "If the dataset uses RefMet IDs (i.e. originates from workbench or is original data), "
                                    "RefMet-to–ChEBI conversion is performed renaming each metabolite column to its "
                                    "corresponding ChEBI ID (dropping any unmapped columns).",
                                    style={"marginBottom": "0.5rem"}
                                ),

                                html.P(
                                    "Next metabolites are mapped to Reactome pathways (using file version 90) Note: one metabolite can map to more than "
                                    "one pathway. If two or more metabolites overlap a pathway, it applies single-sample pathway "
                                    "analysis (ssPA) via KPCA to compute an arbitrary score for each pathway in each patient sample.",
                                    style={"marginBottom": "0.5rem"}
                                ),

                                html.P(
                                    "Finally, differential testing is performed on those pathway scores. It splits those pathway scores "
                                    "into Case and Control groups and runs a two-tailed t-test (with Benjamini–Hochberg FDR correction) "
                                    "on each pathway score to identify pathways that are significantly differentially abundant between "
                                    "groups (adjusted p-value below 0.05).",
                                    style={"marginBottom": "0.5rem"}
                                ),

                                html.P([
                                    html.B("Note:"),
                                    " since KPCA computes an arbitrary score for each pathway "
                                    "no direction (up/down) of change can be inferred from these values."],
                                    style={"marginBottom": "0.5rem"}
                                ),
                            ],
                            style={
                                "backgroundColor": "#f0f0f0",
                                "padding": "1rem",
                                "borderRadius": "5px",
                                "marginBottom": "1.5rem",
                            },
                        ),

                        # always‐visible “num pathways” input
                        html.Div(
                            [
                                dbc.Label("Number of pathways to plot:"),
                                dcc.Input(
                                    id="num-top-pathways",
                                    type="number", min=1, max=50, step=1, value=10,
                                    style={"width": "100px", "marginBottom": "1rem"}
                                ),
                            ],
                            style={"textAlign": "center"},
                        ),

                        # always‐visible Save button for the **chart**
                        html.Div(
                            dbc.Button(
                                "Save chart",
                                id="open-save-modal-pathway-chart",
                                n_clicks=0,
                                style={"backgroundColor": "white", "color": "black"},
                            ),
                            className="d-flex justify-content-end mb-2",
                        ),

                        # chart placeholder (blank until callback fills it)
                        dcc.Loading(
                            id="loading-pathway-chart",
                            children=html.Div(
                                id="pathway-chart-content",
                                style={
                                    "display":        "flex",
                                    "justifyContent": "center",
                                    "width":          "100%",
                                    "minHeight":      "300px",
                                }
                            )
                        ),

                        # Store, Modal, feedback for **chart**
                        dcc.Store(id="pathway-chart-store"),
                        dbc.Modal(
                            [
                                dbc.ModalHeader("Name your pathway chart"),
                                dbc.ModalBody(
                                    dcc.Input(
                                        id="plot-name-input-pathway-chart",
                                        type="text",
                                        placeholder="Enter filename…",
                                        style={"width": "100%"},
                                    )
                                ),
                                dbc.ModalFooter(
                                    dbc.Button(
                                        "Save",
                                        id="confirm-save-plot-button-pathway-chart",
                                        color="primary",
                                        className="ms-auto",
                                        n_clicks=0,
                                    )
                                ),
                            ],
                            id="save-plot-modal-pathway-chart",
                            is_open=False,
                            size="sm",
                        ),


                        # always‐visible Save button for the **table**
                        html.Div(
                            dbc.Button(
                                "Save table",
                                id="open-save-modal-pathway-table",
                                n_clicks=0,
                                style={"backgroundColor": "white", "color": "black"},
                            ),
                            className="d-flex justify-content-end mb-2",
                        ),

                        # table placeholder (blank until callback fills it)
                        dcc.Loading(
                            id="loading-pathway-table",
                            children=html.Div(
                                id="pathway-table-content",
                                style={
                                    "display":        "flex",
                                    "justifyContent": "center",
                                    "width":          "100%",
                                    "minHeight":      "300px",
                                }
                            )
                        ),

                        # Store, Modal, feedback for **table**
                        dcc.Store(id="pathway-table-store"),
                        dbc.Modal(
                            [
                                dbc.ModalHeader("Name your pathway table"),
                                dbc.ModalBody(
                                    dcc.Input(
                                        id="plot-name-input-pathway-table",
                                        type="text",
                                        placeholder="Enter filename…",
                                        style={"width": "100%"},
                                    )
                                ),
                                dbc.ModalFooter(
                                    dbc.Button(
                                        "Save",
                                        id="confirm-save-plot-button-pathway-table",
                                        color="primary",
                                        className="ms-auto",
                                        n_clicks=0,
                                    )
                                ),
                            ],
                            id="save-plot-modal-pathway-table",
                            is_open=False,
                            size="sm",
                        ),

                    ], style={"padding": "1rem"})

def register_callbacks():
    # Callback for pathway analysis using the selected project and file.
    @callback(
        Output("pathway-chart-content", "children"),
        Output("pathway-chart-store",   "data"),
        Output("pathway-table-content", "children"),
        Output("pathway-table-store",   "data"),
        [
            Input("project-dropdown-pop",   "value"),
            Input("project-files-dropdown", "value"),
            Input("num-top-pathways",       "value")
        ]
    )
    def update_pathway_analysis(selected_project, selected_file, top_n):
        # Check if both a project and a file have been selected.
        if not selected_project or not selected_file:
            return html.Div("Please select a project and a file for differential pathway analysis."), None, None, None

        # Construct the full processed file path.
        processed_filepath = os.path.join("Projects", selected_project, "processed-datasets", selected_file)
        
        if not os.path.exists(processed_filepath):
            logger.error(f"Differential pathway tab - Processed file '{processed_filepath}' not found")
            return html.Div(f"Processed file '{processed_filepath}' not found."), None, None, None

        # Extract the study name from the file name.
        parts = selected_file.split('_')
        if len(parts) >= 2:
            study_name = parts[1]
        else:
            study_name = selected_file

        # Build the details file path (unchanged logic).
        folder_details = os.path.join("pre-processed-datasets", study_name)
        details = read_study_details_dpp2(folder_details)
        dataset_source = details.get("Dataset Source", "").lower()
        try:
            # Load the processed data and set its index.
            processed_data = pd.read_csv(processed_filepath)
            processed_data = processed_data.set_index('database_identifier')
        except Exception:
            logger.exception(f"Differential pathway tab - Error reading processed file: {selected_file}")
            return html.Div(f"Error reading processed file"), None, None, None
        
        try:
            # Process GMT file and obtain pathway definitions.
            reactome_paths = sspa.process_gmt(infile='Reactome_Homo_sapiens_pathways_ChEBI_R90.gmt')
            reactome_dict = sspa.utils.pathwaydf_to_dict(reactome_paths)

            # Keep a copy of the original column names so we can count RefMet→ChEBI conversions
            original_cols = list(processed_data.columns)

            # If the dataset uses RefMet IDs, do the conversion
            if dataset_source in ("metabolomics workbench", "original data - refmet ids"):
                keep_cols = {'database_identifier', 'group_type'}
                drop_columns = []
                rename_mapping = {}

                for col in original_cols:
                    if col in keep_cols:
                        rename_mapping[col] = col
                    else:
                        new_name = name2pubchem.get(col, None)
                        if new_name is None or pd.isna(new_name):
                            drop_columns.append(col)
                        else:
                            rename_mapping[col] = new_name

                # Count how many RefMet IDs mapped vs. didn’t
                # (exclude 'database_identifier' & 'group_type' from the “RefMet_ID” set)
                refmet_cols = [c for c in original_cols if c not in keep_cols]
                num_unmapped_refmet = len(drop_columns)
                num_mapped_refmet   = len([c for c in refmet_cols if c not in drop_columns])

                # Now actually drop and rename
                processed_data = processed_data.drop(columns=drop_columns)
                processed_data = processed_data.rename(columns=rename_mapping)
            else:
                # If not RefMet, set these counts to None or 0 so we can skip showing them later
                num_mapped_refmet = None
                num_unmapped_refmet = None

            # Build a mapping of pathway IDs → pathway names
            pathway_names = dict(zip(reactome_paths.index, reactome_paths['Pathway_name']))

            # Strip any “CHEBI:” prefix
            processed_data.columns = processed_data.columns.str.removeprefix("CHEBI:")

            # ─────────────────────────────────────────────────────────────────
            # Compute mapped vs. unmapped metabolites in Reactome
            # ─────────────────────────────────────────────────────────────────
            all_rxn_ids = set().union(*reactome_dict.values())
            metabolite_cols   = set(processed_data.columns) - {"group_type"}
            total_metabolites = len(metabolite_cols)

            mapped_metabolites   = metabolite_cols.intersection(all_rxn_ids)
            unmapped_metabolites = metabolite_cols.difference(all_rxn_ids)

            num_mapped   = len(mapped_metabolites)
            num_unmapped = len(unmapped_metabolites)
            pct_mapped   = (num_mapped / total_metabolites * 100) if total_metabolites > 0 else 0.0

            mapped_pathways = {
                pid
                for pid, members in reactome_dict.items()
                if set(members).intersection(metabolite_cols)
            }
            num_mapped_pathways = len(mapped_pathways)

            # Build the grey‐box Div.  If this was a RefMet dataset, show the mapped/unmapped counts first:
            stats_children = []

            if num_mapped_refmet is not None:
                stats_children += [
                    html.H4("RefMet-to-ChEBI Conversion Summary", style={"marginTop": "0.5rem"}),
                    html.P(f"RefMet IDs successfully converted to ChEBI: {num_mapped_refmet}"),
                    html.P(f"RefMet IDs NOT converted (dropped): {num_unmapped_refmet}"),
                    html.Hr(style={"borderColor": "#ddd", "margin": "0.5rem 0"})
                ]

            stats_children += [
                html.H4("Metabolite Coverage Summary", style={"marginTop": "0.5rem"}),
                html.P(f"Total metabolites detected: {total_metabolites}"),
                html.P(f"• Mapped to ≥1 Reactome pathway: {num_mapped}"),
                html.P(f"• Not mapped to any Reactome pathway: {num_unmapped}"),
                html.P(f"→ Percentage mapped: {pct_mapped:.1f}%"),
                html.P(f"• Unique Reactome pathways covered: {num_mapped_pathways}"),
            ]

            mapping_stats_div = html.Div(
                stats_children,
                style={
                    "border": "1px solid #ddd",
                    "padding": "0.75rem",
                    "marginBottom": "1rem",
                    "backgroundColor": "#fafafa",
                    "borderRadius": "4px"
                },
            )
            # ─────────────────────────────────────────────────────────────────────────────

            # Calculate pathway coverage statistics.
            coverage_dict = {k: len(set(processed_data.columns).intersection(set(v))) 
                            for k, v in reactome_dict.items()}
            nonzero_coverages = [cov for cov in coverage_dict.values() if cov > 0]
            if nonzero_coverages and all(cov == 1 for cov in nonzero_coverages):
                return html.Div([
                    html.H3("Single‑Sample Pathway Analysis"),
                    html.P([
                        "⚠️ Every pathway that overlaps your dataset maps to exactly ",
                        html.Strong("one"),
                        " measured metabolite. KPCA (and most multivariate ssPA methods) require at least ",
                        html.Strong("two"),
                        " features per pathway to compute a principal component. Because no overlapping pathway contains multiple metabolites in this dataset, single‑sample pathway analysis via KPCA cannot be performed."
                    ])
                ]), None, None, None
            
            # Perform ssPA using KPCA.
            scores = sspa.sspa_KPCA(reactome_paths).fit_transform(processed_data.iloc[:, :-1])
            scores['group_type'] = processed_data['group_type']
            
            # Rename pathway columns with pathway names.
            new_columns = {col: pathway_names.get(col, col) for col in scores.columns if col not in ['Group', 'group_type']}
            scores.rename(columns=new_columns, inplace=True)
            
            # Save the KPCA scores.
            results_folder = os.path.join("raw_results_data", "pathway analysis")
            os.makedirs(results_folder, exist_ok=True)
            base_filename = selected_file.replace('.csv', '')
            save_filename = f"KPCA_results{base_filename}.csv"
            save_filepath = os.path.join(results_folder, save_filename)
            scores.to_csv(save_filepath)
            
            # Importance metric and differential testing on pathway scores.
            importance = scores.drop(['group_type'], axis=1).abs().mean().sort_values(ascending=False)
            X_case = scores[scores['group_type'] == 'Case'].select_dtypes(include='number')
            X_ctrl = scores[scores['group_type'] == 'Control'].select_dtypes(include='number')
            valid_columns = [
                col for col in X_case.columns 
                if (np.std(X_case[col].to_numpy(), ddof=1) != 0 or 
                    np.std(X_ctrl[col].to_numpy(), ddof=1) != 0)]
            X_case_valid = X_case[valid_columns]
            X_ctrl_valid = X_ctrl[valid_columns]
            stat, pvals = stats.ttest_ind(X_case_valid, X_ctrl_valid, nan_policy='raise')
            pval_df = pd.DataFrame({
                'P-value': pvals,
                'Stat': stat
            }, index=X_case_valid.columns)
            pval_df['FDR_P-value'] = multipletests(pvals, method='fdr_bh')[1]
            sig = pval_df[pval_df['FDR_P-value'] < 0.05].sort_values('FDR_P-value')
            if sig.empty:
                pathway_table = html.Div("No significantly different pathways (FDR < 0.05).")
                fig_box = None
            else:
                top_paths = sig.index.tolist()[:10]
                sig_sorted = sig.sort_values("FDR_P-value").round(3)
                sig_sorted = sig.sort_values("FDR_P-value").copy()
                # Format values in scientific notation.
                sig_sorted['P-value'] = sig_sorted['P-value'].apply(lambda x: f"{x:.3e}")
                sig_sorted['FDR_P-value'] = sig_sorted['FDR_P-value'].apply(lambda x: f"{x:.3e}")
                sig_sorted['Stat'] = sig_sorted['Stat'].round(3)
                pathway_table = dash_table.DataTable(
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
                top_paths = list(sig_sorted.index)[: (top_n or 10)]
                ordered_paths = list(sig_sorted.loc[top_paths].index)
                title = f"Box Plot of Top {len(top_paths)} Differentially Significant Pathways"
                box_df = scores[top_paths + ['group_type']].reset_index(drop=True)
                box_long = pd.melt(
                    box_df,
                    id_vars=['group_type'],
                    value_vars=top_paths,
                    var_name='Pathway',
                    value_name='Score'
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

                box_long['Group_Label'] = box_long['group_type'].map(group_labels)
                fig_box = px.box(
                    box_long,
                    x="Pathway",
                    y="Score",
                    color="Group_Label",
                    title=title,
                    labels={"Score": "KPCA Pathway Score"},
                    category_orders={"Pathway": ordered_paths}
                )

                # —— size the width depending on requested number of pathways ——  
                NEW_H = 400
                orig_w = fig_box.layout.width  or 700
                orig_h = fig_box.layout.height or 450
                aspect = orig_w / orig_h
                BASE_W = int(aspect * NEW_H) + 200

                # width driven by requested count (but never more than actual)
                requested = top_n or 10
                top_paths = sig_sorted.index.tolist()[:requested]
                actual    = len(top_paths)
                n_for_width = requested if requested <= actual else actual
                BAR_PX      = 50
                bar_needed  = n_for_width * BAR_PX
                NEW_W       = max(BASE_W, bar_needed)

                #  (1) Compute the longest label length in characters
                max_label_len = max(len(str(lbl)) for lbl in ordered_paths)
                #  (2) Turn that into an estimated pixel height needed for rotated labels
                PX_PER_CHAR         = 5   # px of vertical space per character (45°‐rotated)
                estimated_label_px  = max_label_len * PX_PER_CHAR
                NEW_H = 120 + estimated_label_px
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
                
            # serialize both
            fig_json  = pio.to_json(fig_box)
            csv_bytes = sig_sorted.reset_index().to_csv(index=False).encode()
            table_b64 = base64.b64encode(csv_bytes).decode()

            # wrap each in a centered div
            chart_child = html.Div(
                dcc.Graph(figure=fig_box, style={"width":f"{NEW_W}px","height":f"{NEW_H}px"}),
                style={"display":"flex","justifyContent":"center","padding":"0 1rem","boxSizing":"border-box"}
            )
            table_child = html.Div(
                pathway_table,
                style={"width":"100%","padding":"0 1rem","boxSizing":"border-box"}
            )

            # ✅ Log success
            logger.info(f"Differential pathway tab - Successfully perform pathway analysis on: {selected_file}")

            return (
                html.Div(
                    [
                        mapping_stats_div,
                        chart_child
                    ],
                    style={"display": "flex", "flexDirection": "column", "alignItems": "center"}
                ),
                {"type":"plotly","data":pio.to_json(fig_box)},
                table_child,
                {"type":"csv","data":base64.b64encode(
                    sig_sorted.reset_index().to_csv(index=False).encode()
                ).decode()}
            )
        except Exception:
            logger.exception(f"Differential pathway tab - Error during pathway analysis: {selected_file}")
            return html.Div(f"Error during pathway analysis"), None, None, None

    # Callback controls the save box plot pop up from opening
    @callback(
        Output("save-plot-modal-pathway-chart", "is_open"),
        [
            Input("open-save-modal-pathway-chart", "n_clicks"),
            Input("confirm-save-plot-button-pathway-chart", "n_clicks"),
        ],
        [ State("save-plot-modal-pathway-chart", "is_open") ]
    )
    def toggle_pathway_chart_modal(open_n, save_n, is_open):
        if open_n or save_n:
            return not is_open
        return is_open

    # Callback controls the save table pop up from opening
    @callback(
        Output("save-plot-modal-pathway-table", "is_open"),
        [
            Input("open-save-modal-pathway-table", "n_clicks"),
            Input("confirm-save-plot-button-pathway-table", "n_clicks"),
        ],
        [ State("save-plot-modal-pathway-table", "is_open") ]
    )
    def toggle_pathway_table_modal(open_n, save_n, is_open):
        if open_n or save_n:
            return not is_open
        return is_open

    # Callback which saves **chart** as SVG
    @callback(
        Input("confirm-save-plot-button-pathway-chart","n_clicks"),
        [
        State("plot-name-input-pathway-chart","value"),
        State("pathway-chart-store","data"),
        State("project-dropdown-pop","value"),
        ]
    )
    def save_pathway_chart(nc, fn, payload, project):
        # no n_clicks yet → do nothing
        if not nc:
            return
        # validation errors printed to console
        if not project:
            logger.error("Differential pathway tab - No project selected for chart")
            return
        if not fn:
            logger.error("Differential pathway tab - No filename provided for chart")
            return
        if not payload:
            logger.error("Differential pathway tab - No chart data to save")
            return

        fig = pio.from_json(payload["data"])
        w = fig.layout.width  or 700
        h = fig.layout.height or 400

        out_dir = os.path.join("Projects", project,
                            "Plots", "Single-study-analysis",
                            "Differential-pathway-box-plots")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{fn}.svg")

        pio.write_image(fig, path, format="svg", width=int(w), height=int(h))

        # ✅ Log success
        logger.info(f"Differential pathway tab - Successfully saved chart: {path}")


    # Callback which saves **table** as CSV
    @callback(
        Input("confirm-save-plot-button-pathway-table","n_clicks"),
        [
        State("plot-name-input-pathway-table","value"),
        State("pathway-table-store","data"),
        State("project-dropdown-pop","value"),
        ]
    )
    def save_pathway_table(nc, fn, payload, project):
        # no n_clicks yet → do nothing
        if not nc:
            return
        # validation errors printed to console
        if not project:
            logger.error("Differential pathway tab - No project selected for table")
            return
        if not fn:
            logger.error("Differential pathway tab - No filename provided for table")
            return
        if not payload:
            logger.error("Differential pathway tab - No table data to save")
            return

        data = base64.b64decode(payload["data"])
        out_dir = os.path.join("Projects", project,
                            "Plots", "Single-study-analysis",
                            "Differential-pathway-table-plots")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{fn}.csv")
        with open(path, "wb") as f:
            f.write(data)

        # ✅ Log success
        logger.info(f"Differential pathway tab - Successfully saved table: {path}")

