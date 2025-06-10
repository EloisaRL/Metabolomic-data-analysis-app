# pages/multi_study_analysis_page_tabs/upset_plots.py
from dash import html, dcc, callback, Input, Output, State, no_update, dash_table
import dash_bootstrap_components as dbc
import os
import pandas as pd
from marsilea.upset import UpsetData, Upset
import matplotlib.pyplot as plt
import io
import base64
from scipy import stats
from statsmodels.stats.multitest import multipletests
import plotly.io as pio
import libchebipy

UPLOAD_FOLDER = "pre-processed-datasets"

refmet = pd.read_csv(r"C:\Users\Eloisa\Documents\ICL\Tim RA Project - Postgraduate\my_dash_app\refmet.csv", dtype=object)
refmet.columns = refmet.columns.str.strip() 
refmet2chebi = dict(zip(refmet['refmet_name'], refmet['chebi_id']))

def read_study_details_msa(folder):
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

def wrap_fig(fig):
    # If it’s already a Plotly figure (has to_plotly_json), just send it straight to dcc.Graph
    if hasattr(fig, "to_plotly_json"):
        return dcc.Graph(figure=fig)
    # Otherwise assume it’s a Matplotlib Figure and embed as base64-png
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return html.Img(src="data:image/png;base64," + encoded)

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


layout = html.Div([
                    html.H2("Upset Plots of Co-Occurring Metabolites and Differential Metabolites"),

                    # Background processing description in a grey box
                    html.Div(
                        [
                            html.H4("Background processing description", style={"marginBottom": "0.5rem"}),
                            html.P(
                                "If the dataset uses RefMet IDs (i.e. originates from workbench or is original data), "
                                "RefMet-to–ChEBI conversion is performed renaming each metabolite column to its "
                                "corresponding ChEBI ID (dropping any unmapped columns). For all datasets, ChEBI ids are "
                                "converted into Metabolite names, using libchebipy.ChebiEntity, before creating upset plots.",
                                style={"marginBottom": "0.5rem"}
                            ),
                            html.P(
                                "For the upset plot of co-occurring differential metabolites, differential testing is performed "
                                "by first separating metabolite data into Case and Control groups, then runs an independent "
                                "two‐sided t‐test for each metabolite to compare their means. It labels each metabolite as “Up” "
                                "or “Down” based on the sign of the test statistic, applies Benjamini–Hochberg FDR correction "
                                "to the p-values, and finally reports those metabolites with an adjusted p-value below 0.05 as "
                                "differentially abundant."
                            ),
                        ],
                        style={
                            "backgroundColor": "#f0f0f0",
                            "padding": "1rem",
                            "borderRadius": "5px",
                            "marginBottom": "1.5rem",
                        },
                    ),

                    # always-visible “min co-occur” input
                    html.Div(
                        [
                            dbc.Label("Minimum number of co-occurring metabolites:"),
                            dcc.Input(
                                id="min-num-co-occur-metabolites-msa",
                                type="number", min=1, max=50, step=1, value=1,
                                style={"width": "100px", "marginBottom": "1rem"}
                            ),
                        ],
                        id="min-co-occur-wrapper",
                        style={"display": "block", "textAlign": "center"},
                    ),

                    # always-visible Save button for Upset plot
                    html.Div(
                        dbc.Button(
                            "Save upset plot",
                            id="open-save-modal-upset-msa",
                            n_clicks=0,
                            style={"backgroundColor": "white", "color": "black"},
                        ),
                        id="save-upset-wrapper-msa",
                        className="d-flex justify-content-end mb-2",
                        style={"display": "flex"},  # flex → right-aligned via justify-content-end
                    ),

                    # graph placeholder (blank until callback fills it)
                    dcc.Loading(
                        id="loading-upset_plots-msa",
                        children=html.Div(
                            id="upset_plots-content-msa",
                            style={
                                "display":        "flex",
                                "justifyContent": "center",
                                "width":          "100%",
                                "minHeight":      "300px",
                            }
                        )
                    ),

                    # Store, Modal, feedback for Upset
                    dcc.Store(id="upset-plot-store-msa"),
                    dbc.Modal(
                        [
                            dbc.ModalHeader("Name your Upset plot"),
                            dbc.ModalBody(
                                dcc.Input(
                                    id="plot-name-input-upset-msa",
                                    type="text",
                                    placeholder="Enter plot name…",
                                    style={"width": "100%"},
                                )
                            ),
                            dbc.ModalFooter(
                                dbc.Button(
                                    "Save",
                                    id="confirm-save-plot-button-upset-msa",
                                    color="primary",
                                    className="ms-auto",
                                    n_clicks=0,
                                )
                            ),
                        ],
                        id="save-plot-modal-upset-msa",
                        is_open=False,
                        size="sm",
                    ),
                    html.Div(id="save-feedback-upset-msa"),


                    # always-visible “min diff co-occur” input
                    html.Div(
                        [
                            dbc.Label("Minimum number of co-occurring differential metabolites:"),
                            dcc.Input(
                                id="min-num-co-occur-diff-metabolites-msa",
                                type="number", min=1, max=50, step=1, value=1,
                                style={"width": "100px", "marginTop": "1rem", "marginBottom": "1rem"}
                            ),
                        ],
                        id="min-co-occur-wrapper-diff",
                        style={"display": "block", "textAlign": "center"},
                    ),

                    # always-visible Save button for Differential plot
                    html.Div(
                        dbc.Button(
                            "Save differential plot",
                            id="open-save-modal-diff-msa",
                            n_clicks=0,
                            style={"backgroundColor": "white", "color": "black"},
                        ),
                        id="save-diff-wrapper-msa",
                        className="d-flex justify-content-end mb-2",
                        style={"display": "flex"},
                    ),

                    # graph placeholder (blank until callback fills it)
                    dcc.Loading(
                        id="loading-differential-msa",
                        children=html.Div(
                            id="differential-analysis-content-msa",
                            style={
                                "display":        "flex",
                                "justifyContent": "center",
                                "width":          "100%",
                                "minHeight":      "300px",
                            }
                        )
                    ),


                    # Store, Modal, feedback for Differential
                    dcc.Store(id="diff-plot-store-msa"),
                    dbc.Modal(
                        [
                            dbc.ModalHeader("Name your Differential plot"),
                            dbc.ModalBody(
                                dcc.Input(
                                    id="plot-name-input-diff-msa",
                                    type="text",
                                    placeholder="Enter plot name…",
                                    style={"width": "100%"},
                                )
                            ),
                            dbc.ModalFooter(
                                dbc.Button(
                                    "Save",
                                    id="confirm-save-plot-button-diff-msa",
                                    color="primary",
                                    className="ms-auto",
                                    n_clicks=0,
                                )
                            ),
                        ],
                        id="save-plot-modal-diff-msa",
                        is_open=False,
                        size="sm",
                    ),
                    html.Div(id="save-feedback-diff-msa"),

                ])

def register_callbacks():
    @callback(
        Output("upset_plots-content-msa", "children"),
        Output("differential-analysis-content-msa", "children"),
        Output("upset-plot-store-msa",              "data"),   
        Output("diff-plot-store-msa",               "data"),
        [
            Input("process-data-button-msa", "n_clicks"),
            Input("min-num-co-occur-metabolites-msa", "value"),
            Input("min-num-co-occur-diff-metabolites-msa", "value"),
            Input("multi-study-analysis-tabs", "value"),
        ],
        [
            State("project-files-checklist-msa", "value"),
            State("project-dropdown-pop-msa", "value"),
        ]
    )
    def process_upset_data(n_clicks, min_cooccurring, min_co_occur_metabolites, active_tab,
                                selected_files, selected_project):
        # styles for hiding/showing the controls
        hide = {"display": "none", "textAlign": "center"}
        show = {"display": "block", "textAlign": "center"}
        # new vars for the button wrappers
        hide_buttons = {"display": "none"}
        show_buttons = {"display": "flex"}   # flex so justify-content-end works

        # 1) only run when user clicked “Process” and is on the co-occurring tab
        if not n_clicks or active_tab != "upset_plots":
            return no_update, no_update, no_update, no_update

        # 2) validate inputs
        if not selected_project or not selected_files:
            return html.Div("Please select a project and at least one file."), html.Div("Please select a project and at least one file."), no_update, no_update

        # Before looping over files, initialize a list to collect mapping stats
        mapping_records = []

        # 1) load each file, grab the full set of metabolites
        studies = []
        for fname in selected_files:
            path = os.path.join("Projects", selected_project, "processed-datasets", fname)
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path).set_index("database_identifier")
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                continue

            # quick container for name + metabolite set
            class Study: pass
            st = Study()
            
            # derive a clean node_name from filename (just like your other tab)
            parts = fname.split("_")
            study_name = parts[1] if len(parts) >= 3 else fname
            st.node_name = study_name

            folder = os.path.join(UPLOAD_FOLDER, study_name)
            details = read_study_details_msa(folder)
            dataset_source = details.get("Dataset Source", "").lower()

            # ─── map refMet IDs to CheBI IDs ──────────────────────────────────
            all_cols = list(df.columns)
            keep_cols = {'database_identifier', 'group_type'}

            if dataset_source in ("metabolomics workbench", "original data - refmet ids"):
                # 1) Restrict to only the columns you actually want to map
                met_cols = [c for c in all_cols if c not in keep_cols]

                # 2) Find which of those can’t be mapped
                unmapped = [c for c in met_cols if c not in refmet2chebi]

                # Compute mapping metrics
                total_refmet   = len(met_cols)
                num_unmapped   = len(unmapped)
                num_mapped     = total_refmet - num_unmapped
                pct_unmapped   = (num_unmapped / total_refmet * 100) if total_refmet else 0.0

                # Append this study’s stats to mapping_records
                mapping_records.append({
                    "study_name":    study_name,
                    "total_refmet":  total_refmet,
                    "num_mapped":    num_mapped,
                    "num_unmapped":  num_unmapped,
                    "pct_unmapped":  round(pct_unmapped, 1)
                })

                # 3) Drop only the unmapped metabolite columns
                df = df.drop(columns=unmapped)

                # 4) Rename only those that _are_ in your lookup
                to_rename = {c: refmet2chebi[c] for c in df.columns if c in refmet2chebi}
                df = df.rename(columns=to_rename)

            ##### Renaming to from ChEBI -> Metabolite name #####
            df_renamed = df.copy()

            # Define columns to exclude from renaming
            exceptions = {'database_identifier', 'group_type'}

            # Build a mapping of old column names to new names
            new_column_names = {}

            for col in df_renamed.columns:
                if col in exceptions:
                    new_column_names[col] = col
                else:
                    chebi_id = str(col).replace("CHEBI:", "")
                    try:
                        entity = libchebipy.ChebiEntity(chebi_id)
                        new_column_names[col] = entity.get_name()
                    except Exception:
                        new_column_names[col] = col  # Fallback if not a valid ChEBI ID

            # Apply the renaming
            df_renamed.columns = [new_column_names[col] for col in df_renamed.columns]

            # now every column name is guaranteed to be a CheBI string
            st.metabolites = set(df_renamed.columns)

            studies.append(st)

        if not studies:
            return html.Div("No studies were successfully processed."), html.Div("No studies were successfully processed."), no_update, no_update

        # 2) build the upset data from raw sets
        all_sets_coor = [st.metabolites for st in studies]
        upset_data_coor = UpsetData.from_sets(
            all_sets_coor,
            sets_names=[st.node_name for st in studies]
        )

        # 3) draw the upset plot: side-bars are total per‐study, main bars are intersections
        us_coor = Upset(
            data=upset_data_coor,
            orient="h",             # horizontal bars
            add_labels=True,
            min_cardinality=min_cooccurring
        )
        us_coor.add_legends(box_padding=0)
        us_coor.set_margin(0.3)
        us_coor.add_title(top="Co-occurring metabolites across studies", pad=0.25)

        fig_coor = us_coor.render() or plt.gcf()

        ########################################################################
        # Differential upset plot
        ########################################################################
        studies = []
        for file in selected_files:
            filepath = os.path.join("Projects", selected_project, "processed-datasets", file)
            if not os.path.exists(filepath):
                continue
            try:
                df = pd.read_csv(filepath)
                df = df.set_index('database_identifier')
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
            
            # Extract the node name: use the word after the first '_' but before the second '_'
            parts = file.split('_')
            if len(parts) >= 3:
                node_name = parts[1]
            else:
                node_name = file  # Fallback if not enough '_' characters.

            # ─── map refMet IDs to CheBI IDs ──────────────────────────────────
            all_cols = list(df.columns)
            total = len(all_cols)  
            keep_cols = {'database_identifier', 'group_type'} 

            folder = os.path.join(UPLOAD_FOLDER, node_name)
            details = read_study_details_msa(folder)   # or however you get dataset_source
            src = details.get("Dataset Source", "").lower()
            if src in ("metabolomics workbench", "original data - refmet ids"):
                # 1) restrict to only the columns you actually want to map
                met_cols = [c for c in all_cols if c not in keep_cols]

                # 2) find which of those can’t be mapped
                unmapped = [c for c in met_cols if c not in refmet2chebi]

                # report on mapping efficiency
                total = len(met_cols)
                num_unmapped = len(unmapped)
                pct_unmapped = num_unmapped/total*100 if total else 0
                print(f"{node_name}: {num_unmapped}/{total} metabolites "
                    f"({pct_unmapped:.1f}%) could not be mapped")

                # 3) drop only the unmapped metabolite columns
                df = df.drop(columns=unmapped)

                # 4) rename only those that _are_ in your lookup
                keep_cols = {'database_identifier', 'group_type'}
                drop_columns = []
                rename_mapping = {}
                for col in df.columns:
                    if col in keep_cols:
                        rename_mapping[col] = col
                    else:
                        new_name = refmet2chebi.get(col, None)
                        if new_name is None or pd.isna(new_name):
                            drop_columns.append(col)
                        else:
                            rename_mapping[col] = new_name
                df = df.drop(columns=drop_columns)
                df = df.rename(columns=rename_mapping)
                """ to_rename = {c: refmet2chebi[c] for c in df.columns 
                            if c in refmet2chebi}
                df = df.rename(columns=to_rename) """

            ##### Renaming to from ChEBI -> Metabolite name #####
            df_renamed = df.copy()

            # Define columns to exclude from renaming
            exceptions = {'database_identifier', 'group_type'}

            # Build a mapping of old column names to new names
            new_column_names = {}

            for col in df_renamed.columns:
                if col in exceptions:
                    new_column_names[col] = col
                else:
                    chebi_id = str(col).replace("CHEBI:", "")
                    try:
                        entity = libchebipy.ChebiEntity(chebi_id)
                        new_column_names[col] = entity.get_name()
                    except Exception:
                        new_column_names[col] = col  # Fallback if not a valid ChEBI ID

            # Apply the renaming
            df_renamed.columns = [new_column_names[col] for col in df_renamed.columns]

            class DA:
                pass
            da = DA()
            da.processed_data = df_renamed
            da.pathway_data = None
            da.node_name = node_name
            #da.node_name = f"{file}"
            da.pathway_level = False
            da.da_testing = da_testing.__get__(da, DA)
            try:
                da.da_testing()
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
            
            studies.append(da)

        if not studies:
            return html.Div("No studies were successfully processed."), html.Div("No studies were successfully processed."), no_update, no_update

        all_da_sets_diff = [set(st.DA_metabolites) for st in studies if hasattr(st, "DA_metabolites")]
        if not all_da_sets_diff or not all_da_sets_diff[0]:
            return html.Div("No differentially abundant metabolites found across the selected studies."), html.Div("No differentially abundant metabolites found across the selected studies."), no_update, no_update

        all_mets_diff = set.union(*all_da_sets_diff)
        data = []
        for st in studies:
            mets = set(st.DA_metabolites) if hasattr(st, "DA_metabolites") else set()
            row = {met: 1 if met in mets else 0 for met in all_mets_diff}
            data.append(row)
        biadj_df = pd.DataFrame(data)

        all_da_sets_diff = [set(st.DA_metabolites) for st in studies]

        # 1) build the UpsetData from all DA sets
        upset_data_diff = UpsetData.from_sets(
            all_da_sets_diff,
            sets_names=[st.node_name for st in studies]
        )

        # 2) draw the full DA-occurrence upset plot
        us_diff = Upset(
            data=upset_data_diff,
            orient="h",
            add_labels=True,      # exactly as in your last snippet
            min_cardinality=min_co_occur_metabolites
        )
        us_diff.add_legends(box_padding=0)
        us_diff.set_margin(0.3)
        us_diff.add_title(top="Co-occurring differential metabolites across studies", pad=0.25)

        # 3) render and return
        fig_diff = us_diff.render() or plt.gcf()
        
        # now wrap each one appropriately
        graph_coor = wrap_fig(fig_coor)
        graph_diff = wrap_fig(fig_diff)

        # 2) prepare store payloads
        def make_payload(fig):
            if hasattr(fig, "to_json"):
                # Plotly figure → store JSON
                return {"type": "plotly", "data": fig.to_json()}
            else:
                # Matplotlib figure → store base64‐SVG
                buf = io.BytesIO()
                fig.savefig(buf, format="svg", bbox_inches="tight")
                buf.seek(0)
                svg_b64 = base64.b64encode(buf.read()).decode("utf-8")
                return {"type": "mpl", "data": svg_b64}

        coor_payload = make_payload(fig_coor)
        diff_payload = make_payload(fig_diff)

        # Build the mapping‐summary table (only if mapping_summary_df has rows)
        mapping_summary_df = pd.DataFrame(mapping_records)
        if not mapping_summary_df.empty:
            mapping_table = dash_table.DataTable(
                data=mapping_summary_df.to_dict("records"),
                columns=[
                    {"name": "Study Name",    "id": "study_name"},
                    {"name": "Total RefMet",  "id": "total_refmet"},
                    {"name": "Mapped",        "id": "num_mapped"},
                    {"name": "Unmapped",      "id": "num_unmapped"},
                    {"name": "% Unmapped",    "id": "pct_unmapped"},
                ],
                sort_action="native",
                page_size=10,
                style_table={
                    "overflowX": "auto",
                    "border": "1px solid #ccc",
                    "borderRadius": "5px",
                    "marginTop": "1rem"
                },
                style_header={
                    "backgroundColor": "#f2f2f2",
                    "fontWeight": "bold",
                    "textAlign": "left",
                    "borderBottom": "1px solid #ddd",
                    "padding": "8px"
                },
                style_cell={
                    "textAlign": "left",
                    "padding": "8px",
                    "borderBottom": "1px solid #eee"
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"}
                ]
            )
            # Wrap it in a Div with a title, so it appears under the diff plot:
            mapping_summary_div = html.Div(
                [
                    html.H4("RefMet→ChEBI Mapping Summary", style={"marginTop": "2rem"}),
                    mapping_table
                ],
                style={
                    "border": "1px solid #ddd",
                    "padding": "1rem",
                    "borderRadius": "5px",
                    "backgroundColor": "#fafafa",
                    "width": "100%",
                    "boxSizing": "border-box",
                    "marginTop": "1rem"
                }
            )
        else:
            # If there’s no mapping data, return an empty placeholder
            mapping_summary_div = html.Div()

        combined_diff = html.Div(
            [
                graph_diff,
                mapping_summary_div
            ],
            style={"display": "flex", "flexDirection": "column", "alignItems": "center"}
        )

        return graph_coor, combined_diff, coor_payload, diff_payload
        

    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "Project")


    # Upset‐plot modal toggle (unchanged)
    @callback(
        Output("save-plot-modal-upset-msa", "is_open"),
        [Input("open-save-modal-upset-msa","n_clicks"),
        Input("confirm-save-plot-button-upset-msa","n_clicks")],
        [State("save-plot-modal-upset-msa","is_open")]
    )
    def toggle_upset_modal(o, c, is_open):
        if o or c:
            return not is_open
        return is_open

    # Upset‐plot save
    @callback(
        Output("save-feedback-upset-msa", "children"),
        Input("confirm-save-plot-button-upset-msa", "n_clicks"),
        [
            State("project-dropdown-pop-msa", "value"),
            State("plot-name-input-upset-msa", "value"),
            State("upset-plot-store-msa", "data"),
        ]
    )
    def save_upset_plot(n_clicks, project, filename, payload):
        if not n_clicks:
            return no_update
        if not project:
            return dbc.Alert("Select a project before saving.", color="warning")
        if not filename:
            return dbc.Alert("Enter a name before saving.", color="warning")
        if not payload:
            return dbc.Alert("No plot data available.", color="danger")

        # Build the full directory:
        base_dir = os.path.join("Projects", project,
                                "Plots", "Multi-study-analysis", "Co-occurring-metabolites-upset-plots")
        # If it doesn't exist, bail out with an error
        if not os.path.isdir(base_dir):
            # print to console:
            print(f"ERROR: Cannot save plot — directory not found: {base_dir}")
            # show a red alert to the user:
            return dbc.Alert(
                f"❌ Could not save `{filename}.svg` because the folder "
                f"`{os.path.relpath(base_dir)}` does not exist.",
                color="danger",
                dismissable=True
            )

        out_path = os.path.join(base_dir, f"{filename}.svg")

        # Save the plot
        if payload["type"] == "plotly":
            fig = pio.from_json(payload["data"])
            pio.write_image(fig, out_path, format="svg")
        else:
            svg_bytes = base64.b64decode(payload["data"])
            with open(out_path, "wb") as f:
                f.write(svg_bytes)

        return dbc.Alert(
            f"Saved Upset plot as `{filename}.svg` in `{os.path.relpath(base_dir)}`.",
            color="success"
        )


    # Differential‐plot modal toggle (unchanged)
    @callback(
        Output("save-plot-modal-diff-msa", "is_open"),
        [Input("open-save-modal-diff-msa","n_clicks"),
        Input("confirm-save-plot-button-diff-msa","n_clicks")],
        [State("save-plot-modal-diff-msa","is_open")]
    )
    def toggle_diff_modal(o, c, is_open):
        if o or c:
            return not is_open
        return is_open

    # Differential‐plot save
    @callback(
        Output("save-feedback-diff-msa", "children"),
        Input("confirm-save-plot-button-diff-msa", "n_clicks"),
        [
            State("project-dropdown-pop-msa", "value"),
            State("plot-name-input-diff-msa", "value"),
            State("diff-plot-store-msa", "data"),
        ]
    )
    def save_diff_plot(n_clicks, project, filename, payload):
        if not n_clicks:
            return no_update
        if not project:
            return dbc.Alert("Select a project before saving.", color="warning")
        if not filename:
            return dbc.Alert("Enter a name before saving.", color="warning")
        if not payload:
            return dbc.Alert("No plot data available.", color="danger")

        # Build the full directory:
        base_dir = os.path.join("Projects", project,
                                "Plots", "Multi-study-analysis", "Differential-co-occurring-metabolites-upset-plots")
        
        # If it doesn't exist, bail out with an error
        if not os.path.isdir(base_dir):
            # print to console:
            print(f"ERROR: Cannot save plot — directory not found: {base_dir}")
            # show a red alert to the user:
            return dbc.Alert(
                f"❌ Could not save `{filename}.svg` because the folder "
                f"`{os.path.relpath(base_dir)}` does not exist.",
                color="danger",
                dismissable=True
            )

        out_path = os.path.join(base_dir, f"{filename}.svg")

        # Save the plot
        if payload["type"] == "plotly":
            fig = pio.from_json(payload["data"])
            pio.write_image(fig, out_path, format="svg")
        else:
            svg_bytes = base64.b64decode(payload["data"])
            with open(out_path, "wb") as f:
                f.write(svg_bytes)

        return dbc.Alert(
            f"Saved Differential plot as `{filename}.svg` in `{os.path.relpath(base_dir)}`.",
            color="success"
        )

    # … and so on …