# pages/single_study_analysis_page_tabs/differential_metabolites.py
from dash import html, dcc, callback, Input, Output, dash_table, State, no_update
import dash_bootstrap_components as dbc
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
import base64
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json

def da_testing(self):
        '''
        Performs differential analysis testing, adds pval_df attribute containing results.
        '''
        if self.pathway_level == True:
            dat = self.pathway_data
        else:
            dat = self.processed_data

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

# Layout fragment
layout = html.Div([
                    # Title
                    html.H2("Differential Metabolite Analysis"),

                    # Background processing description
                    html.Div(
                        [
                            html.H4("Background processing description", style={"marginBottom": "0.5rem"}),
                            html.P(
                                "Differential testing is performed by first separating metabolite "
                                "data into Case and Control groups, then runs an independent two‐sided t‐test "
                                "for each metabolite to compare their means. It labels each metabolite as “Up” "
                                "or “Down” based on the sign of the test statistic, applies Benjamini–Hochberg "
                                "FDR correction to the p-values, and finally reports those metabolites with an "
                                "adjusted p-value below 0.05 as differentially abundant."
                            ),
                        ],
                        style={
                            "backgroundColor": "#f0f0f0",
                            "padding": "1rem",
                            "borderRadius": "5px",
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
                    html.Div(id="save-feedback-chart"),


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
                    html.Div(id="save-feedback-table"),

                ], style={"padding": "1rem"})

# Register this tab’s callbacks
def register_callbacks():
    @callback(
        # 4 outputs now: chart‐DIV, chart‐store, table‐DIV, table‐store
        Output("differential-chart-content", "children"),
        Output("diff-chart-store",        "data"),
        Output("differential-table-content","children"),
        Output("diff-table-store",        "data"),
        [
            Input("project-dropdown-pop",       "value"),
            Input("project-files-dropdown",     "value"),
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
        except Exception as e:
            err = dbc.Alert(f"Error running differential analysis: {e}", color="danger")
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

        box_df = df[top_mets + ["group_type"]].reset_index(drop=True)
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

        return (
            chart_child,
            {"type":"plotly","data":fig_json},
            table_child,
            {"type":"csv","data":table_b64},
        )
    
    # Toggle Chart modal (unchanged)
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

    # Toggle Table modal (unchanged)
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


    # Save **chart** as SVG
    @callback(
        Output("save-feedback-chart","children"),
        Input("confirm-save-plot-chart","n_clicks"),
        [
        State("plot-name-input-chart","value"),
        State("diff-chart-store","data"),
        State("project-dropdown-pop","value"),
        ]
    )
    def save_chart(n_clicks, filename, payload, project):
        if not n_clicks:
            return no_update
        if not project:
            return dbc.Alert("Select a project before saving.", color="warning")
        if not filename:
            return dbc.Alert("Enter a name before saving.", color="warning")
        if not payload:
            return dbc.Alert("No chart data.", color="danger")

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

        return dbc.Alert(f"Chart saved as `{filename}.svg` ({width}×{height}px).", color="success")



    # Save **table** as CSV
    @callback(
        Output("save-feedback-table","children"),
        Input("confirm-save-plot-table","n_clicks"),
        [
        State("plot-name-input-table","value"),
        State("diff-table-store","data"),
        State("project-dropdown-pop","value"),
        ]
    )
    def save_table(n_clicks, filename, payload, project):
        if not n_clicks:
            return no_update
        if not project:
            return dbc.Alert("Select a project before saving.", color="warning")
        if not filename:
            return dbc.Alert("Enter a name before saving.", color="warning")
        if not payload:
            return dbc.Alert("No table data.", color="danger")

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

        return dbc.Alert(f"Table saved as `{filename}.csv`.", color="success")
