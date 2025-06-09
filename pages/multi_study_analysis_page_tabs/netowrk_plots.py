# pages/multi_study_analysis_page_tabs/network_plots.py
from dash import html, dcc, callback, Input, Output, callback_context, State, no_update
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash.exceptions import PreventUpdate
import os
import pandas as pd
import json
from scipy import stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from collections import Counter
from networkx.algorithms import bipartite
import networkx as nx
import seaborn as sns
import sspa
import numpy as np
import libchebipy
import io
import base64
import matplotlib.pyplot as plt
import glob


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

# Load GMT once
REACTOME_PATHS = sspa.process_gmt(
    infile='Reactome_Homo_sapiens_pathways_ChEBI_R90.gmt'
)
REACTOME_DICT  = sspa.utils.pathwaydf_to_dict(REACTOME_PATHS)
PATHWAY_NAMES  = dict(
    zip(REACTOME_PATHS.index, REACTOME_PATHS['Pathway_name'])
)

def get_pathway_data(obj,
                     gmt_file="Reactome_Homo_sapiens_pathways_ChEBI_R90.gmt"):
    """
    Expects obj.processed_data to be a DataFrame whose columns are:
      [CHEBI:xxx stripped feature columns ..., 'group_type'].
    Populates on obj:
      - reactome_paths, reactome_dict, pathway_names  (cached)
      - pathway_coverage       (dict pathway_id -> # measured features)
      - pathway_scores         (DataFrame of KPCA scores + metadata)
      - pval_df                (DataFrame of P-value, Stat, Direction, FDR_P-value)
      - DA_pathways            (list of human pathway names passing FDR<0.05)
    """
    
    # ---------------------------------------------------------------------
    # 1) Load / cache the Reactome GMT
    # ---------------------------------------------------------------------
    if not hasattr(obj, "reactome_paths"):
        rp = sspa.process_gmt(infile=gmt_file)
        rd = sspa.utils.pathwaydf_to_dict(rp)
        pn = dict(zip(rp.index, rp["Pathway_name"]))
        obj.reactome_paths = rp
        obj.reactome_dict  = rd
        obj.pathway_names  = pn
    else:
        rp, rd, pn = (
            obj.reactome_paths,
            obj.reactome_dict,
            obj.pathway_names
        )

    # ---------------------------------------------------------------------
    # 2) Prepare your data & compute coverage
    # ---------------------------------------------------------------------
    df = obj.processed_data.copy()
    print(df)
    # strip any CHEBI: prefix
    df.columns = df.columns.str.removeprefix("CHEBI:")

    # must have these two cols
    if not {"group_type"}.issubset(df.columns):
        raise KeyError("processed_data must contain 'group_type' columns")

    coverage = {
        pw: len(set(df.columns).intersection(feats))
        for pw, feats in rd.items()
    }
    obj.pathway_coverage = coverage

    # bail if *every* overlapping pathway has exactly one feature
    nonzero_counts = [c for c in coverage.values() if c > 0]
    if (len(nonzero_counts) > 0) and all(c == 1 for c in nonzero_counts):
        obj.pathway_scores = pd.DataFrame(columns=["group_type"])
        obj.pval_df        = pd.DataFrame(
            columns=["P-value","Stat","Direction","FDR_P-value"]
        )
        obj.DA_pathways    = []
        print(f"[{getattr(obj,'node_name','')}] skipping KPCA: single-feature pathways only")
        return

    # ---------------------------------------------------------------------
    # 3) Run KPCA
    # ---------------------------------------------------------------------
    X = df.drop(columns=["group_type"])
    print('start sspa')
    kpca = sspa.sspa_KPCA(rp)
    scores = kpca.fit_transform(X)
    #print(scores)
    print('after sspa')
    # if it came back as a numpy array, wrap it in a DataFrame
    if not isinstance(scores, pd.DataFrame):
        scores = pd.DataFrame(
            scores,
            index=X.index,
            columns=rp.index
        )

    # re‐attach metadata
    scores["group_type"] = df["group_type"]

    # rename to human names
    scores.rename(columns=pn, inplace=True)
    print('after rename')
    obj.pathway_scores = scores

    # 4) Differential testing
    # -----------------------
    # figure out which columns are your KPCA scores (all numeric ones)
    numeric_cols = scores.select_dtypes(include=[np.number]).columns.tolist()

    X_case = scores.loc[scores["group_type"]=="Case",   numeric_cols]
    X_ctrl = scores.loc[scores["group_type"]=="Control", numeric_cols]

    # need ≥2 samples each
    if X_case.shape[0] < 2 or X_ctrl.shape[0] < 2:
        obj.pval_df     = pd.DataFrame(columns=[...])
        obj.DA_pathways = []
        print("…skipping DA: insufficient samples")
        return

    # filter out zero‐variance pathways
    var_case = X_case.var(ddof=1)
    var_ctrl = X_ctrl.var(ddof=1)
    valid    = var_case[(var_case > 0) | (var_ctrl > 0)].index.tolist()
    if not valid:
        obj.pval_df     = pd.DataFrame(columns=[...])
        obj.DA_pathways = []
        print("…no non-zero‐variance pathways")
        return
    print('after filter of valid columns')
    # t-test + BH correction
    stat, pvals = stats.ttest_ind(
        X_case[valid], X_ctrl[valid], nan_policy="omit"
    )
    X_case_valid = X_case[valid]
    fdr = multipletests(pvals, method="fdr_bh")[1]
    print('after stat tests')
    pval_df = pd.DataFrame({
        "P-value":     pvals,
        "Stat":        stat,
        "Direction":   ["Up" if s>0 else "Down" for s in stat],
        "FDR_P-value": fdr
    }, index=X_case_valid.columns).sort_values("FDR_P-value")
    print('1')
    obj.pval_df = pval_df
    obj.DA_pathways = valid  # these are already the human‐readable names
    print('2')
    # finally, the list of *human* names passing FDR < 0.05
    da_ids = pval_df.index[pval_df["FDR_P-value"] < 0.05].tolist()
    print('3')
    obj.DA_pathways = [pn.get(pid, pid) for pid in da_ids]
    print('Done')
    print(f"[{getattr(obj,'node_name','')}] found {len(obj.DA_pathways)} DA pathways")


layout = html.Div([
                    html.H2("Network Graphs of Differential Metabolites and Pathways"),

                    # Placeholder for dynamic background processing description
                    html.Div(
                        id="network-background-div",
                        style={
                            "backgroundColor": "#f0f0f0",
                            "padding": "1rem",
                            "borderRadius": "5px",
                            "marginBottom": "1rem",
                        }
                    ),

                    # --- Options row at the top ---
                    dbc.Row(
                        [
                            # New “Network level” selector
                            dbc.Col([
                                dbc.Label("Network level:"),
                                dcc.Dropdown(
                                    id="network-level-dropdown-msa",
                                    options=[
                                        {"label": "Differential metabolite", "value": "diff-metabolite"},
                                        {"label": "Pathway",   "value": "pathway"},
                                    ],
                                    value="diff-metabolite",
                                    clearable=False,
                                )
                            ], width=3),

                            # Existing layout selector
                            dbc.Col([
                                dbc.Label("Select layout:"),
                                dcc.Dropdown(
                                    id="network-layout-dropdown-msa",
                                    options=[
                                        {"label": "COSE layout",       "value": "cose"},
                                        {"label": "FCOSE layout", "value": "fcose"},
                                        {"label": "COLA layout",     "value": "cola"},
                                        {"label": "Circular layout",     "value": "circular"},
                                        {"label": "Shell layout",        "value": "shell"},
                                        {"label": "Spectral layout",     "value": "spectral"},
                                        {"label": "Random layout",       "value": "random"},
                                    ],
                                    value="cose",
                                    clearable=False,
                                )
                            ], width=3),

                            # Node‐style: will be updated dynamically
                            dbc.Col([
                                dbc.Label("Node style:"),
                                dcc.Dropdown(
                                    id="network-node-style-dropdown-msa",
                                    # initial options (for “metabolite”)
                                    options=[
                                        {"label": "Pie charts",    "value": "pie"},
                                        {"label": "Circle markers","value": "circle"},
                                        {"label": "T statistic","value": "t_statistic"},
                                        {"label": "Bipartite","value": "bipartite"},
                                    ],
                                    value="pie",
                                    clearable=False,
                                )
                            ], width=2),

                            # Min co‐occurrences
                            dbc.Col([
                                dbc.Label("Min. co-occurrences:"),
                                dcc.Input(
                                    id="num-metabolites-network-msa",
                                    type="number",
                                    min=1, max=50, step=1,
                                    value=2,
                                    style={"width": "80px"}
                                )
                            ], width=2),
                        ],
                        align="center",
                        style={"margin": "1rem 0"}
                    ),
                    dbc.Row(
                        [
                            # blank 4‐col spacer
                            dbc.Col(width=4),

                            # centered 4‐col for Refresh
                            dbc.Col(
                                dbc.Button(
                                    "Refresh graphs",
                                    id="refresh-network-button-msa",
                                    color="primary",
                                    n_clicks=0,
                                    size="sm",
                                    type="button",
                                ),
                                width=4,
                                className="text-center",  # center contents
                            ),

                            # right‐aligned 4‐col for Save plot
                            dbc.Col(
                                dbc.Button(
                                    "Save plot",
                                    id="save-plot-button-msa",
                                    n_clicks=0,
                                    type="button",        
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                ),
                                width=4,
                                className="text-end",  # align contents to the right
                            ),
                        ],
                        style={"marginBottom": "1rem"},
                    ),
                    dbc.Modal(
                        [
                            dbc.ModalHeader("Name your studies"),
                            dbc.ModalBody(
                                html.Div([
                                    # 1) Your existing studies picker
                                    dcc.Dropdown(
                                        id="bipartite-study-dropdown",
                                        options=[],
                                        multi=False,
                                        placeholder="Select one or more processed study files"
                                    ),

                                    # 2) New disease chooser / input on one line
                                    html.Div(
                                        [
                                            # Label
                                            html.Span(
                                                "Choose or enter the disease name:",
                                                style={"marginRight": "0.5rem", "fontWeight": "500"}
                                            ),

                                            # Existing diseases dropdown (empty for now)
                                            dcc.Dropdown(
                                                id="bipartite-disease-dropdown",
                                                options=[], 
                                                placeholder="Choose disease",
                                                style={
                                                    "display": "inline-block",
                                                    "verticalAlign": "middle",
                                                    "width": "200px",
                                                    "marginRight": "0.5rem"
                                                }
                                            ),

                                            # Free-text input next to it
                                            dcc.Input(
                                                id="bipartite-disease-input",
                                                type="text",
                                                placeholder="Or type here",
                                                style={
                                                    "display": "inline-block",
                                                    "verticalAlign": "middle",
                                                    "width": "200px"
                                                }
                                            ),
                                        ],
                                        style={"marginTop": "1rem", "whiteSpace": "nowrap"}
                                    ),
                                    # ← new container, hidden by default
                                    html.Div(
                                        id="study-group-details-container",
                                        style={"display": "none", "marginTop": "1.5rem"},
                                        children=[

                                            # Main heading
                                            html.H5("Study group details", style={"textAlign": "center"}),

                                            # Two‐column split
                                            dbc.Row(
                                                [
                                                    # ◄ Left: group types as radio buttons
                                                    dbc.Col(
                                                        [
                                                            html.H6("group types", style={"textAlign": "center"}),
                                                            dcc.RadioItems(
                                                                id="group-types-radio_msa",
                                                                options=[],    # filled by callback
                                                                labelStyle={"display": "block", "padding": "0.25rem"},
                                                            ),
                                                        ],
                                                        width=6,
                                                        style={
                                                            "borderRight": "1px solid #ccc",
                                                            "paddingRight": "1rem"
                                                        },
                                                    ),

                                                    # ► Right: group classes (unique labels) go here
                                                    dbc.Col(
                                                        [
                                                            html.H6("group classes", style={"textAlign": "center"}),
                                                            html.Div(
                                                                id="group-classes-list_msa",
                                                                style={
                                                                    "height": "150px",
                                                                    "overflowY": "auto",
                                                                    "padding": "10px",
                                                                    "border": "1px dashed #ccc"
                                                                },
                                                                children="Select a group type to see its labels"
                                                            ),
                                                        ],
                                                        width=6,
                                                        style={"paddingLeft": "1rem"},
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                    # ← new line at the very bottom of the ModalBody
                                    html.Div(
                                        id="study-control-case-text",
                                        style={"marginTop": "1rem", "fontStyle": "italic", "textAlign": "center"},
                                        children=""  # will be filled by callback
                                    )
                                ])
                            ),
                            dbc.ModalFooter(
                                dbc.Button("OK", id="bipartite-modal-close", className="ml-auto")
                            ),
                        ],
                        id="bipartite-modal",
                        is_open=False,             # start closed
                        size="lg",
                    ),
                    dbc.Modal(
                        [
                            dbc.ModalHeader("Name of plot"),
                            dbc.ModalBody(
                                dcc.Input(
                                    id="plot-name-input-msa",
                                    type="text",
                                    placeholder="Enter plot name",
                                    style={"width": "100%"},
                                )
                            ),
                            dbc.ModalFooter(
                                # Save button replaces Close
                                dbc.Button(
                                    "Save",
                                    id="confirm-save-plot-button-msa",
                                    n_clicks=0,
                                    type="button", 
                                    color="primary",
                                    className="ms-auto",
                                    
                                )
                            ),
                        ],
                        id="save-plot-modal-msa",
                        is_open=False,
                        size="sm",
                    ),
                    # hidden store for path
                    dcc.Store(id="selected-study-store_msa", storage_type="memory"),
                    #dcc.Store(id="modal-state-store"),
                    #html.Div(id="svg-saver-dummy"),
                    #tml.Div(id="save-svg-output-msa", style={"display": "none"}),
                    #html.Div(id="svg-post-output-msa", style={"display":"none"}),
                    #html.Div(id="dummy-output-msa", style={"display": "none"}),
                    #html.Div(id="save-feedback-msa"),
                    # Wrap the content in a dcc.Loading component.
                    # --- Cytoscape graph inside a Loading spinner ---
                    dcc.Loading(
                        id="loading-network-graphs-msa",
                        children=cyto.Cytoscape(
                            id="metabolic-network-cytoscape-msa",
                            elements=[],              # start empty
                            layout={'name':'cose'},   # default layout
                            stylesheet=[],            # default styles
                            style={
                                'width':'100%',
                                'height':'600px',
                                'backgroundColor':'white'
                            },
                            #generateImage={}          # ← seed generateImage up front!
                        )
                    )
                    
                ])

def register_callbacks():
    # Background processing description
    @callback(
        Output("network-background-div", "children"),
        Input("network-level-dropdown-msa", "value"),
        Input("network-node-style-dropdown-msa", "value"),
    )
    def update_background_description(network_level, node_style):
        """
        Returns a small html.Div (grey‐boxed) with different text depending
        on the selected network level and node style.
        """
        lines = []

        if network_level == "pathway":
            # pathway level (pie only)
            lines.append(
                html.H4("Background processing description", style={"marginBottom": "0.5rem"})
            )
            lines.append(
                html.P(
                    "If the dataset uses RefMet IDs (i.e. originates from workbench or is original data), RefMet-to–ChEBI conversion is performed renaming each metabolite column to its corresponding ChEBI ID (dropping any unmapped columns).",
                    style={"marginBottom": "0.5rem"}
                )
            )
            lines.append(
                html.P(
                    "For all datasets, metabolites are mapped to Reactome pathways (file version 90). If two or more metabolites overlap a pathway, it applies single-sample pathway analysis (ssPA) via KPCA to compute an arbitrary score for each pathway in each patient sample. Differential testing is performed (two-tailed t-test with Benjamini–Hochberg FDR correction) on those pathway scores to identify differential pathways (FDR adjusted p-value below 0.05).",
                    style={"marginBottom": "0.5rem"}
                )
            )
            lines.append(
                html.P(
                    "The network plot shows the differential pathways which co-occur in two or more studies (the number of studies which they co-occur are represented by the pie charts)."
                )
            )

        else:
            # diff-metabolite level
            if node_style == "pie":
                lines.append(
                    html.H4("Background processing description", style={"marginBottom": "0.5rem"})
                )
                lines.append(
                    html.P(
                        "For all datasets, differential testing is performed (two-tailed t-test with Benjamini–Hochberg FDR correction) on the metabolite data to identify differential metabolites (FDR adjusted p-value below 0.05).",
                        style={"marginBottom": "0.5rem"}
                    )
                )
                lines.append(
                    html.P(
                        "The network plot shows the differential metabolites which co-occur in two or more studies (the number of studies which they co-occur are represented by the pie charts)."
                    )
                )

            elif node_style == "circle":
                lines.append(
                    html.H4("Background processing description", style={"marginBottom": "0.5rem"})
                )
                lines.append(
                    html.P(
                        "For all datasets, differential testing is performed (two-tailed t-test with Benjamini–Hochberg FDR correction) on the metabolite data to identify differential metabolites (FDR adjusted p-value below 0.05).",
                        style={"marginBottom": "0.5rem"}
                    )
                )
                lines.append(
                    html.P(
                        "The network plot shows the differential metabolites which co-occur in two or more studies (the number of studies which they co-occur are represented by the size of the nodes)."
                    )
                )

            elif node_style == "t_statistic":
                lines.append(
                    html.H4("Background processing description", style={"marginBottom": "0.5rem"})
                )
                lines.append(
                    html.P(
                        "For all datasets, differential testing is performed (two-tailed t-test with Benjamini–Hochberg FDR correction) on the metabolite data to identify differential metabolites (FDR adjusted p-value below 0.05). This test also produces a t-statistic representing the standardized difference in mean metabolite abundance between the case and control group for that metabolite.",
                        style={"marginBottom": "0.5rem"}
                    )
                )
                lines.append(
                    html.P(
                        "The network plot shows the differential metabolites which co-occur in two or more studies (the t-statistic for each study that found that metabolite differential is shown in the bar graph)."
                    )
                )

            else:  # bipartite
                lines.append(
                    html.H4("Background processing description", style={"marginBottom": "0.5rem"})
                )
                lines.append(
                    html.P(
                        "For all datasets, differential testing is performed (two-tailed t-test with Benjamini–Hochberg FDR correction) on the metabolite data to identify differential metabolites (FDR adjusted p-value below 0.05). This test also produces a t-statistic representing the standardized difference in mean metabolite abundance between the case and control group for that metabolite.",
                        style={"marginBottom": "0.5rem"}
                    )
                )
                lines.append(
                    html.P(
                        "The network plot shows the differential metabolites which co-occur in two or more studies (the study that the metabolite is differential in is represented by the edges and the more edges the differential metabolite has the lighter the colour of the node)."
                    )
                )

        return lines


    #########################################
    ##### Controls for network settings #####
    #########################################

    @callback(
        Output("network-node-style-dropdown-msa", "options"),
        Output("network-node-style-dropdown-msa", "value"),
        Input("network-level-dropdown-msa", "value"),
        prevent_initial_call=False
    )
    def update_node_style(level):
        if level == "pathway":
            # Only pie charts allowed for pathways
            opts  = [{"label": "Pie charts", "value": "pie"}]
            value = "pie"
        else:
            # Options for differential metabolites
            opts  = [
                {"label": "Pie charts",    "value": "pie"},
                {"label": "Circle markers","value": "circle"},
                {"label": "T statistic","value": "t_statistic"},
                {"label": "Bipartite","value": "bipartite"},
            ]
            # make sure we don’t auto‐reset to something invalid
            # keep current value if it’s in opts, otherwise default to first
            prev = callback_context.states.get("network-node-style-dropdown-msa.value")
            value = prev if prev in {o["value"] for o in opts} else opts[0]["value"]
        return opts, value

    @callback(
        Output("bipartite-modal", "is_open"),
        [
            Input("network-node-style-dropdown-msa", "value"),
            Input("bipartite-modal-close", "n_clicks"),
        ],
        [State("bipartite-modal", "is_open")],
    )
    def toggle_bipartite_modal(node_style, close_clicks, is_open):
        # if they switch *to* bipartite, open the modal
        if node_style == "bipartite" and not is_open:
            return True
        # if they hit OK (or Close), close it
        if close_clicks:
            return False
        # otherwise, leave it as is
        return is_open

    @callback(
        Output("bipartite-study-dropdown", "options"),
        Input("project-files-checklist-msa", "value"),
    )
    def update_bipartite_studies(selected_files):
        if not selected_files:
            return []
        options = []
        for fname in selected_files:
            parts = fname.split("_")
            # if filename has at least three parts, take the middle one as study name
            if len(parts) >= 3:
                study_name = parts[1]
            else:
                study_name = fname
            options.append({
                "label": study_name,
                "value": fname,        # keep the full filename as the value
            })
        return options

    @callback(
        [ Output("group-types-radio_msa", "options"),
        Output("study-group-details-container", "style") ],
        Input("bipartite-study-dropdown", "value"),
        State("selected-study-store_msa", "data"),
        prevent_initial_call=True
    )
    def show_group_popup(selected_study, stored_study):

        # No study selected → do nothing
        if not selected_study:
            raise PreventUpdate

        # Normalize list → single string
        if isinstance(selected_study, list):
            if not selected_study:
                raise PreventUpdate
            selected_study = selected_study[0]
        print('Selected study')
        study_name = selected_study.split("_")[1] if len(selected_study.split("_")) >= 3 else selected_study
        print(study_name)
        folder = os.path.join(UPLOAD_FOLDER, study_name)
        if not os.path.exists(folder):
            return [], {"display": "none"}

        # Read metadata to decide source
        details       = read_study_details_msa(folder)
        dataset_source = details.get("Dataset Source", "").lower()
        if dataset_source not in ["metabolomics workbench", "metabolights", "original data - refmet ids", "original data - chebi ids"]:
            return [], {"display": "none"}

        # Build group‐type options
        group_options = []
        if dataset_source in (
            "metabolomics workbench",
            "original data - refmet ids",
            "original data - chebi ids",
        ):
            csvs = [f for f in os.listdir(folder) if f.endswith(".csv")]
            if csvs:
                path = os.path.join(folder, csvs[0])
                try:
                    df = pd.read_csv(path)
                    if "Class" in df.columns and not df.empty:
                        first = str(df.iloc[0]["Class"])
                        items = [g.strip() for g in first.split("|") if g.strip()]
                        group_options = [{"label": g, "value": g} for g in items]
                except Exception:
                    pass

        elif dataset_source == "metabolights":
            # 1) build the pattern
            pattern = os.path.join(folder, "s_*.txt")

            # 2) expand the pattern into actual files
            matches = glob.glob(pattern)

            # 3) handle zero or many matches, and pick one
            if not matches:
                raise FileNotFoundError(f"No metadata file found matching {pattern!r}")
            elif len(matches) > 1:
                # you could choose the newest, the first, or raise an error
                matches.sort()  # alphabetical; or sort by os.path.getmtime for newest
            meta_filepath = matches[0]
            #meta_filepath = os.path.join(folder, "s_*.txt")
            if os.path.exists(meta_filepath):
                try:
                    meta_df = pd.read_csv(meta_filepath, sep="\t", encoding="unicode_escape")
                    group_options = [
                        {"label": col, "value": col}
                        for col in meta_df.columns
                        if "Factor Value" in col
                    ]
                except Exception:
                    pass

        # If nothing to show, keep it closed
        if not group_options:
            return [], {"display": "none"}

        # Otherwise open modal, populate radioItems, and reveal the container
        return (
            group_options,
            {"display": "block", "marginTop": "1.5rem"}
        )

    @callback(
        Output("group-classes-list_msa", "children"),
        [
            Input("group-types-radio_msa",          "value"),
            Input("bipartite-study-dropdown", "value"),
        ],
        prevent_initial_call=True
    )
    def populate_group_classes(selected_group, selected_study):
        # Prompt if nothing chosen
        if not selected_group:
            return "Select a group type to see its labels"
        if not selected_study:
            return "No study selected."

        # Normalize list
        if isinstance(selected_study, list):
            if not selected_study:
                return "No study selected."
            selected_study = selected_study[0]

        study_name = selected_study.split("_")[1] if len(selected_study.split("_")) >= 3 else selected_study
        folder = os.path.join(UPLOAD_FOLDER, study_name)
        if not os.path.exists(folder):
            return "Data folder not found."

        details       = read_study_details_msa(folder)
        dataset_source = details.get("Dataset Source", "").lower()

        labels = []
        if dataset_source in (
            "metabolomics workbench",
            "original data - refmet ids",
            "original data - chebi ids",
        ):
            csvs = [f for f in os.listdir(folder) if f.endswith(".csv")]
            if csvs:
                path = os.path.join(folder, csvs[0])
                try:
                    df = pd.read_csv(path)
                    # get the order of types from first row
                    if "Class" in df.columns and not df.empty:
                        order = [g.strip() for g in str(df.iloc[0]["Class"]).split("|")]
                        if selected_group in order:
                            idx = order.index(selected_group)
                            # collect unique labels for that index
                            seen = set()
                            for val in df["Class"]:
                                parts = [p.strip() for p in str(val).split("|")]
                                if len(parts) > idx:
                                    seen.add(parts[idx])
                            labels = sorted(seen)
                except Exception:
                    return "Error reading CSV file."

        elif dataset_source == "metabolights":
            # 1) build the pattern
            pattern = os.path.join(folder, "s_*.txt")

            # 2) expand the pattern into actual files
            matches = glob.glob(pattern)

            # 3) handle zero or many matches, and pick one
            if not matches:
                raise FileNotFoundError(f"No metadata file found matching {pattern!r}")
            elif len(matches) > 1:
                # you could choose the newest, the first, or raise an error
                matches.sort()  # alphabetical; or sort by os.path.getmtime for newest
            meta_filepath = matches[0]
            #meta_filepath = os.path.join(folder, "s_*.txt")
            if os.path.exists(meta_filepath):
                try:
                    meta_df = pd.read_csv(meta_filepath, sep="\t", encoding="unicode_escape")
                    if selected_group in meta_df.columns:
                        labels = sorted(meta_df[selected_group].dropna().unique())
                except Exception:
                    return "Error reading metadata file."
        else:
            return "Unsupported dataset source."

        if not labels:
            return "No labels found for that group type."

        # Render as bullet list
        return html.Ul([html.Li(str(lbl)) for lbl in labels])

    @callback(
        [
            Output("bipartite-disease-dropdown", "options"),
            Output("bipartite-disease-dropdown", "value"),
            Output("bipartite-disease-input", "value"),
            Output("selected-study-store_msa", "data"),
        ],
        Input("bipartite-study-dropdown", "value"),
        [
            State("selected-study-store_msa",   "data"),
            State("bipartite-disease-dropdown", "value"),
            State("bipartite-disease-input",    "value"),
            State("project-dropdown-pop-msa",   "value"),
        ],
        prevent_initial_call=True
    )
    def update_disease_for_study(
        selected_study,
        previous_study,
        previous_dropdown_val,
        previous_input_val,
        selected_project
    ):
        # 1) sanity check
        if not selected_project:
            raise PreventUpdate

        # 2) path to the per‐project file
        project_dir = os.path.join("Projects", selected_project)
        os.makedirs(project_dir, exist_ok=True)
        mapping_file = os.path.join(project_dir, "disease_associations.json")

        # 3) load existing mapping { study_name: disease_name, … }
        try:
            with open(mapping_file, "r") as f:
                mapping = json.load(f)
        except Exception:
            mapping = {}

        # 4) save the disease for the *previous* study
        if previous_study:
            # prefer whatever is selected in the dropdown; otherwise use typed input
            prev = previous_dropdown_val or (previous_input_val.strip() if previous_input_val else None)
            if prev:
                mapping[previous_study] = prev
            else:
                # if they cleared it, remove that key
                mapping.pop(previous_study, None)

            # write the updated mapping back to disk
            with open(mapping_file, "w") as f:
                json.dump(mapping, f, indent=2)

        # 5) build the union of all diseases for the dropdown options
        all_diseases = sorted({d for d in mapping.values() if d})
        options = [{"label": d, "value": d} for d in all_diseases]

        # 6) figure out what (if anything) to pre-select for the *new* study
        selected_value = mapping.get(selected_study, None)

        # 7) clear the free-text input every time we switch
        input_value = ""

        # 8) remember this study as “previous” for next time
        return options, selected_value, input_value, selected_study

    @callback(
        Output("study-control-case-text", "children"),
        Input("bipartite-study-dropdown", "value"),
        State("project-dropdown-pop-msa", "value"),
        prevent_initial_call=True
    )
    def update_control_case_text(selected_study, selected_project):
        # nothing to do until both are chosen
        if not selected_study or not selected_project:
            raise PreventUpdate

        # derive the study key exactly as you do elsewhere
        parts = selected_study.split("_")
        study_key = parts[1] if len(parts) >= 3 else selected_study

        # locate and load the project_details_file.json
        project_dir = os.path.join("Projects", selected_project)
        details_path = os.path.join(project_dir, "project_details_file.json")
        try:
            with open(details_path, "r", encoding="utf-8") as f:
                proj = json.load(f)
        except Exception:
            return ""  # silent failure if file missing or malformed

        # pull out group_filter → Control/Case
        studies = proj.get("studies", {})
        cfg = studies.get(study_key, {}).get("group_filter", {})
        ctrl = cfg.get("Control", [])
        case = cfg.get("Case", [])

        # format into a single line
        ctrl_txt = ", ".join(ctrl) if ctrl else "None"
        case_txt = ", ".join(case) if case else "None"
        return f"Control: {ctrl_txt} | Case: {case_txt}"


    #######################################################################
    #################### Producing network graph ##########################
    #######################################################################

    # Make sure Cytoscape can accept inline images
    cyto.load_extra_layouts()

    @callback(
        Output("metabolic-network-cytoscape-msa", "elements"),
        Output("metabolic-network-cytoscape-msa", "layout"),
        Output("metabolic-network-cytoscape-msa", "stylesheet"),
        [
            # only this button click triggers a refresh
            Input("refresh-network-button-msa", "n_clicks"),
            Input("bipartite-modal-close", "n_clicks"),
        ],
        [
            # everything else becomes State
            State("num-metabolites-network-msa",     "value"),
            State("network-layout-dropdown-msa",     "value"),
            State("network-node-style-dropdown-msa", "value"),
            State("network-level-dropdown-msa",      "value"),
            State("multi-study-analysis-tabs",       "value"),
            State("project-files-checklist-msa",     "value"),
            State("project-dropdown-pop-msa",        "value"),
            State("bipartite-study-dropdown",       "value"),
            State("bipartite-disease-dropdown",     "value"),
            State("bipartite-disease-input",        "value"),
        ]
    )
    def update_metabolic_network(refresh_clicks, close_clicks,
                                min_cooccurring, layout_choice,
                                node_style, network_level, active_tab,
                                selected_files, selected_project, 
                                selected_study, selected_disease, input_disease):
        # figure out which Input fired
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # ─── Handle the bipartite OK click ───────────────────────────────────────────
        #if trigger_id == "bipartite-modal-close" and node_style == "bipartite":
        if close_clicks and node_style == "bipartite":
            print('In the bipartite graph code')
            # sanity checks
            if not (selected_project and selected_files and selected_disease):
                #return html.Div("Select a project, study and disease first.")
                return no_update, no_update, no_update
            
            # 1. load disease associations
            assoc_path = os.path.join('Projects', selected_project, 'disease_associations.json')
            with open(assoc_path, 'r', encoding='utf-8') as f:
                associations = json.load(f)

            # 2. build a list of Analysis instances (only those with DA_metabolites)
            studies = []
            for fname in selected_files:
                csv_path = os.path.join("Projects", selected_project, "processed-datasets", fname)
                if not os.path.exists(csv_path):
                    continue

                df = pd.read_csv(csv_path).set_index("database_identifier")
                class Analysis: pass
                da = Analysis()
                da.processed_data = df
                da.da_testing    = da_testing.__get__(da, Analysis)
                da.pathway_level = False
                da.node_name      = fname.split("_")[1] if len(fname.split("_")) >= 3 else fname

                print('da testing')
                try:
                    da.da_testing()
                except Exception as e:
                    print(f"[da testing] Error in da testing for: {e}")
                    continue
                print('mets get')
                mets = getattr(da, "DA_metabolites", [])
                if not mets:
                    continue
                
                disease = associations.get(fname)

                if not disease:
                    continue

                # store the study’s “node name” and its metabolites
                # store disease on the object
                da.disease         = disease
                da.DA_metabolites  = mets
                studies.append(da)

            # DEBUG: make sure we collected something
            print(f"Collected {len(studies)} studies with DA metabolites")

            # if nothing to plot
            if not studies:
                #return html.Div("No differential metabolites found across selected studies.")
                return no_update, no_update, no_update

            # --- build metabolite co-occurrence counts ----------
            pair_counts = Counter()
            for st in studies:
                items = sorted(set(st.DA_metabolites))
                for u, v in combinations(items, 2):
                    # order the pair so (A,B) == (B,A)
                    pair_counts[tuple(sorted((u, v)))] += 1

            # threshold at least 2 (or whatever user passed)
            threshold = max(min_cooccurring or 2, 2)
            # keep only those pairs whose count >= threshold
            cooc_edges = [(u, v, c) for (u, v), c in pair_counts.items() if c >= threshold]

            # metabolites to keep = every node that appears in any surviving pair
            cooc_mets = set()
            for u, v, _ in cooc_edges:
                cooc_mets.add(u)
                cooc_mets.add(v)

            # 3. Build bipartite graph: bottom = diseases, top = metabolites
            B = nx.Graph()
            # bottom nodes are disease names
            bottom_nodes = [s.disease for s in studies]
            B.add_nodes_from(bottom_nodes, bipartite=1)
            # top nodes are all metabolites (flatten the lists)
            all_mets = [m for s in studies for m in s.DA_metabolites]

            B.add_nodes_from(cooc_mets, bipartite=0)
            # edges connect each disease to any of its DA_metabolites *if* that metabolite survived
            for st in studies:
                for met in st.DA_metabolites:
                    if met in cooc_mets:
                        B.add_edge(st.disease, met)

            # 4. extract the two node‐sets
            bottom_nodes, top_nodes = bipartite.sets(B)

            # 6. compute degrees for styling
            degree_dict = dict(B.degree())
            max_deg = max(degree_dict.values()) if degree_dict else 1

            max_deg_top = (
                max(degree_dict[n] for n in top_nodes)
                if top_nodes else 1
            )

            max_deg_bottom = (
                max(degree_dict[n] for n in bottom_nodes)
                if bottom_nodes else 1
            )

            # 7. build Cytoscape elements exactly as before,
            #    but use bottom_nodes/top_nodes to assign classes
            elements = []
            for node in B.nodes():
                if node in bottom_nodes:
                    cls   = "bottom"
                    label = node
                else:
                    cls   = "top"
                    label = ""
                elements.append({
                    "data": {
                        "id":    node,
                        "label": label,
                        "degree": degree_dict[node]
                    },
                    "classes": cls
                })
            for u, v in B.edges():
                elements.append({"data": {"source": u, "target": v}})

            # 8. your existing stylesheet + Cytoscape call…
            stylesheet = [
                {
                    "selector": ".top",
                    "style": {
                        "label": "",
                        "width": 20, "height": 20,
                        "background-color": 
                            f"mapData(degree, 0, {max_deg_top}, #006d2c, #e5f5e0)"
                    }
                },
                {
                    "selector": ".bottom",
                    "style": {
                        "label": "data(label)",
                        "width": 40, "height": 40,
                        'text-valign':   'center',
                        'text-halign':   'center',
                        "background-color": "#ADD8E6"
                    }
                },
                {
                    "selector": "edge",
                    "style": {"line-color": "#ccc", "width": 1}
                }
            ]

            # --- Stylesheet ---
            layout_map = {
                "cose": "cose", "fcose": "fcose", "COLA": "cola", "circular": "circle",
                "random": "random", "shell": "concentric", "spectral": "grid"
            }
            layout_name = layout_map.get(layout_choice, "cose")

            return elements, {'name': layout_name}, stylesheet

        # ─── Otherwise fall back to your existing “Refresh” behavior ─────────────────
        if trigger_id == "refresh-network-button-msa":
            # 1) never run before any Refresh click
            if not refresh_clicks:
                return no_update, no_update, no_update

            # 2) only run when the network tab is active
            if active_tab != "network-graphs":
                return no_update, no_update, no_update

            # 3) your existing validation & graph‐building code…
            if not selected_project or not selected_files:
                #return html.Div("Please select a project and at least one file.")
                return no_update, no_update, no_update

            # --- Load & analyze studies at the chosen network level ---
            studies = []
            for fname in selected_files:
                path = os.path.join(
                    "Projects", selected_project,
                    "processed-datasets", fname
                )
                if not os.path.exists(path):
                    continue
                
                df = pd.read_csv(path).set_index("database_identifier")
                # create a little analysis container
                class Analysis: pass
                da = Analysis()
                da.processed_data = df
                da.node_name     = fname.split("_")[1] if len(fname.split("_")) >= 3 else fname

                # full set of studies, even those that may later be filtered out
                all_study_names = [
                    fname.split("_")[1] if len(fname.split("_")) >= 3 else fname
                    for fname in selected_files
                ]

                # Build the details file path (unchanged logic).
                folder_details = os.path.join("pre-processed-datasets", da.node_name)
                details = read_study_details_msa(folder_details)
                dataset_source = details.get("Dataset Source", "").lower()
                
                # bind your two methods onto this instance
                da.da_testing      = da_testing.__get__(da, Analysis)
                da.get_pathway_data = get_pathway_data.__get__(da, Analysis)

                # pick which analysis to run
                if network_level == "pathway":
                    try:
                        if dataset_source in (
                            "metabolomics workbench",
                            "original data - refmet ids",
                        ):
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
                            da.processed_data = df

                        get_pathway_data(da)
                        
                    except Exception as e:
                        print(f"[update_metabolic_network] Error computing pathways for {da.node_name}: {e}")
                        continue
                    
                    # keep only if any differential pathways
                    if hasattr(da, "DA_pathways") and len(da.DA_pathways) > 0:
                        studies.append(da)
                        
                else:
                    da.pathway_level = False
                    try:
                        da.da_testing()
                    except Exception:
                        continue

                    # only keep if DA testing produced metabolites
                    if hasattr(da, "DA_metabolites") and len(da.DA_metabolites) > 0:
                        studies.append(da)

            if not studies:
                #return html.Div("No studies with differentially abundant metabolites.")
                return no_update, no_update, no_update

            print("Number of differential pathways per study:")
            for st in studies:
                # For metabolite mode you might still have DA_metabolites
                paths = getattr(st, "DA_pathways", getattr(st, "DA_metabolites", []))
                print(f"  • {st.node_name}: {len(paths)}")
            # --- NEW: drop any pathway that only appears in one study ---
            if network_level == "pathway":
                # 1) count in how many studies each pathway has coverage>0
                pathway_study_counts = Counter()
                for st in studies:
                    for pw, cov in st.pathway_coverage.items():
                        if cov > 0:
                            pathway_study_counts[pw] += 1

                # 2) keep only those seen in 2+ studies
                valid_pathways = {pw for pw, cnt in pathway_study_counts.items() if cnt > 1}

                # 3) prune each study’s coverage dict
                for st in studies:
                    st.pathway_coverage = {
                        pw: cov
                        for pw, cov in st.pathway_coverage.items()
                        if pw in valid_pathways
                    }

            # --- Build co-occurrence graph ---
            pair_counts = Counter()
            for st in studies:
                if network_level == "diff-metabolite":
                    # metabolites that passed your DA test
                    items = sorted(set(st.DA_metabolites))
                else:  # pathway mode
                    # here I take every pathway with at least one covered metabolite
                    #items = sorted(pw for pw, cov in st.pathway_coverage.items() if cov > 0)
                    items = sorted(set(st.DA_pathways))
                    print('da pathways')
                    print(items)

                for u, v in combinations(items, 2):
                    pair_counts[(u, v)] += 1

            threshold = max(min_cooccurring or 2, 2)
            edges = [(u, v, c) for (u, v), c in pair_counts.items() if c >= threshold]
            if not edges:
                label = "diff-metabolite" if network_level == "diff-metabolite" else "pathway"
                #return html.Div(f"No {label} pairs co-occurring in ≥ {threshold} studies.")
                return no_update, no_update, no_update

            G = nx.Graph()
            for u, v, cnt in edges:
                G.add_edge(u, v, weight=cnt)

            # --- Lookup human names (only for metabolites) ---
            chebi_to_name = {}
            for node in G.nodes():
                if network_level == "diff-metabolite":
                    try:
                        chebi_to_name[node] = libchebipy.ChebiEntity(node).get_name()
                    except Exception:
                        chebi_to_name[node] = node
                else:
                    # pathway IDs are already human-readable (or you can map them here)
                    chebi_to_name[node] = node

            # --- Prepare Cytoscape elements ---
            study_names = [st.node_name for st in studies]

            # Precompute palettes
            pie_pal   = sns.color_palette("Set3", n_colors=len(all_study_names)).as_hex()
            color_map = dict(zip(all_study_names, pie_pal))

            # degree = # of edges incident to each node
            deg_dict   = dict(G.degree())
            max_degree = max(deg_dict.values())

            # ——— compute which studies ever show up in a pie ———
            used_studies = set()
            for node in G.nodes():
                if network_level == "diff-metabolite":
                    present = [node in st.DA_metabolites for st in studies]
                else:
                    present = [node in st.DA_pathways for st in studies]
                for nm, ok in zip(all_study_names, present):
                    if ok:
                        used_studies.add(nm)

            # study_counts = in how many studies each node appears
            if network_level == "diff-metabolite":
                study_counts = {
                    node: sum(node in st.DA_metabolites for st in studies)
                    for node in G.nodes()
                }
            else:
                study_counts = {
                    node: sum(node in st.DA_pathways for st in studies)
                    for node in G.nodes()
                }

            max_count = len(studies)

            # build a gradient palette for study_counts
            circle_pal = sns.color_palette("BuPu", n_colors=max_count+1).as_hex()

            # size‐scaling parameters stay the same
            min_size = 30
            max_size = 80

            elements = []
            for node in G.nodes():
                deg     = G.degree(node)
                count   = study_counts[node]
                data    = {
                    'id': node,
                    'label': chebi_to_name[node],
                    'degree': deg,
                    'study_count': count
                }

                if node_style == "pie":
                    # build pie-as-data-URI
                    # decide presence by network level
                    if network_level == "diff-metabolite":
                        present = [node in st.DA_metabolites for st in studies]
                    else:  # pathway
                        # pathway_coverage[node] > 0 means that pathway was hit
                        #present = [st.pathway_coverage.get(node, 0) > 0 for st in studies]
                        present = [node in st.DA_pathways for st in studies]

                    #present = [node in st.DA_metabolites for st in studies]
                    labels  = [nm for nm, ok in zip(all_study_names, present) if ok]
                    sizes   = [1] * len(labels)
                    fig, ax = plt.subplots(figsize=(1,1), dpi=300)
                    fig.patch.set_facecolor('none')
                    ax.set_facecolor('none')
                    ax.pie(
                        sizes,
                        colors=[color_map[l] for l in labels],
                        wedgeprops={'linewidth':0, 'edgecolor':'none','antialiased':False}
                    )
                    ax.set(aspect='equal')
                    ax.axis('off')
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', transparent=True,
                                bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
                    data['pieURI'] = uri


                elif node_style == "t_statistic":
                    # 1) pull out the t-statistics
                    tstats = [
                        st.pval_df['Stat'].get(node, np.nan)
                        for st in studies
                    ]

                    
                    # 2) draw a tiny bar chart
                    fig, ax = plt.subplots(figsize=(1,1), dpi=300)
                    # make entire figure transparent
                    fig.patch.set_facecolor('none')
                    ax.patch.set_facecolor('none')

                    # leave a 5% inset inside the figure so bars don’t butt the border
                    margin = 0.05
                    fig.subplots_adjust(
                        left=margin,
                        right=1 - margin,
                        bottom=margin,
                        top=1 - margin
                    )

                    # plot your bars
                    ax.bar(
                        range(len(study_names)),
                        tstats,
                        color=[color_map[nm] for nm in study_names]
                    ) 

                    
                    # draw the zero‐line
                    ax.axhline(0, color='gray', linewidth=0.5)

                    # remove ticks
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # reserve a slot for every study, even if its t‐stat is NaN
                    ax.set_xlim(-0.5, len(study_names) - 0.5)
                    ax.margins(x=0)

                    # 3) center the zero‐line vertically by choosing symmetric limits
                    max_abs = np.nanmax(np.abs(tstats))
                    padding = 1.05  # 5% headroom
                    ax.set_ylim(-max_abs * padding, max_abs * padding)

                    # 4) move the “bottom” spine to y=0
                    ax.spines['bottom'].set_position(('data', 0))
                    # hide the other spines
                    for spine in ['top','left','right']:
                        ax.spines[spine].set_visible(False)
                    # keep only the bottom spine visible
                    ax.spines['bottom'].set_visible(True)

                    # 5) save without cropping away our empty space
                    buf = io.BytesIO()
                    plt.savefig(
                        buf, format='png', transparent=True,
                        bbox_inches=None,    # <-- don’t auto-tight-crop
                        pad_inches=0         # <-- leave exactly the figure size you asked for
                    )
                    plt.close(fig)
                    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
                    data['barURI'] = uri

                # assign every node the same class
                elements.append({
                    'data': data,
                    'classes': 'node'
                })

            # edges
            for u, v, cnt in edges:
                elements.append({'data': {'source': u, 'target': v, 'weight': cnt}})
            

            # filter your color map down to only those studies
            used_color_map = {
                nm: col
                for nm, col in color_map.items()
                if nm in used_studies
            }

            if node_style == "pie" or node_style == "t_statistic":
                # 1) Legend layout constants
                LEGEND_X        = 900
                LEGEND_Y_START  = 50
                LEGEND_Y_GAP    = 30
                BOX_SIZE        = 20   # size of the color swatch
                FONT_SIZE       = 12
                LABEL_MARGIN    = 8    # gap between box and text

                # 2) Build one "legend-node" per entry
                legend_nodes = []
                y = LEGEND_Y_START

                for name, color in color_map.items():
                    if name not in used_studies:
                        continue   # skip studies that never got a slice

                    legend_nodes.append({
                        "data":     {"id": f"legend-{name}", "label": name},
                        "position": {"x": LEGEND_X, "y": y},
                        "locked":   True,
                        "grabbable": True,
                        "classes":  "legend-node",
                        "style": {
                            "background-color": color,   # from the full map
                            "shape":            "rectangle",
                            "width":            BOX_SIZE,
                            "height":           BOX_SIZE
                        }
                    })
                    y += LEGEND_Y_GAP

                elements += legend_nodes

            # --- Stylesheet ---
            layout_map = {
                "cose": "cose", "fcose": "fcose", "circular": "circle", "COLA": "cola",
                "random": "random", "shell": "concentric", "spectral": "grid"
            }
            layout_name = layout_map.get(layout_choice, "cose")

            stylesheet = [
                # reset edges
                {
                    'selector': 'edge',
                    'style': {
                        'width':       'mapData(weight, {}, {}, 1, 6)'.format(threshold, max(pair_counts.values())),
                        'line-color': '#ccc', 'curve-style': 'bezier'
                    }
                },
                # base node label/size
                {
                    'selector': '.node',
                    'style': {
                        'label':         'data(label)',
                        'text-valign':   'center',
                        'text-halign':   'center',
                        'font-size':     '12px',
                        'border-width':       '2px',    # optional: give it a border
                        'border-color':       '#fff',   # optional: white border
                        'width':
                            f"mapData(degree, 0, {max_degree}, {min_size}, {max_size})",
                        'height':
                            f"mapData(degree, 0, {max_degree}, {min_size}, {max_size})"
                    }
                }
            ]
            if node_style == "pie" or node_style == "t_statistic":
                # 4) Add a single stylesheet rule
                stylesheet += [
                    {
                    "selector": ".legend-node",
                    "style": {
                        "label":         "data(label)",    # show the name
                        "text-valign":   "center",         # vertically centered on the box
                        "text-halign":   "right",          # text to the right of the node
                        "text-margin-x": LABEL_MARGIN,     # small gap
                        "font-size":     f"{FONT_SIZE}px",
                        "color":         "#000"
                    }
                    }
                ]

            if node_style == "pie":
                stylesheet.append({
                    'selector': '.node',
                    'style': {
                        'background-image':  'data(pieURI)',
                        'background-fit':    'none',
                        'background-width': '200px',
                        'background-height':'200px',
                        'background-clip':   'node',
                        'shape':            'ellipse'
                    }
                })
            elif node_style == "t_statistic":
                stylesheet.append({
                    'selector': '.node',
                    'style': {
                        # node size driven by degree → min_size/max_size as before
                        'width':  f"mapData(degree, 0, {max_degree}, {min_size}, {max_size})",
                        'height': f"mapData(degree, 0, {max_degree}, {min_size}, {max_size})",

                        # white background behind the chart
                        'background-color':  'white',
                        'background-opacity': 1,

                        # chart image comes from your data URI
                        'background-image':  'data(barURI)',

                        # scale & center the image to the node
                        'background-fit':     'contain',
                        'background-position':'center center',
                        'background-repeat':  'no-repeat',

                        # clip the image to the node shape
                        'background-clip':    'node',
                        'shape':              'round-rectangle',

                        # *** thin grey border ***
                        'border-width':      '1px',
                        'border-color':      '#888',      # or 'grey' / '#ccc'
                        'border-opacity':    1
                    }
                })
            else:  # circle
                stylesheet.append({
                    'selector': '.node',
                    'style': {
                        'background-image':  'none',
                        'background-color':
                            f"mapData(study_count, 0, {max_count}, {circle_pal[0]}, {circle_pal[-1]})",
                        'width':
                            f"mapData(degree, 0, {max_degree}, {min_size}, {max_size})",
                        'height':
                            f"mapData(degree, 0, {max_degree}, {min_size}, {max_size})",
                        'shape': 'ellipse'
                    }
                })

            return elements, {'name': layout_name}, stylesheet
            
    # 4) Callback to open/close the modal
    @callback(
        Output("save-plot-modal-msa", "is_open"),
        [
            Input("save-plot-button-msa", "n_clicks"),
            Input("confirm-save-plot-button-msa", "n_clicks"),
        ],
        [State("save-plot-modal-msa", "is_open")]
    )
    def toggle_modal(open_clicks, save_clicks, is_open):
        ctx = callback_context.triggered
        if not ctx:
            return is_open
        trigger_id = ctx[0]["prop_id"].split(".")[0]
        if trigger_id in ["save-plot-button-msa", "confirm-save-plot-button-msa"]:
            return not is_open
        return is_open

    # 5) Callback to trigger the client‐side SVG download
    """ @callback(
        Output("metabolic-network-cytoscape-msa", "generateImage"),
        Input("confirm-save-plot-button-msa", "n_clicks"),
        State("project-dropdown-pop-msa",       "value"),
        State("plot-name-input-msa",            "value"),
        State("network-level-dropdown-msa",     "value"),
        prevent_initial_call=True
    )
    def download_svg(n_clicks, project_name, plot_name, network_level):
        # 1) slugify inputs
        proj = (project_name or "project").strip().replace(" ", "-")
        lvl  = (network_level or "network").strip().replace(" ", "-")
        base = (plot_name      or "network").strip().replace(" ", "-")

        # 2) build filename: proj_<level>__<base>
        filename = f"{proj}_{lvl}__{base}"

        return {
            "type":     "svg",
            "action":   "download",
            "filename": filename
        } """

    @callback(
        Output("metabolic-network-cytoscape-msa", "generateImage"),
        Input("confirm-save-plot-button-msa", "n_clicks"),
        State("project-dropdown-pop-msa",       "value"),
        State("plot-name-input-msa",            "value"),
        State("network-level-dropdown-msa",     "value"),
        prevent_initial_call=True
    )
    def trigger_store_png(n_clicks, project_name, plot_name, network_level):
        # slugify inputs
        proj = (project_name or "project").strip().replace(" ", "-")
        lvl  = (network_level or "network").strip().replace(" ", "-")
        base = (plot_name      or "network").strip().replace(" ", "-")
        filename = f"{proj}_{lvl}__{base}"

        return {
            "type":     "png",       # ask for PNG
            "action":   "store",     # store into imageData
            "filename": filename,    # your naming convention
            "options": {
                "bg":   "#ffffff",   # ensure white background
                "full": True,        # include entire graph area
                "scale": 1           # default scale
            }
        }


    # map network levels to folder names
    LEVEL_TO_FOLDER = {
        "diff-metabolite": "Differential-metabolites-network-plots",
        "pathway":         "Differential-pathway-network-plots",
    }

    from werkzeug.utils import secure_filename
    import re
    PROJECTS_ROOT = os.path.abspath(os.path.expanduser(r"C:\Users\Eloisa\Documents\ICL\Tim RA Project - Postgraduate\my_dash_app\Projects"))

    @callback(
    Output("svg-store", "data"),
    Input("metabolic-network-cytoscape-msa", "imageData"),
    State("project-dropdown-pop-msa",   "value"),
    State("network-level-dropdown-msa", "value"),
    prevent_initial_call=True
    )
    def save_svg_to_server(image_data_uri, project_name, network_level):
        if not image_data_uri:
            return no_update

        # 1) parse the incoming data URI
        header, b64 = image_data_uri.split(",", 1)
        svg_bytes = base64.b64decode(b64)

        # 2) slugify project & level
        proj = (project_name or "project").strip().replace(" ", "-")
        lvl  = (network_level or "network").strip().replace(" ", "-")

        # 3) extract the original filename from the header
        #    Dash-Cytoscape puts it into header: 'data:image/svg+xml;filename=proj_lvl__base.svg;base64'
        m = re.search(r'filename=([^;]+)', header)
        filename = secure_filename(m.group(1)) if m else f"{proj}_{lvl}__network.png"

        # 4) pick your folder
        folder_name = LEVEL_TO_FOLDER.get(lvl, LEVEL_TO_FOLDER["metabolite"])
        out_dir = os.path.join(
            PROJECTS_ROOT,
            proj,
            "Plots",
            "Multi-study-analysis",
            folder_name
        )
        os.makedirs(out_dir, exist_ok=True)

        # 5) write it out
        dest = os.path.join(out_dir, filename)
        with open(dest, "wb") as f:
            f.write(svg_bytes)

        # Optionally return something into the Store so you know it ran:
        return {"status": "saved", "path": dest}
    

    """ # Trigger a store‐PNG
    @callback(
        Output("metabolic-network-cytoscape-msa", "generateImage"),
        Input("confirm-save-plot-button-msa","n_clicks"),
        State("plot-name-input-msa","value"),
        prevent_initial_call=True
    )
    def trigger_store_png(n, name):
        return {"type":"png","action":"store","filename":(name or "network").strip()}


    # Catch & write the blob
    @callback(
        Output("save-feedback-msa", "children"),
        Input("metabolic-network-cytoscape-msa","imageData"),
        State("plot-name-input-msa","value"),
        prevent_initial_call=True
    )
    def save_network_png(imageData, name):
        if not imageData:
            return no_update
        header,b64 = imageData.split(",",1)
        data = base64.b64decode(b64)
        folder = os.path.join(os.getcwd(),"assets","networks")
        os.makedirs(folder,exist_ok=True)
        fname = (name or "network").strip()+".png"
        path = os.path.join(folder,fname)
        with open(path,"wb") as f:
            f.write(data)
        return dbc.Alert(f"Saved network as `{fname}` in `/assets/networks`", color="success")

    """