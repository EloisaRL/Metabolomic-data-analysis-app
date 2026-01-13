# pages/multi_study_analysis_page_tabs/network_plots.py
from .shared_functions.data_processing import da_testing
from .shared_functions.helper import (read_study_details_msa,
                                      _safe_chebi_name)
from dash import html, dcc, callback, Input, Output, callback_context, State, no_update, dash_table
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash.exceptions import PreventUpdate
import os, re, gzip, csv
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
import io
import base64
import matplotlib.pyplot as plt
import glob
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

import sqlite3
import requests

CACHE_DIR = Path("assets/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = CACHE_DIR / "chebi_name_map.db"


UPLOAD_FOLDER = "pre-processed-datasets"

refmet = pd.read_csv("refmet.csv", dtype=object)
refmet.columns = refmet.columns.str.strip() 
refmet2chebi = dict(zip(refmet['refmet_name'], refmet['chebi_id']))

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
    kpca = sspa.sspa_KPCA(rp)
    scores = kpca.fit_transform(X)
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
    # t-test + BH correction
    stat, pvals = stats.ttest_ind(
        X_case[valid], X_ctrl[valid], nan_policy="omit"
    )
    X_case_valid = X_case[valid]
    fdr = multipletests(pvals, method="fdr_bh")[1]
    pval_df = pd.DataFrame({
        "P-value":     pvals,
        "Stat":        stat,
        "Direction":   ["Up" if s>0 else "Down" for s in stat],
        "FDR_P-value": fdr
    }, index=X_case_valid.columns).sort_values("FDR_P-value")
    obj.pval_df = pval_df
    obj.DA_pathways = valid  # these are already the human‐readable names
    # finally, the list of *human* names passing FDR < 0.05
    da_ids = pval_df.index[pval_df["FDR_P-value"] < 0.05].tolist()
    obj.DA_pathways = [pn.get(pid, pid) for pid in da_ids]
    #print(f"[{getattr(obj,'node_name','')}] found {len(obj.DA_pathways)} DA pathways")

# =============================== #
# Layout of the Network plots tab #
# =============================== #

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

                            # Refresh button (hiddern by default)
                            dbc.Col(
                                dbc.Button(
                                    "Refresh Table",
                                    id="bipartite-disease-reload",
                                    color="primary",
                                    outline=True, 
                                    size="sm",
                                    # hidden by default; we'll show it when node_style == "bipartite"
                                    style={"marginTop": "1.5rem", "marginLeft": "auto", "display": "none"},
                                ),
                                width="auto",
                            ),
                        ],
                        align="center",
                        style={"margin": "1rem 0"}
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "space-between",
                            "width": "100%",
                        },
                        children=[
                            # Table container (expands to left side)
                            html.Div(
                                id="bipartite-disease-table-container",
                                style={"flexGrow": 1, "marginRight": "1rem"},
                            ),
                            # Save button (on the right)
                            dbc.Button(
                                "Save group types",
                                id="bipartite-save-group-types",
                                color="primary",
                                outline=True, 
                                style={"display": "none"},  # hidden by default
                            ),
                        ],
                    ),
                    html.Div(id="bipartite-save-status", className="text-muted", style={"fontSize": "0.9rem"}),
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
                            dbc.ModalHeader("Name of plot"),
                            dbc.ModalBody(
                                [
                                    dbc.Alert(
                                        # add a warning emoji manually instead of icon=True
                                        [html.Span("⚠️", className="me-2"),
                                        "Note: Your plot will be saved to your Downloads folder."],
                                        color="warning",
                                        className="mb-3",
                                    ),
                                    dcc.Input(
                                        id="plot-name-input-msa",
                                        type="text",
                                        placeholder="Enter plot name",
                                        style={"width": "100%"},
                                    ),
                                ]
                            ),
                            dbc.ModalFooter(
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
                    # Network graph warning
                    html.Div(id="network-warning", style={"margin": "0.75rem 0"}),
                    # Wrap the content in a dcc.Loading component.
                    # --- Cytoscape graph inside a Loading spinner ---
                    html.Div([
                        dcc.Loading(
                            id="loading-network-graphs-msa",
                            children=html.Div([
                                # Cytoscape Graph
                                cyto.Cytoscape(
                                    id="metabolic-network-cytoscape-msa",
                                    elements=[],
                                    layout={'name': 'cose'},
                                    stylesheet=[],
                                    style={
                                        'width': '100%',
                                        'height': '600px',
                                        'backgroundColor': 'white'
                                    }
                                ),

                                # Conversion Table (Shown Below the Graph)
                                dash_table.DataTable(
                                    id='refmet-conversion-table',
                                    columns=[
                                        {"name": "Study",           "id": "study_name"},
                                        {"name": "Total RefMet",    "id": "total_refmet"},
                                        {"name": "Mapped",          "id": "num_mapped"},
                                        {"name": "Unmapped",        "id": "num_unmapped"},
                                        {"name": "% Unmapped",      "id": "pct_unmapped"},
                                    ],
                                    data=[],
                                    style_cell={'textAlign': 'center'},
                                    style_table={'marginTop': '20px', 'overflowX': 'auto'},
                                ),

                                # Pathway Coverage Table (Only populated if node level == 'pathway')
                                dash_table.DataTable(
                                    id='pathway-coverage-table',
                                    columns=[
                                        {"name": "Study", "id": "study_name"},
                                        {"name": "Total Metabolites", "id": "total_metabolites"},
                                        {"name": "Mapped to Pathways", "id": "mapped_to_pathways"},
                                        {"name": "Not Mapped", "id": "not_mapped"},
                                        {"name": "% Mapped", "id": "pct_mapped"},
                                        {"name": "# Pathways Covered", "id": "num_pathways_covered"},
                                    ],
                                    data=[],
                                    style_cell={'textAlign': 'center'},
                                    style_table={'marginTop': '30px', 'overflowX': 'auto'},
                                )
                            ])
                        )
                    ])
                    
                ])



def register_callbacks():
    # Callback which controls the background description shown
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
                    "For all datasets, ChEBI ids are mapped to Reactome pathways (file version 90). If two or more metabolites overlap a pathway, it applies single-sample pathway analysis (ssPA) via KPCA to compute an arbitrary score for each pathway in each patient sample. Differential testing is performed (two-tailed t-test with Benjamini–Hochberg FDR correction) on those pathway scores to identify differential pathways (FDR adjusted p-value below 0.05).",
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
                        "For each dataset, differential testing is performed (two-tailed t-test with Benjamini–Hochberg FDR correction) "
                        "to identify metabolites that are significantly different (FDR-adjusted p < 0.05). Identified ChEBI IDs are then "
                        "converted into metabolite names using the ChEBI 2.0 API.",
                        style={"marginBottom": "0.5rem"}
                    )
                )
                lines.append(
                    html.P(
                        "The network plot displays these differential metabolites as nodes. An edge connects two metabolites if they "
                        "are found to be differential in the same study, with the edge weight reflecting the number of studies where "
                        "this co-occurrence happens. Node size is proportional to how many different metabolites a given metabolite "
                        "co-occurs with (its connectivity in the network), highlighting metabolites that act as 'hubs' across studies. "
                        "The pie charts on the nodes indicate in which studies each metabolite is differential."
                    )
                )
                lines.append(
                    html.Div(
                                [
                                    html.B("Important note"),
                                    html.P(
                                        [
                                            "For all studies ChEBI ids are converted in metabolite names using the "
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
                                        "Advanced users can consult the app log file to see when the ids are being converted.",
                                        style={"marginBottom": 0}
                                    ),
                                ],
                                style={
                                    "backgroundColor": "#ffffff",
                                    "borderLeft": "4px solid #0074D9",
                                    "padding": "0.75rem 1rem",
                                    "borderRadius": "3px",
                                },
                            )
                )

            elif node_style == "circle":
                lines.append(
                    html.H4("Background processing description", style={"marginBottom": "0.5rem"})
                )
                lines.append(
                    html.P(
                        "For all datasets, differential testing is performed (two-tailed t-test with Benjamini–Hochberg FDR correction) on the metabolite data to identify differential metabolites (FDR adjusted p-value below 0.05). Then ChEBI ids are converted into Metabolite names using the ChEBI 2.0 API, prior to creating network plot.",
                        style={"marginBottom": "0.5rem"}
                    )
                )
                lines.append(
                    html.P(
                        "The network plot shows the differential metabolites which co-occur in two or more studies (the number of studies which they co-occur are represented by the size of the nodes)."
                    )
                )
                lines.append(
                    html.Div(
                                [
                                    html.B("Important note"),
                                    html.P(
                                        [
                                            "For all studies ChEBI ids are converted in metabolite names using the "
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
                                        "Advanced users can consult the app log file to see when the ids are being converted.",
                                        style={"marginBottom": 0}
                                    ),
                                ],
                                style={
                                    "backgroundColor": "#ffffff",
                                    "borderLeft": "4px solid #0074D9",
                                    "padding": "0.75rem 1rem",
                                    "borderRadius": "3px",
                                },
                            )
                )

            elif node_style == "t_statistic":
                lines.append(
                    html.H4("Background processing description", style={"marginBottom": "0.5rem"})
                )
                lines.append(
                    html.P(
                        "For each dataset, differential testing is performed (two-tailed t-test with Benjamini–Hochberg FDR correction) "
                        "to identify metabolites that are significantly different (FDR-adjusted p < 0.05). This test also produces a "
                        "t-statistic, which reflects the standardized difference in mean metabolite abundance between the case and control groups. "
                        "Identified ChEBI IDs are then converted into metabolite names using the ChEBI 2.0 API.",
                        style={"marginBottom": "0.5rem"}
                    )
                )
                lines.append(
                    html.P(
                        "The network plot displays these differential metabolites as nodes. An edge connects two metabolites if they "
                        "are found to be differential in the same study, with the edge weight reflecting the number of studies where "
                        "this co-occurrence occurs. Node size is proportional to how many other metabolites a given metabolite co-occurs "
                        "with (its connectivity in the network), highlighting metabolites that act as 'hubs' across studies. "
                        "Within each node, a bar chart shows the t-statistics for that metabolite across the studies in which it was "
                        "differential, with bar colour indicating the study. This allows comparison of both the direction and magnitude "
                        "of differential abundance across datasets."
                    )
                )
                lines.append(
                    html.Div(
                                [
                                    html.B("Important note"),
                                    html.P(
                                        [
                                            "For all studies ChEBI ids are converted in metabolite names using the "
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
                                        "Advanced users can consult the app log file to see when the ids are being converted.",
                                        style={"marginBottom": 0}
                                    ),
                                ],
                                style={
                                    "backgroundColor": "#ffffff",
                                    "borderLeft": "4px solid #0074D9",
                                    "padding": "0.75rem 1rem",
                                    "borderRadius": "3px",
                                },
                            )
                )

            else:  # bipartite
                lines.append(
                    html.H4("Background processing description", style={"marginBottom": "0.5rem"})
                )
                lines.append(
                    html.P(
                        "For all datasets, differential testing is performed (two-tailed t-test with Benjamini–Hochberg FDR correction) on the metabolite data to identify differential metabolites (FDR adjusted p-value below 0.05). This test also produces a t-statistic representing the standardized difference in mean metabolite abundance between the case and control group for that metabolite. Then ChEBI ids are converted into Metabolite names using the ChEBI 2.0 API, prior to creating network plot.",
                        style={"marginBottom": "0.5rem"}
                    )
                )
                lines.append(
                    html.P(
                        "The network plot shows the differential metabolites which co-occur in two or more studies (the study that the metabolite is differential in is represented by the edges and the more edges the differential metabolite has the lighter the colour of the node).",
                        style={"marginBottom": "0.5rem"}
                    )
                )
                lines.append(
                    html.P(
                        "If a selected study does not have a disease type added, it will not be included in the graph."
                    )
                )
                lines.append(
                    html.Div(
                                [
                                    html.B("Important note"),
                                    html.P(
                                        [
                                            "For all studies ChEBI ids are converted in metabolite names using the "
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
                                        "Advanced users can consult the app log file to see when the ids are being converted.",
                                        style={"marginBottom": 0}
                                    ),
                                ],
                                style={
                                    "backgroundColor": "#ffffff",
                                    "borderLeft": "4px solid #0074D9",
                                    "padding": "0.75rem 1rem",
                                    "borderRadius": "3px",
                                },
                            )
                )

        return lines


    #########################################
    ##### Controls for network settings #####
    #########################################
    # Callback which controls the node style settings based on the whether the network level is differential metabolites or pathways
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

    # callback to toggle visibility
    @callback(
        Output("bipartite-disease-reload", "style"),
        Input("network-node-style-dropdown-msa", "value"),
    )
    def toggle_refresh_visibility(node_style):
        base = {"marginTop": "1.5rem", "marginLeft": "auto"}
        return base if node_style == "bipartite" else {**base, "display": "none"}
    
    # callback to toggle visibility
    @callback(
        Output("bipartite-save-group-types", "style"),
        Input("network-node-style-dropdown-msa", "value"),
    )
    def toggle_save_button(node_style):
        base = {"marginTop": "0.5rem"}
        return base if node_style == "bipartite" else {**base, "display": "none"}
    
    @callback(
        Output("bipartite-disease-table-container", "children"),
        Input("network-node-style-dropdown-msa", "value"),
        Input("bipartite-disease-reload", "n_clicks"),
        Input("project-dropdown-pop-msa", "value"),
        Input("bipartite-save-group-types", "n_clicks"),  
        State("project-files-checklist-msa", "value"),
    )
    def render_bipartite_table(node_style, n_clicks_reload, selected_project, n_clicks_save, selected_files):
        if node_style != "bipartite" or not selected_files or not selected_project:
            return None

        # Load study metadata
        project_details_path = os.path.join("Projects", selected_project, "project_details_file.json")
        try:
            with open(project_details_path, "r", encoding="utf-8") as f:
                payload = json.load(f).get("studies", {})
        except Exception:
            payload = {}

        # ---------- Load previously saved disease associations (if any) ----------
        project_dir  = os.path.join("Projects", selected_project)
        mapping_file = os.path.join(project_dir, "disease_associations.json")
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                saved_mapping = json.load(f)  # { "<study>": "<disease_type>" }
            if not isinstance(saved_mapping, dict):
                saved_mapping = {}
        except Exception:
            saved_mapping = {}

        # Build rows + tooltips
        seen = set()
        rows = []
        tooltips = []

        for fname in selected_files:
            parts = fname.split("_")
            study_name = parts[1] if len(parts) >= 3 else fname
            if study_name in seen:
                continue
            seen.add(study_name)

            info = payload.get(study_name, {}) if study_name else {}
            gf = info.get("group_filter", {}) or {}
            control = gf.get("Control") or []
            case    = gf.get("Case")    or [] or {}


            # normalise to list -> comma-separated
            if isinstance(control, str): control = [control]
            if isinstance(case, str):    case    = [case]

            control_txt = ", ".join([str(x) for x in control]) or "N/A"
            case_txt    = ", ".join([str(x) for x in case])    or "N/A"

            # 👇 prefill disease_type from saved JSON (empty string if not present)
            rows.append({
                "study": study_name,
                "disease_type": (saved_mapping.get(study_name) or "").strip(),
            })
            tooltips.append({
                "study": f"𝗖𝗼𝗻𝘁𝗿𝗼𝗹: {control_txt}\n𝗖𝗮𝘀𝗲: {case_txt}",
                "disease_type": ""
            })

        table = dash_table.DataTable(
            id="bipartite-disease-table",
            columns=[
                {"name": "Study", "id": "study", "editable": False},
                {"name": "Disease type", "id": "disease_type", "editable": True},
            ],
            data=rows,
            tooltip_data=tooltips,      # 👈 hover text
            tooltip_duration=None,      # keep visible while hovering
            editable=True,
            # 👇 style the disease_type column so it always looks like a text input
            style_data_conditional=[
                {
                    "if": {"column_id": "disease_type"},
                    "backgroundColor": "white",
                    "border": "1px solid #ccc",
                    "textAlign": "left",
                    "padding": "4px",
                },
                {
                    "if": {"column_id": "study"},
                    "cursor": "help",
                },
            ],
            row_deletable=False,
            cell_selectable=True,
            style_table={"overflowX": "auto"},
            style_cell={"padding": "6px"},
            style_header={"fontWeight": "600"},
            page_action="none",
            persistence=True,
            persisted_props=["data"],
            persistence_type="session", 
            # 👇 force tooltip to keep newlines
            css=[{
                "selector": ".dash-table-tooltip",
                "rule": "white-space: pre-line;"
            }],
        )

        return dbc.Card(
            dbc.CardBody([
                html.H6("Assign disease type per study", className="mb-2"),
                html.Small(
                    "💡 After typing, press Enter to confirm the value in the cell. Then click the save button before creating the graph.",
                    className="text-muted d-block mb-2"
                ),
                html.Div(table),
            ]),
            className="mb-2"
        )
    
    @callback(
        Output("bipartite-save-status", "children"),
        Input("bipartite-save-group-types", "n_clicks"),
        State("bipartite-disease-table", "data"),
        State("project-dropdown-pop-msa", "value"),
        prevent_initial_call=True,
    )
    def save_disease_types(n_clicks, table_data, selected_project):
        if not n_clicks:
            raise PreventUpdate
        if not selected_project or not table_data:
            return "Nothing to save."

        # Ensure project dir + file path
        project_dir = os.path.join("Projects", selected_project)
        os.makedirs(project_dir, exist_ok=True)
        mapping_file = os.path.join(project_dir, "disease_associations.json")

        # Load existing mapping (if any)
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            if not isinstance(mapping, dict):
                mapping = {}
        except Exception:
            mapping = {}

        # Update mapping from table rows (skip blanks, overwrite existing)
        saved = 0
        for row in table_data:
            study = (row.get("study") or "").strip()
            disease = (row.get("disease_type") or "").strip()
            if study and disease:
                mapping[study] = disease
                saved += 1

        # Write back
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

        return f"Saved {saved} assignment{'s' if saved != 1 else ''} to {mapping_file}."



    #######################################################################
    #################### Producing network graph ##########################
    #######################################################################
    # Make sure Cytoscape can accept inline images
    cyto.load_extra_layouts()
    # Callback that produces the network graph
    @callback(
        Output("network-warning", "children"),  
        Output("metabolic-network-cytoscape-msa", "elements"),
        Output("metabolic-network-cytoscape-msa", "layout"),
        Output("metabolic-network-cytoscape-msa", "stylesheet"),
        [
            # only this button click triggers a refresh
            Input("refresh-network-button-msa", "n_clicks"),
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
        ]
    )
    def update_metabolic_network(refresh_clicks,
                                min_cooccurring, layout_choice,
                                node_style, network_level, active_tab,
                                selected_files, selected_project):
        # figure out which Input fired
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        elements = []
        stylesheet = []
        used_studies = set()

        # ─── Handle the bipartite OK click ───────────────────────────────────────────
        if refresh_clicks and node_style == "bipartite":
            # sanity checks
            if not (selected_project and selected_files):
                #return html.Div("Select a project, study and disease first.")
                return no_update, no_update, no_update, no_update
            
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
                study_name = fname.split("_")[1] if len(fname.split("_")) >= 3 else fname
                df = pd.read_csv(csv_path).set_index("database_identifier")
                class Analysis: pass
                da = Analysis()
                da.da_testing    = da_testing.__get__(da, Analysis)
                da.pathway_level = False
                da.node_name      = study_name

                folder_details = os.path.join("pre-processed-datasets", da.node_name)
                details = read_study_details_msa(folder_details)
                dataset_source = details.get("Dataset Source", "").lower()

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
                else:
                    da.processed_data = df

                try:
                    da.da_testing()
                except Exception:
                    logger.exception(f"Network plots tab - Error in da testing for {fname}")
                    continue
                mets = getattr(da, "DA_metabolites", [])
                if not mets:
                    continue
                
                disease = associations.get(study_name)

                if not disease:
                    continue

                # store the study’s “node name” and its metabolites
                # store disease on the object
                da.disease         = disease
                da.DA_metabolites  = mets
                studies.append(da)

            # if nothing to plot
            if not studies:
                logger.error("Network plots tab - No differential metabolites found across selected studies.")
                #return no_update, no_update, no_update
                return (
                    dbc.Alert("No differential metabolites found across selected studies."),
                    [],  {"name": "preset"}, [], 
                )

            # ---- 1) Count, per metabolite, in how many studies it appears (unique per study) ----
            met_counts = Counter()
            for st in studies:
                met_counts.update(set(st.DA_metabolites))   # set() so a metabolite counts at most once per study

            # ---- 2) Apply study-occurrence threshold to metabolites (never below 2) ----
            threshold = max((min_cooccurring or 2), 2)
            allowed_mets = {m for m, c in met_counts.items() if c >= threshold}

            if not allowed_mets:
                label = "differential metabolites"
                logger.error(f"Bipartite graph – No {label} meet the ≥{threshold} studies threshold.")
                return (
                    dbc.Alert(f"Bipartite graph – No {label} meet the ≥{threshold} studies threshold."),
                    [],  {"name": "preset"}, [], 
                )

            # ---- 3) Build bipartite graph: bottom = diseases, top = (filtered) metabolites ----
            B = nx.Graph()

            # unique disease nodes (in case multiple studies share the same disease label)
            diseases = sorted({s.disease for s in studies})
            B.add_nodes_from(diseases, bipartite=1)

            # add only metabolites that passed the study-occurrence filter
            B.add_nodes_from(sorted(allowed_mets), bipartite=0)

            # connect each disease to the allowed metabolites reported by any of its studies
            # If you want edge weights = # of studies (disease, metabolite) co-occur in, track and set 'weight'.
            edge_weights = Counter()
            for st in studies:
                dis = st.disease
                for met in set(st.DA_metabolites):  # set() to avoid duplicate edges from the same study
                    if met in allowed_mets:
                        edge_weights[(dis, met)] += 1

            # add edges (with weight attribute if you want to style by thickness/opacity)
            for (dis, met), w in edge_weights.items():
                B.add_edge(dis, met, weight=w)

            # ---- 4) Drop isolates (diseases with no passing metabolites; very rare but tidy) ----
            isolates = list(nx.isolates(B))
            if isolates:
                B.remove_nodes_from(isolates)

            # ---- 5) (Optional) compute degrees for styling (node size = connectivity in bipartite graph)
            degree_dict = dict(B.degree())
            max_deg = max(degree_dict.values()) if degree_dict else 1
            
            # Recompute partitions after any isolate removals
            #bottom_nodes, top_nodes = bipartite.sets(B)  # bottom = diseases, top = metabolites
            bottom_nodes = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 1}  # diseases
            top_nodes    = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 0}  # metabolites

            # Degrees (for top color mapping)
            degree_dict = dict(B.degree())

            # Max degree for TOP partition (use at least 1 to avoid mapData(0,0,...))
            max_deg_top = max((degree_dict.get(n, 0) for n in top_nodes), default=0)
            if max_deg_top == 0:
                max_deg_top = 1  # prevents divide-by-zero / invalid mapData domain

            # Build Cytoscape elements
            elements = []

            for node in B.nodes():
                deg = degree_dict.get(node, 0)
                if node in bottom_nodes:
                    cls = "bottom"
                    label = node      # shown via stylesheet: "label": "data(label)"
                else:
                    cls = "top"
                    label = ""        # hidden anyway by stylesheet for .top

                elements.append({
                    "data": {
                        "id": node,
                        "label": label,
                        "degree": deg,   # used by stylesheet color map for .top
                    },
                    "classes": cls
                })

            # Edges (weight not used by your stylesheet, but harmless to include)
            for u, v, data in B.edges(data=True):
                w = int(data.get("weight", 1))
                elements.append({"data": {"source": u, "target": v, "weight": w}})

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

            return None, elements, {'name': layout_name}, stylesheet

        # ─── Otherwise fall back to your existing “Refresh” behavior ─────────────────
        if trigger_id == "refresh-network-button-msa":
            # 1) never run before any Refresh click
            if not refresh_clicks:
                return no_update, no_update, no_update, no_update

            # 2) only run when the network tab is active
            if active_tab != "network-graphs":
                return no_update, no_update, no_update, no_update

            # 3) your existing validation & graph‐building code…
            if not selected_project or not selected_files:
                #return html.Div("Please select a project and at least one file.")
                return (
                    dbc.Alert("Please select a project and at least one file."),
                    [],  {"name": "preset"}, [], 
                )

            # --- Load & analyze studies at the chosen network level ---
            studies = []
            all_study_names = [] 
            # Sort by alphabetically order
            selected_files.sort()
            for fname in selected_files:
                path = os.path.join(
                    "Projects", selected_project,
                    "processed-datasets", fname
                )
                if not os.path.exists(path):
                    logger.error(f"Network plots tab - This path doesn't exist: {path}")
                    continue
                
                df = pd.read_csv(path).set_index("database_identifier")
                # create a little analysis container
                class Analysis: pass
                da = Analysis()
                #da.processed_data = df
                da.node_name     = fname.split("_")[1] if len(fname.split("_")) >= 3 else fname

                # full set of studies, even those that may later be filtered out
                """ all_study_names = [
                    fname.split("_")[1] if len(fname.split("_")) >= 3 else fname
                    for fname in selected_files
                ] """
                name = fname.split("_")[1] if len(fname.split("_")) >= 3 else fname
                all_study_names.append(name)

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
                        else:
                            da.processed_data = df

                        get_pathway_data(da)
                        
                    except Exception:
                        logger.exception(f"Network plots tab - Error computing pathways for {da.node_name}")
                        continue
                    
                    # keep only if any differential pathways
                    """ if hasattr(da, "DA_pathways") and len(da.DA_pathways) > 0:
                        studies.append(da) """
                    paths = getattr(da, "DA_pathways", [])
                    if paths:   # truthy only if non-empty
                        print(da.node_name)
                        studies.append(da)
                        
                else:
                    da.pathway_level = False
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
                        else:
                            da.processed_data = df

                        da.da_testing()
                    except Exception:
                        logger.exception(f"Network plots tab - Error computing pathways for {da.node_name}")
                        continue

                    # only keep if DA testing produced metabolites
                    if hasattr(da, "DA_metabolites") and len(da.DA_metabolites) > 0:
                        studies.append(da)

            if not studies:
                #return html.Div("No studies with differentially abundant metabolites.")
                return (
                    dbc.Alert("No studies with differentially abundant metabolites."),
                    [],  {"name": "preset"}, [], 
                )

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
            
            # ---- 1) Gather items per study (unique within a study) ----
            per_study_items = []
            for st in studies:
                if network_level == "diff-metabolite":
                    items = set(st.DA_metabolites)
                else:
                    # choose one: DA pathways (shown) or any coverage>0 (commented)
                    items = set(st.DA_pathways)
                    # items = {pw for pw, cov in st.pathway_coverage.items() if cov > 0}
                per_study_items.append(items)

            # ---- 2) Count in how many studies each node appears ----
            node_counts = Counter()
            for items in per_study_items:
                node_counts.update(items)

            # ---- 3) Apply node-occurrence threshold (never below 2) ----
            threshold = max((min_cooccurring or 2), 2)
            allowed_nodes = {n for n, c in node_counts.items() if c >= threshold}
            if not allowed_nodes:
                label = "differential metabolites" if network_level == "diff-metabolite" else "differential pathways"
                logger.error(f"Network plots tab - No {label} meet the ≥{threshold} studies threshold.")
                return (
                    dbc.Alert(f"No {label} meet the ≥{threshold} studies threshold."),
                    [],  {"name": "preset"}, [], 
                )

            # ---- 4) Count co-occurrence pairs, but ONLY among allowed nodes ----
            pair_counts = Counter()
            for items in per_study_items:
                kept = sorted(x for x in items if x in allowed_nodes)
                for u, v in combinations(kept, 2):
                    pair_counts[(u, v)] += 1

            # Keep all edges that occurred at least once (or raise if you want another edge cut-off)
            edges = [(u, v, c) for (u, v), c in pair_counts.items() if c >= threshold]

            if not edges:
                return (
                    dbc.Alert(
                        f"No {('differential metabolite' if network_level=='diff-metabolite' else 'differenital pathway')} pairs "
                        f"remain after applying ≥{threshold} studies threshold.",
                        color="warning", dismissable=True, is_open=True
                    ),
                    [],  # elements
                    {"name": "preset"},  # layout
                    [],  # stylesheet
                )

            # ---- 5) Build graph; remove isolates that survive thresholding but form no pairs ----
            G = nx.Graph()
            G.add_nodes_from(allowed_nodes)              # ensure all surviving nodes present
            for u, v, w in edges:
                G.add_edge(u, v, weight=w)

            # ChEBI 2.0 - based name lookup for differential metabolites
            chebi_to_name = {}
            if network_level == "diff-metabolite":
                logger.info(f"Network plots tab - Starting ChEBI id to metabolite name conversion for all studies.")
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    for node in G.nodes():
                        chebi_to_name[node] = _safe_chebi_name(node, cursor)
                    conn.commit()  # commit ONCE
                logger.info(f"Network plots tab - Finished ChEBI id to metabolite name conversion for all studies.")
            else:
                for node in G.nodes():
                    # pathway IDs are already human-readable (or you can map them here)
                    chebi_to_name[node] = node


            # --- Prepare Cytoscape elements ---
            #study_names = [st.node_name for st in studies]

            # Precompute palettes
            #pie_pal   = sns.color_palette("Set3", n_colors=len(all_study_names)).as_hex()
            #color_map = dict(zip(all_study_names, pie_pal))

            # degree = # of edges incident to each node
            deg_dict   = dict(G.degree())
            max_degree = max(deg_dict.values())
       
            # nodes that survived the ≥ threshold filtering
            node_set = set(G.nodes())

            # --- collect used studies (order-preserving, no helpers) ---
            used_studies = []
            seen_studies = set()
            for st in studies:  # preserves the order of `studies`
                if network_level == "diff-metabolite":
                    items = set(st.DA_metabolites)
                else:
                    items = set(st.DA_pathways)
                    # If earlier you used coverage>0 instead of DA pathways, switch to:
                    # items = {pw for pw, cov in st.pathway_coverage.items() if cov > 0}

                if node_set & items:  # non-empty intersection => this study contributes at least one kept node
                    if st.node_name not in seen_studies:
                        used_studies.append(st.node_name)
                        seen_studies.add(st.node_name)
            print('used studies')
            print(used_studies)
            #pie_pal   = sns.color_palette("Set3", n_colors=len(used_studies)).as_hex()
            #color_map = dict(zip(used_studies, pie_pal))
            ordered_used = [nm for nm in all_study_names if nm in used_studies]  # stable order
            palette = sns.color_palette("Set3", n_colors=len(ordered_used)).as_hex()
            color_map = dict(zip(ordered_used, palette))
            studies[:] = [st for st in studies if st.node_name in used_studies]
            study_names = [st.node_name for st in studies]

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
                    # Building pie charts

                    # decide presence by network level
                    if network_level == "diff-metabolite":
                        present = [node in st.DA_metabolites for st in studies]
                    else:  # pathway
                        # pathway_coverage[node] > 0 means that pathway was hit
                        #present = [st.pathway_coverage.get(node, 0) > 0 for st in studies]
                        present = [node in st.DA_pathways for st in studies]

                    #present = [node in st.DA_metabolites for st in studies]
                    #labels  = [nm for nm, ok in zip(all_study_names, present) if ok]
                    labels  = [nm for nm, ok in zip(study_names, present) if ok]
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

                for i, name in enumerate(ordered_used):
                    if name not in used_studies:
                        continue   # skip studies that never got a slice

                    legend_nodes.append({
                        "data":     {"id": f"legend-{name}-{i}", "label": name, "study": name},
                        "position": {"x": LEGEND_X, "y": y},
                        "locked":   True,
                        "grabbable": True,
                        "classes":  "legend-node"
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
                # base legend node rule
                stylesheet += [
                    {
                        "selector": ".legend-node",
                        "style": {
                            "label": "data(label)",
                            "text-valign": "center",
                            "text-halign": "right",
                            "text-margin-x": LABEL_MARGIN,
                            "font-size": f"{FONT_SIZE}px",
                            "color": "#000",
                            "shape": "rectangle",
                            "width": BOX_SIZE,
                            "height": BOX_SIZE,
                        },
                    }
                ]

                # 🔧 add per-study color rules
                study_rules = [
                    {
                        "selector": f'node.legend-node[study = "{name}"]',
                        "style": {"background-color": color_map[name]},
                    }
                    for name in color_map.keys()
                ]
                stylesheet += study_rules

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

            return None, elements, {'name': layout_name}, stylesheet
    
    # Callback: Updates RefMet mapping and pathway coverage tables based on selected studies and network level
    @callback(
        Output("refmet-conversion-table", "data"),
        Output("pathway-coverage-table", "data"),
        Input("refresh-network-button-msa", "n_clicks"),
        State("project-files-checklist-msa", "value"),
        State("project-dropdown-pop-msa", "value"),
        State("network-level-dropdown-msa", "value"),
    )
    def update_refmet_and_pathway_tables(n_clicks, selected_files, selected_project, node_level):
        if not n_clicks or not (selected_project and selected_files):
            return no_update, no_update

        # ─── Load Reactome definitions ─────────────────────────────────────────────
        reactome_dict = {}
        pathway_names = {}
        if node_level == "pathway":
            reactome_paths = sspa.process_gmt(infile='Reactome_Homo_sapiens_pathways_ChEBI_R90.gmt')
            reactome_dict = sspa.utils.pathwaydf_to_dict(reactome_paths)
            pathway_names = dict(zip(reactome_paths.index, reactome_paths['Pathway_name']))
            all_rxn_ids = set().union(*reactome_dict.values())

        # ─── Tables to fill ────────────────────────────────────────────────────────
        refmet_mapping_records = []
        pathway_coverage_records = []

        for fname in selected_files:
            study_name = fname.split("_")[1] if len(fname.split("_")) >= 3 else fname
            csv_path = os.path.join('Projects', selected_project, "processed-datasets", fname)

            try:
                df = pd.read_csv(csv_path).set_index("database_identifier")
            except Exception as e:
                logger.exception(f"Network plots tab - Error reading csv {fname}")
                continue

            # Get dataset source (e.g., 'metabolomics workbench' or 'refmet ids')
            folder_details = os.path.join("pre-processed-datasets", study_name)
            details = read_study_details_msa(folder_details)
            dataset_source = details.get("Dataset Source", "").lower()

            # ─── RefMet → ChEBI Mapping Stats ───────────────────────────────────────
            if dataset_source in ("metabolomics workbench", "original data - refmet ids"):
                all_cols = list(df.columns)
                keep_cols = {'database_identifier', 'group_type'}
                met_cols = [c for c in all_cols if c not in keep_cols]
                unmapped = [c for c in met_cols if c not in refmet2chebi]

                total_refmet = len(met_cols)
                num_unmapped = len(unmapped)
                num_mapped = total_refmet - num_unmapped
                pct_unmapped = (num_unmapped / total_refmet * 100) if total_refmet else 0.0

                refmet_mapping_records.append({
                    "study_name": study_name,
                    "total_refmet": total_refmet,
                    "num_mapped": num_mapped,
                    "num_unmapped": num_unmapped,
                    "pct_unmapped": round(pct_unmapped, 1)
                })

            # ─── Pathway Mapping Stats (Always if node_level == 'pathway') ───────────
            if node_level == "pathway":
                processed_data = df.copy()
                original_cols = list(processed_data.columns)
                keep_cols = {'database_identifier', 'group_type'}

                if dataset_source in ("metabolomics workbench", "original data - refmet ids"):
                    drop_columns = []
                    rename_mapping = {}

                    for col in original_cols:
                        if col in keep_cols:
                            rename_mapping[col] = col
                        else:
                            new_name = refmet2chebi.get(col)
                            if not new_name or pd.isna(new_name):
                                drop_columns.append(col)
                            else:
                                rename_mapping[col] = new_name

                    refmet_cols = [c for c in original_cols if c not in keep_cols]
                    processed_data = processed_data.drop(columns=drop_columns)
                    processed_data = processed_data.rename(columns=rename_mapping)

                # Strip CHEBI prefix
                processed_data.columns = processed_data.columns.str.removeprefix("CHEBI:")

                metabolite_cols = set(processed_data.columns) - {"group_type"}
                total_metabolites = len(metabolite_cols)
                mapped_metabolites = metabolite_cols.intersection(all_rxn_ids)
                unmapped_metabolites = metabolite_cols.difference(all_rxn_ids)

                num_mapped = len(mapped_metabolites)
                num_unmapped = len(unmapped_metabolites)
                pct_mapped = (num_mapped / total_metabolites * 100) if total_metabolites > 0 else 0.0

                mapped_pathways = {
                    pid for pid, members in reactome_dict.items()
                    if set(members).intersection(metabolite_cols)
                }

                pathway_coverage_records.append({
                    "study_name": study_name,
                    "total_metabolites": total_metabolites,
                    "mapped_to_pathways": num_mapped,
                    "not_mapped": num_unmapped,
                    "pct_mapped": round(pct_mapped, 1),
                    "num_pathways_covered": len(mapped_pathways)
                })

        return refmet_mapping_records, pathway_coverage_records

    # Callback: Opens or closes the save plot modal when the save or confirm button is clicked
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

    # Callback: Generates and downloads the network plot as an SVG file with a project-based filename
    @callback(
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

        logger.info(f"Network plots tab - Saving network plot to downloads folder with name: {filename}")

        return {
            "type":     "svg",
            "action":   "download",
            "filename": filename
        } 
