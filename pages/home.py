# pages/home.py
from dash import html
import dash_bootstrap_components as dbc

# Steps configuration
STEPS = [
    {"label": "Data Pre-Processing", "icon": "bi bi-database-gear", "description": "Upload, clean and prepare your data for analysis."},
    {"label": "Datasets", "icon": "bi bi-table", "description": "View processed datasets ready for analysis."},
    {"label": "Single-Study Analysis", "icon": "fa-solid fa-magnifying-glass", "description": "Analyse differential metabolites and pathways within a single study."},
    {"label": "Multi-Study Analysis", "icon": "fa-solid fa-magnifying-glass", "description": "Integrate and compare results across multiple studies."},
    {"label": "Plots", "icon": "bi bi-file-bar-graph", "description": "Visualise your results."},
]

layout = dbc.Container(
    [
        # App title and short description
        html.H1("Metabolomic Data Analysis App", className="mt-5 mb-3"),
        dbc.Card(
            dbc.CardBody(
                [
                    html.P(
                        "This app provides an interactive platform for visualising and analysing differential metabolites and pathways across single and multi-study datasets. It allows users to explore metabolic data, comparing results from individual studies and integrating data from large-scale public repositories such as MetaboLights and MetabolomicsWorkbench or your own data with Refmet or Chebi ids.",
                        className="mb-2"
                    ),
                    html.P(
                        "By leveraging harmonized metabolite annotations, the app enables users to perform in-depth analyses, uncover multi-study signatures, and maximise the reusability and reproducibility of metabolomics data. With a focus on improving the standardization and integration of metabolomics datasets, this tool aims to support scientific discovery through comprehensive and accessible data visualisation and analysis.",
                        className="mb-0"
                    )
                ]
            ),
            className="shadow-sm mb-5"
        ),

        # Steps section
        html.H2("Getting Started", className="h4 mb-4"),
        dbc.ListGroup(
            [
                dbc.ListGroupItem(
                    dbc.Row(
                        [
                            dbc.Col(dbc.Badge(f"{i+1}", color="primary", className="mr-2"), width="auto"),
                            dbc.Col(html.I(className=step['icon'], style={"fontSize": "1.5rem", "verticalAlign": "middle", "marginRight": "1rem"}), width="auto"),
                            dbc.Col(
                                html.Div([
                                    html.Span(step['label'], className="font-weight-bold"),
                                    " - ",
                                    html.Span(step['description'])
                                ]),
                                className="pl-2"
                            ),
                        ],
                        align="center"
                    ),
                    className="mb-2 shadow-none"
                )
                for i, step in enumerate(STEPS)
            ],
            flush=True
        ),

        # Call to action
        html.Div(
            dbc.Button("Start Now", href="/data-pre-processing", color="success", size="lg"),
            className="text-center mt-5 mb-5"
        )
    ],
    fluid=True,
    style={"maxWidth": "720px"}
)
