from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from pages import home, data_pre_processing, datasets, single_study_analysis, multi_study_analysis, plots
from app import app

# Sidebar with icons above each name.
sidebar = html.Div(
    [
        dbc.Nav(
            [
                dbc.NavLink(
                    [
                        html.Div(
                            html.I(className="bi bi-house", style={"fontSize": "24px"}),
                            style={"textAlign": "center"}
                        ),
                        html.Div("Home", style={"fontSize": "12px", "textAlign": "center"})
                    ],
                    href="/",
                    active="exact",
                    style={"padding": "10px"}
                ),
                dbc.NavLink(
                    [
                        html.Div(
                            html.I(className="bi bi-database-gear", style={"fontSize": "24px"}),
                            style={"textAlign": "center"}
                        ),
                        html.Div("Data Pre-Processing", style={"fontSize": "12px", "textAlign": "center"})
                    ],
                    href="/data-pre-processing",
                    active="exact",
                    style={"padding": "10px"}
                ),
                dbc.NavLink(
                    [
                        html.Div(
                            html.I(className="bi bi-table", style={"fontSize": "24px"}),
                            style={"textAlign": "center"}
                        ),
                        html.Div("Datasets", style={"fontSize": "12px", "textAlign": "center"})
                    ],
                    href="/datasets",
                    active="exact",
                    style={"padding": "10px"}
                ),
                dbc.NavLink(
                    [
                        html.Div(
                            html.I(className="fa-solid fa-magnifying-glass", style={"fontSize": "24px"}),
                            style={"textAlign": "center"}
                        ),
                        html.Div("Single-Study Analysis", style={"fontSize": "12px", "textAlign": "center"})
                    ],
                    href="/single_study_analysis",
                    active="exact",
                    style={"padding": "10px"}
                ),
                dbc.NavLink(
                    [
                        html.Div(
                            html.I(className="fa-solid fa-magnifying-glass", style={"fontSize": "24px"}),
                            style={"textAlign": "center"}
                        ),
                        html.Div("Multi-Study Analysis", style={"fontSize": "12px", "textAlign": "center"})
                    ],
                    href="/multi-study-analysis",
                    active="exact",
                    style={"padding": "10px"}
                ),
                dbc.NavLink(
                    [
                        html.Div(
                            html.I(className="bi bi-file-bar-graph", style={"fontSize": "24px"}),
                            style={"textAlign": "center"}
                        ),
                        html.Div("Plots", style={"fontSize": "12px", "textAlign": "center"})
                    ],
                    href="/plots",
                    active="exact",
                    style={"padding": "10px"}
                ),

            ],
            vertical=True,
            pills=True,
        )
    ],
    style={
        "width": "11rem",
        "height": "100vh",
        "position": "fixed",
        "top": 0,
        "left": 0,
        "backgroundColor": "#f8f9fa",
        "borderRight": "1px solid #dee2e6",  # Vertical divider
        "overflowX": "hidden",
        "padding": "1rem",
    },
)

# Main content with a dynamic title and a horizontal line
content = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        # Header area for title and horizontal line.
        html.Div(
            [
                html.H2("My Page Title", id="dynamic-title", style={"marginBottom": "0"}),
                html.Hr(style={"marginTop": "0", "borderColor": "#dee2e6"}),
            ],
            style={"padding": "0rem 0rem 1rem 0rem"}
        ),
        # Page content appears below the header.
        html.Div(id="page-content", style={"padding": "0rem"}),
    ],
    style={"marginLeft": "15rem"},  # Offset content to the right of the sidebar
)

app.layout = html.Div([sidebar, 
                       content,
                       dcc.Store(id="modal-state-store")])

# Callback to dynamically update both the page content and the title.
@app.callback(
    [Output("page-content", "children"), Output("dynamic-title", "children")],
    [Input("url", "pathname"), Input("url", "search")]
)

def display_page_and_title(pathname, search):
    if pathname == "/data-pre-processing":
        return data_pre_processing.layout, "Data Pre-Processing"
    elif pathname == "/datasets":
        return datasets.layout, "Datasets"
    elif pathname == "/single_study_analysis":
        return single_study_analysis.layout, "Study Analysis"
    elif pathname == "/multi-study-analysis":
        return multi_study_analysis.layout, "Multi-Study Analysis"
    elif pathname == "/plots":
        return plots.layout, "Plots"
    elif pathname == "/":
        return home.layout, "Home"
    else:
        return home.layout, "Home"



if __name__ == "__main__":
    app.run_server(debug=True)

