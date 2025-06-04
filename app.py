# app.py
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Initialize Dash app
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css", 
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.0/css/all.min.css"
]
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True
)

server = app.server  # If you need the underlying Flask server for deployment

app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            setTimeout(function() {
                var dropdownInput = document.querySelector("input[role='combobox']");
                if (dropdownInput) {
                    dropdownInput.blur();
                    console.log("Forced blur, active element:", document.activeElement);
                }
            }, 400);
        }
        return "";
    }
    """,
    Output("dummy-div", "children"),
    Input("confirm-group-label-btn_dpp2", "n_clicks")
)








