import os
from dash import html, dcc, callback_context, no_update
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
from app import app
from dash import html
import base64

# Base folder for projects.
projects_folder = "Projects"

# Helper to get subfolders of any folder.
# helper as before
def list_subfolders(folder):
    try:
        return sorted([f for f in os.listdir(folder)
                       if os.path.isdir(os.path.join(folder, f))])
    except FileNotFoundError:
        return []

def list_files(folder):
    try:
        return sorted([f for f in os.listdir(folder)
                       if os.path.isfile(os.path.join(folder, f))])
    except FileNotFoundError:
        return []
    

layout = html.Div([
    # store the user’s project choice
    dcc.Store(id='selected-project-store'),

    # 1) Modal that pops up immediately
    dbc.Modal(
        [
            dbc.ModalHeader("Select a Project"),
            dbc.ModalBody(
                dcc.Dropdown(
                    id='project-dropdown-modal',
                    options=[
                        {'label': p.replace('-', ' '), 'value': p}
                        for p in list_subfolders(projects_folder)
                    ],
                    placeholder='Choose project…',
                    clearable=False,
                    style={'width': '100%'}
                )
            ),
            dbc.ModalFooter(
                dbc.Button(
                    "Confirm",
                    id='confirm-project-btn',
                    color='primary',
                    disabled=True
                )
            ),
        ],
        id='project-modal-plots',
        is_open=True,
        backdrop='static',   # force them to choose
        keyboard=False
    ),

    # 2) Main content (hidden until modal closes)
    html.Div(
        [
            # subtitle + plot‐folder dropdown in one row
            html.Div(
                [
                    html.H3( id='project-title', style={'margin': 0} ),

                    dcc.Dropdown(
                        id='plot-folder-dropdown',
                        options=[],      # filled by callback
                        value='',
                        placeholder='Select plot folder',
                        clearable=False,
                        style={'width': '250px'}
                    ),
                ],
                style={
                    'display': 'flex',
                    'justifyContent': 'space-between',
                    'alignItems': 'center',
                    'marginBottom': '20px'
                }
            ),

            # subfolder table 
            # MAIN CONTENT: two columns
            html.Div([
                # LEFT: table wrapper
                html.Div(
                    id='subfolder-table-wrapper',
                    style={
                        'width': '60%',
                        'display': 'inline-block',
                        'verticalAlign': 'top',
                        'marginRight': '2%'
                    }
                ),

                # RIGHT: preview box
                html.Div(
                    id='preview-box',
                    children=html.Div("Click a file to preview.", style={'color':'#888'}),
                    style={
                        'width': '35%',
                        'display': 'inline-block',
                        'verticalAlign': 'top',
                        'border': '1px solid #ccc',
                        'borderRadius': '5px',
                        'padding': '10px',
                        'minHeight': '300px'
                    }
                ),
            ], style={'margin': '20px 50px'}),

            # file list under clicked subfolder
            #html.Div(id='file-list', style={'marginTop': '20px'}),
        ],
        id='page-content',
        style={'margin': '0 50px'}  # indent main area
    ),
])

# — enable the “Confirm” button once they pick something —
@app.callback(
    Output('confirm-project-btn', 'disabled'),
    Input('project-dropdown-modal', 'value')
)
def _enable_confirm(val):
    return val is None

# — when they hit Confirm: close modal, store project, set subtitle —
@app.callback(
    Output('project-modal-plots', 'is_open'),
    Output('selected-project-store', 'data'),
    Output('project-title', 'children'),
    Input('confirm-project-btn', 'n_clicks'),
    State('project-dropdown-modal', 'value'),
    prevent_initial_call=True
)
def _confirm_project(n_clicks, project):
    # only fires when they click Confirm
    return False, project, project.replace('-', ' ')

# — populate the plot‐folder dropdown from the chosen project —
@app.callback(
    Output('plot-folder-dropdown', 'options'),
    Output('plot-folder-dropdown', 'value'),
    Input('selected-project-store', 'data'),
)
def update_plot_folders(selected_project):
    if not selected_project:
        return [], ''
    path = os.path.join(projects_folder, selected_project, "Plots")
    raw = list_subfolders(path)
    opts = [{'label': f.replace('-', ' '), 'value': f} for f in raw]
    return opts, ''


# paste these style dicts somewhere at module‐level so you can reuse them
style_table = {
    "width": "100%",
    "border": "none",                  # no outer border/vertical lines
    "borderRadius": "5px",
    "boxShadow": "2px 2px 5px rgba(0, 0, 0, 0.1)",
    "overflowX": "auto",
    "fontFamily": "Arial",
    "borderCollapse": "collapse",      # collapse any cell borders
}
style_header = {
    "backgroundColor": "#f2f2f2",
    "fontFamily": "Arial",
    "fontSize": "16px",
    "fontWeight": "bold",
    "textAlign": "left",
    "borderBottom": "1px solid #ddd",
    "padding": "10px"
}
style_cell = {
    "fontFamily": "Arial",
    "fontSize": "14px",
    "textOverflow": "ellipsis",
    "whiteSpace": "nowrap",
    "overflow": "hidden",
    "textAlign": "left",
    "borderBottom": "1px solid #ddd",  # only horizontal separators
    "padding": "5px",                  # less vertical space
}

""" @app.callback(
    Output('subfolder-table-wrapper', 'children'),
    Input ('plot-folder-dropdown','value'),
    State('selected-project-store','data'),
)
def render_subfolder_table(plot_folder, project):
    if not project or not plot_folder:
        return html.Div("Select a plot-folder above…")

    base = os.path.join(projects_folder, project, "Plots", plot_folder)
    subs = list_subfolders(base)

    # header row (if you want one)
    thead = html.Thead(
        html.Tr(html.Th("Folder", style=style_cell))
    )

    # table body
    rows = []
    for idx, sub in enumerate(subs):
        human = sub.replace('-', ' ')
        files = list_files(os.path.join(base, sub))

        # collapsible summary + file-boxes
        details = html.Details([
            html.Summary(human, style={
                **style_cell,
                "cursor": "pointer"
            }),
            html.Div(
                [
                    html.Div(
                        f,
                        style={
                            'border': '1px solid #ddd',
                            'borderRadius': '5px',
                            'padding': '5px 10px',
                            'margin': '4px',
                            'whiteSpace': 'nowrap',
                            'fontFamily': 'Arial',
                            'fontSize': '14px'
                        }
                    )
                    for f in files
                ] or ["No files."],
                style={'display': 'flex', 'flexWrap': 'wrap', 'padding': '8px 16px'}
            )
        ], open=False)

        # zebra‐stripe on odd rows
        row_style = {"backgroundColor": "#f9f9f9"} if idx % 2 else {}

        rows.append(
            html.Tr(html.Td(details, style=style_cell), style=row_style)
        )

    table = html.Table([thead, html.Tbody(rows)], style=style_table)
    return table """

@app.callback(
    Output('subfolder-table-wrapper', 'children'),
    Input('plot-folder-dropdown', 'value'),
    State('selected-project-store', 'data'),
)
def render_subfolder_table(plot_folder, project):
    if not project or not plot_folder:
        return html.Div("Select a plot-folder above…", style={'fontStyle':'italic'})

    base = os.path.join(projects_folder, project, "Plots", plot_folder)
    subs = list_subfolders(base)

    # Build the table rows
    rows = []
    for idx, sub in enumerate(subs):
        human = sub.replace('-', ' ')
        sub_path = os.path.join(base, sub)
        files = list_files(sub_path)

        # make each file a button with a pattern-matching ID
        file_buttons = [
            html.Button(
                f,
                id={'type': 'file-button', 'path': os.path.join(sub_path, f)},
                n_clicks=0,
                style={
                    'border': '1px solid #ddd',
                    'borderRadius': '3px',
                    'padding': '4px 8px',
                    'margin': '4px',
                    'whiteSpace': 'nowrap',
                    'cursor': 'pointer',
                    'fontSize': '13px'
                }
            )
            for f in files
        ] or [html.Div("No files.", style={'color':'#888', 'padding':'8px'})]

        # one <details> per subfolder
        details = html.Details([
            html.Summary(human, style={
                'padding': '5px',
                'cursor': 'pointer',
                'fontSize': '14px'
            }),
            html.Div(file_buttons, style={
                'display': 'flex', 'flexWrap': 'wrap', 'padding': '5px 10px'
            })
        ], open=False, style={
            'borderBottom': '1px solid #ddd',
            'padding': '0'
        })

        # zebra-striping
        tr_style = {'backgroundColor': '#f9f9f9'} if idx % 2 else {}
        rows.append(html.Tr(html.Td(details), style=tr_style))

    # wrap in a narrow table
    return html.Table(
        [html.Tbody(rows)],
        style={
            'width': '100%',
            'borderCollapse': 'collapse',
            'fontFamily': 'Arial',
            'fontSize': '14px'
        }
    )


@app.callback(
    Output('preview-box', 'children'),
    Input({'type': 'file-button', 'path': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def preview_file(n_clicks_list):
    ctx = callback_context
    file_id = ctx.triggered_id  # this will be the dict {'type':'file-button','path':...}

    # if somehow triggered without a file-button click, do nothing
    if not file_id or 'path' not in file_id:
        return no_update

    path = file_id['path']
    ext = os.path.splitext(path)[1].lower()

    if ext == '.svg':
        with open(path, 'rb') as f:
            svg_bytes = f.read()
        b64 = base64.b64encode(svg_bytes).decode('utf-8')
        return html.Img(
            src='data:image/svg+xml;base64,' + b64,
            style={'width': '100%', 'height': 'auto'}
        )
    elif ext == '.csv':
        return html.Div(
            "Cannot preview CSV files.",
            style={'fontStyle': 'italic', 'color': '#a00'}
        )
    else:
        return html.Div(
            f"No preview available for *{ext}* files.",
            style={'fontStyle': 'italic', 'color': '#a00'}
        )


""" 
@app.callback(
    Output("plots-files-table", "children"),
    [Input("project-dropdown-plt", "value"), Input("url", "pathname")]
)
def show_plots(selected_project, pathname):
    # If no project has been selected, show an instructive message.
    if not selected_project:
        return html.Div("Select a project to show processed datasets", style={"fontFamily": "Arial"})

    # Build the base path for processed datasets in the selected project.
    base_path = os.path.join(projects_folder, selected_project, "Plots")

    if not os.path.exists(base_path):
        return html.Div(f"No '{base_path}' folder found.", style={"fontFamily": "Arial"})

    files = os.listdir(base_path)
    if not files:
        return html.Div(f"No files found in '{base_path}'.", style={"fontFamily": "Arial"})

    file_records = []
    for f in sorted(files):
        file_path = os.path.join(base_path, f)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(f)[1].lower()
        plot_type = ext.replace('.', '').upper() if ext in ['.csv', '.tsv', '.parquet'] else 'UNKNOWN'
        rows, cols = 0, 0
        try:
            if ext == '.csv':
                df_temp = pd.read_csv(file_path)
                rows, cols = df_temp.shape
            elif ext == '.tsv':
                df_temp = pd.read_csv(file_path, sep='\t')
                rows, cols = df_temp.shape
            elif ext == '.parquet':
                df_temp = pd.read_parquet(file_path)
                rows, cols = df_temp.shape
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        last_updated = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%d/%m/%Y %I:%M %p")
        file_records.append({
            "Name": f,
            "Plot type": plot_type,
            "Last updated": last_updated
        })

    # Update title text: just use the project name with hyphens replaced by spaces.
    title_text = selected_project.replace("-", " ")
    header = html.H3(title_text, style={"fontFamily": "Arial"})

    columns = [
        {"name": col, "id": col} for col in ["Name", "Plot type", "Last updated"]
    ]

    table = dash_table.DataTable(
        data=file_records,
        columns=columns,
        page_size=10,
        style_table={
            "overflowX": "auto",
            "marginRight": "50px",
            "border": "1px solid #ccc",
            "borderRadius": "5px",
            "boxShadow": "2px 2px 5px rgba(0, 0, 0, 0.1)"
        },
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
        },
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"}],
        style_as_list_view=True
    )

    return html.Div([header, table])
 """
