import pandas as pd
import numpy as np

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

import plotly.graph_objects as go
import plotly.io as pio

import amp_consts
import amp_st_functs
from amp_functs import (
    build_plot,
    get_plot_help_digest,
    get_plot_docstring,
    get_final_index,
    ParamInitializer,
    apply_wrangling,
)

# app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#303030",
    "border-left:": "solid",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

cmp_upload = dbc.Alert(
    dbc.Table(
        html.Tr(
            [
                html.Td(dcc.Upload("Drag and Drop or ")),
                html.Td(
                    html.Div(
                        dbc.Button(dcc.Upload(html.A("Upload File")), color="primary")
                    ),
                ),
            ]
        ),
        borderless=False,
        # color="dark",
    ),
    color="dark",
)

sidebar = html.Div(
    [
        html.P("A simple sidebar layout with navigation links", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


div_plotting = html.Div(
    [
        cmp_upload,
    ],
    id="page-ploting",
)
div_wrangling = html.Div(id="page-wrangling", style=CONTENT_STYLE)
div_settings = html.Div(id="page-settings", style=CONTENT_STYLE)
# tab_plotting = dbc.Card(dbc.CardBody([div_plotting]), className="mt-3")
# tab_wrangling = dbc.Card(dbc.CardBody([div_wrangling]), className="mt-3")
# tab_settings = dbc.Card(dbc.CardBody([div_settings]), className="mt-3")

jumbotron = html.Div(
    dbc.Container(
        [
            html.H1(
                "Ex Taedio",
                className="display-3",
            ),
            html.P(
                "Dataframe (CSV) plotting dashboard",
                className="lead",
            ),
        ],
        fluid=True,
        # className="py-3",
    ),
)

div_main = html.Div(
    children=[
        jumbotron,
        dbc.Tabs(
            [
                dbc.Tab(div_plotting, label="Plotting"),
                dbc.Tab(div_wrangling, label="Wrangling"),
                dbc.Tab(div_settings, label="Settings"),
            ]
        ),
    ],
    id="div-content",
    style=CONTENT_STYLE,
)

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        sidebar,
        div_main,
    ],
)


if __name__ == "__main__":
    app.run_server(port=8888, debug=True)
