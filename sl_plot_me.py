import io

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.io as pio

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime as dt

from dw_cssegis_data import get_wrangled_cssegis_df

PICK_ONE = "Pick one..."
NONE_SELECTED = "None selected"

PLOT_SCATTER = "Scatter"
PLOT_LINE = "Line"
PLOT_BAR = "Bar"
PLOT_HISTOGRAM = "Histogram"
PLOT_BOX = "Box plot - if you must"
PLOT_VIOLIN = "Violin plot"
PLOT_DENSITY_HEATMAP = "Density heat map"
PLOT_DENSITY_CONTOUR = "Density contour"
PLOT_PARALLEL_CATEGORIES = "Parallel categories"
PLOT_PARALLEL_COORDINATES = "Parallel coordinates"
PLOT_SCATTER_MATRIX = "Scatter matrix"
PLOT_PCA_2D = "PCA (2D)"
PLOT_PCA_3D = "PCA (3D)"
PLOT_CORR_MATRIX = "Correlation matrix"

PLOT_HAS_X = [
    PLOT_SCATTER,
    PLOT_LINE,
    PLOT_BAR,
    PLOT_HISTOGRAM,
    PLOT_BOX,
    PLOT_VIOLIN,
    PLOT_DENSITY_HEATMAP,
    PLOT_DENSITY_CONTOUR,
]
PLOT_HAS_Y = [
    PLOT_SCATTER,
    PLOT_LINE,
    PLOT_BAR,
    PLOT_BOX,
    PLOT_VIOLIN,
    PLOT_DENSITY_HEATMAP,
    PLOT_DENSITY_CONTOUR,
]
PLOT_HAS_Z = [
    PLOT_DENSITY_HEATMAP,
    PLOT_DENSITY_CONTOUR,
]
PLOT_HAS_COLOR = [
    PLOT_SCATTER,
    PLOT_PCA_2D,
    PLOT_PCA_3D,
    PLOT_LINE,
    PLOT_BAR,
    PLOT_HISTOGRAM,
    PLOT_BOX,
    PLOT_VIOLIN,
    PLOT_DENSITY_CONTOUR,
    PLOT_PARALLEL_CATEGORIES,
    PLOT_PARALLEL_COORDINATES,
    PLOT_SCATTER_MATRIX,
]
PLOT_HAS_TEXT = [
    PLOT_SCATTER,
    PLOT_PCA_2D,
    PLOT_PCA_3D,
    PLOT_LINE,
    PLOT_BAR,
    PLOT_PCA_2D,
    PLOT_PCA_3D,
]
PLOT_HAS_FACET = [
    PLOT_SCATTER,
    PLOT_PCA_2D,
    PLOT_LINE,
    PLOT_BAR,
    PLOT_HISTOGRAM,
    PLOT_BOX,
    PLOT_VIOLIN,
    PLOT_DENSITY_HEATMAP,
    PLOT_DENSITY_CONTOUR,
]
PLOT_HAS_LOG = [
    PLOT_SCATTER,
    PLOT_LINE,
    PLOT_BAR,
    PLOT_HISTOGRAM,
    PLOT_BOX,
    PLOT_VIOLIN,
    PLOT_DENSITY_HEATMAP,
    PLOT_DENSITY_CONTOUR,
]
PLOT_HAS_BINS = [
    PLOT_DENSITY_HEATMAP,
    PLOT_DENSITY_CONTOUR,
    PLOT_DENSITY_CONTOUR,
]
PLOT_HAS_MARGINAL = [
    PLOT_HISTOGRAM,
]
PLOT_HAS_MARGINAL_XY = [
    PLOT_SCATTER,
    PLOT_PCA_2D,
    PLOT_DENSITY_HEATMAP,
    PLOT_DENSITY_CONTOUR,
]
PLOT_HAS_BAR_MODE = [
    PLOT_HISTOGRAM,
]
PLOT_HAS_SIZE = [
    PLOT_SCATTER,
    PLOT_PCA_2D,
    PLOT_PCA_3D,
]
PLOT_HAS_SHAPE = [
    PLOT_SCATTER,
    PLOT_PCA_2D,
    PLOT_PCA_3D,
]
PLOT_HAS_TREND_LINE = [
    PLOT_SCATTER,
    PLOT_DENSITY_CONTOUR,
]
PLOT_HAS_POINTS = [
    PLOT_BOX,
    PLOT_VIOLIN,
]


URL_CSSE = "url_csse"
URL_COVID_DATA = "https://raw.githubusercontent.com/coviddata/coviddata/master/data/sources/jhu_csse/standardized/standardized.csv"
URL_ECDC_COVID = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
URL_IRIS = "url_iris"
URL_TIPS = "url_tips"
URL_GAPMINDER = "url_gapminder"
URL_WIND = "url_wind"
URL_ELECTION = "url_election"
URL_CARSHARE = "url_carshare"


def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


def format_csv_link(url):
    if url == URL_CSSE:
        return "COVID-19 - CSSE, modified version"
    elif url == URL_ECDC_COVID:
        return "COVID-19 - European Centre for Disease Prevention and Control"
    elif url == URL_COVID_DATA:
        return "COVID-19 - Github coviddata"
    elif url == URL_IRIS:
        return "Iris plotly express data set"
    elif url == URL_CARSHARE:
        return "Carshare plotly express data set"
    elif url == URL_GAPMINDER:
        return "Gapminder plotly express data set"
    elif url == URL_TIPS:
        return "Tips plotly express data set"
    elif url == URL_WIND:
        return "Wind plotly express data set"
    else:
        return url


@st.cache
def get_dataframe_from_url(url):
    if url == PICK_ONE:
        return None
    elif isinstance(url, io.StringIO):
        return pd.read_csv(url)
    elif url == URL_CSSE:
        return get_wrangled_cssegis_df(allow_cache=True)
    elif url == URL_IRIS:
        return px.data.iris()
    elif url == URL_CARSHARE:
        return px.data.carshare()
    elif url == URL_GAPMINDER:
        return px.data.gapminder()
    elif url == URL_TIPS:
        return px.data.tips()
    elif url == URL_WIND:
        return px.data.wind()
    elif url == URL_ELECTION:
        return px.data.election()
    elif isinstance(url, str):
        return pd.read_csv(url)
    else:
        return None


st.title("Ex Taedio")

st.markdown("""Build plots the (kind of) easy way.""")

st.subheader("Display options")
use_side_bar = st.checkbox(
    label="Put the questions to customize the plot in th sidebar? Recommended.", value=True
)
if st.checkbox(
    label="Force wide display? - can also be set from the settings. This overrides the settings if checked.",
    value=True,
):
    _max_width_()

st.subheader("Advanced settings, the more you check the weirder it gets.")
show_dw_options = st.checkbox(
    label="Show dataframe customization options - Remove columns or rows."
)
show_advanced_settings = st.checkbox(
    label="Show plot customization advanced options", value=False
)


def filter_none(value):
    return None if value == NONE_SELECTED else value


def add_histogram(fig, x, index, name="", marker=None):
    fig.add_trace(
        go.Histogram(
            x=x,
            showlegend=index == 1,
            marker={} if marker is None else {"color": marker},
            name=name,
        ),
        row=index,
        col=index,
    )


def add_scatter(fig, x, y, row, col, marker=None, opacity=0.5, legend=False):
    fig.add_scatter(
        x=x,
        y=y,
        mode="markers",
        marker={} if marker is None else {"color": marker},
        opacity=opacity,
        showlegend=legend,
        row=row,
        col=col,
    )


def add_2d_hist(fig, x, y, row, col, legend=False):
    fig.add_trace(
        go.Histogram2dContour(
            x=x, y=y, reversescale=True, xaxis="x", yaxis="y", showlegend=legend,
        ),
        row=row,
        col=col,
    )


def customize_plot():

    st.header("Load dataframe, usually a CSV file")
    selected_file = st.selectbox(
        label="Source file: ",
        options=[
            PICK_ONE,
            "Load local file",
            URL_CARSHARE,
            URL_ELECTION,
            URL_GAPMINDER,
            URL_IRIS,
            URL_TIPS,
            URL_WIND,
            URL_CSSE,
            URL_COVID_DATA,
            URL_ECDC_COVID,
        ],
        index=0,
        format_func=format_csv_link,
    )
    if selected_file == "Load local file":
        selected_file = st.file_uploader(label="Select file to upload")
        if selected_file is None:
            return
    df_loaded = get_dataframe_from_url(url=selected_file)
    if df_loaded is None:
        return
    df = df_loaded.copy().reset_index(drop=True)

    if show_dw_options:
        # Filter
        st.header("Filtering")

        st.subheader("Selected data frame")
        max_dot_size = st.number_input(
            label="Lines to display", min_value=5, max_value=1000, value=5
        )
        st.dataframe(df.head(max_dot_size))

        st.subheader("Dataframe description")
        st.dataframe(df.describe())

        st.subheader("Select columns to keep")
        kept_columns = st.multiselect(
            label="Columns to keep", options=df.columns.to_list(), default=df.columns.to_list(),
        )
        df = df[kept_columns]

        st.subheader("Filter rows")
        filter_columns = st.multiselect(
            label="Select which columns you want to use to filter the rows:",
            options=df.select_dtypes(include=["object", "datetime"]).columns.to_list(),
            default=None,
        )
        filters = {}
        for column in filter_columns:
            st.subheader(f"{column}: ")
            select_all = st.checkbox(label=f"{column} Select all:")
            elements = list(df[column].unique())
            filters[column] = st.multiselect(
                label=f"Select which {column} to include",
                options=elements,
                default=None if not select_all else elements,
            )
        if len(filters) > 0:
            for k, v in filters.items():
                df = df[df[k].isin(v)]

        st.subheader("Clean up")
        if st.checkbox(label="Remove duplicates", value=False):
            df = df.drop_duplicates()
        if st.checkbox(label="Remove rows with NA values", value=False):
            df = df.dropna(axis="index")

        df = df.reset_index(drop=True)

        # Preview dataframe
        st.subheader("filtered data frame")
        st.dataframe(df.describe())

    qs = st.sidebar if use_side_bar else st
    if use_side_bar:
        pass
    else:
        st.header("Plot customization")

    num_columns = df.select_dtypes(include=[np.number]).columns.to_list()
    cat_columns = df.select_dtypes(include=["object", "datetime"]).columns.to_list()
    all_columns = df.columns.to_list()

    # Select mode
    plot_mode = qs.selectbox(label="Plot mode: ", options=["Static", "Animation"], index=0,)
    if plot_mode == "Animation":
        usual_time_columns = [
            "date",
            "date_time",
            "datetime",
            "timestamp",
            "time",
            "daterep",
        ]
        time_columns = [
            col for col in df.columns.to_list() if col.lower() in usual_time_columns
        ]
        if not time_columns:
            time_columns = [PICK_ONE]
        time_columns.extend(
            [col for col in df.columns.to_list() if col.lower() not in time_columns]
        )
        time_column = qs.selectbox(label="Date/time column: ", options=time_columns, index=0,)
        if time_column != PICK_ONE and (
            time_column in usual_time_columns
            or qs.checkbox(label="Convert to date?", value=time_column in usual_time_columns)
        ):
            try:
                new_time_column = "date_pmgd_" + dt.now().strftime("%Y%m%d")
                cf_columns = [c.casefold() for c in df.columns.to_list()]
                if (
                    (time_column.lower() in ["year", "month", "day"])
                    and (len(set(("year", "month", "day")).intersection(set(cf_columns))) > 1)
                    and qs.checkbox(label='Merge "year", "month", "day" columns?')
                ):
                    src_columns = [c for c in df.columns.to_list()]
                    if "year".casefold() in cf_columns:
                        date_series = df[
                            src_columns[cf_columns.index("year".casefold())]
                        ].astype("str")
                    else:
                        date_series = dt.now().strftime("%Y")
                    if "month".casefold() in cf_columns:
                        date_series = (
                            date_series
                            + "-"
                            + df[src_columns[cf_columns.index("month".casefold())]].astype(
                                "str"
                            )
                        )
                    else:
                        date_series = date_series + "-01"
                    if "day".casefold() in cf_columns:
                        date_series = (
                            date_series
                            + "-"
                            + df[src_columns[cf_columns.index("day".casefold())]].astype("str")
                        )
                    else:
                        date_series = date_series + "-01"
                    df[new_time_column] = pd.to_datetime(date_series)
                else:
                    df[new_time_column] = pd.to_datetime(df[time_column])
                time_column = new_time_column
            except Exception as e:
                qs.error(
                    f"Unable to set {time_column} as time reference column because {repr(e)}"
                )
                time_column = PICK_ONE
            else:
                df = df.sort_values([time_column])
        if time_column == PICK_ONE:
            qs.warning("""Time column needed for animations.""")
            return
        anim_cat_group = qs.selectbox(
            label="Animation category group", options=[NONE_SELECTED] + cat_columns, index=0
        )
    else:
        time_column = PICK_ONE

    # Select type
    allowed_plots = [
        PLOT_SCATTER,
        PLOT_BAR,
        PLOT_HISTOGRAM,
        PLOT_VIOLIN,
        PLOT_BOX,
        PLOT_DENSITY_HEATMAP,
        PLOT_DENSITY_CONTOUR,
        PLOT_PCA_2D,
    ]
    if plot_mode == "Static":
        allowed_plots.extend(
            [
                PLOT_LINE,
                PLOT_PARALLEL_CATEGORIES,
                PLOT_PARALLEL_COORDINATES,
                PLOT_SCATTER_MATRIX,
                PLOT_PCA_3D,
                PLOT_CORR_MATRIX,
            ]
        )
    plot_type = qs.selectbox(label="Plot type: ", options=allowed_plots, index=0,)

    # Customize X axis
    if plot_type in PLOT_HAS_X:
        if plot_type in [PLOT_SCATTER, PLOT_LINE]:
            x_columns = [PICK_ONE] + all_columns
            y_columns = [PICK_ONE] + all_columns
        elif plot_type in [PLOT_BAR]:
            x_columns = [PICK_ONE] + cat_columns
            y_columns = [PICK_ONE] + all_columns
        elif plot_type in [PLOT_BOX, PLOT_VIOLIN]:
            x_columns = [NONE_SELECTED] + cat_columns
            y_columns = [PICK_ONE] + all_columns
        elif plot_type == PLOT_HISTOGRAM:
            x_columns = [PICK_ONE] + all_columns
            y_columns = [PICK_ONE] + all_columns
        elif plot_type in [PLOT_DENSITY_HEATMAP, PLOT_DENSITY_CONTOUR]:
            x_columns = [PICK_ONE] + num_columns
            y_columns = [PICK_ONE] + num_columns
        x_axis = qs.selectbox(
            label="X axis",
            options=x_columns,
            index=x_columns.index(time_column)
            if (time_column in x_columns) and (plot_mode != "Animation")
            else 0,
        )
        if (
            show_advanced_settings
            and x_axis in num_columns
            and plot_type
            not in [PLOT_PARALLEL_CATEGORIES, PLOT_PARALLEL_COORDINATES, PLOT_SCATTER_MATRIX]
        ):
            log_x = qs.checkbox(label="Log X axis?")
        else:
            log_x = False
        if x_axis == PICK_ONE:
            st.warning("""Please pic a column for the X AXIS.""")
            return
    else:
        x_axis = NONE_SELECTED

    if plot_type == PLOT_HISTOGRAM:
        hist_modes = ["count", "sum", "avg", "min", "max"]
        hist_func = qs.selectbox(
            label="Histogram function",
            options=hist_modes,
            format_func=lambda x: {
                "count": "Count",
                "sum": "Sum",
                "avg": "Average",
                "min": "Minimum",
                "max": "Maximum",
            }.get(x, "Unknown histogram mode"),
        )
        y_axis = None
    elif plot_type in PLOT_HAS_Y:
        # Customize Y axis
        y_axis = qs.selectbox(label="Y axis", options=y_columns, index=0)
        if (
            show_advanced_settings
            and y_axis in num_columns
            and plot_type
            not in [PLOT_PARALLEL_CATEGORIES, PLOT_PARALLEL_COORDINATES, PLOT_SCATTER_MATRIX]
        ):
            log_y = qs.checkbox(label="Log Y axis?")
        else:
            log_y = False
        if y_axis == PICK_ONE:
            st.warning("""Please pic a column for the X AXIS.""")
            return
    else:
        y_axis = NONE_SELECTED

    # Color column
    if plot_type in PLOT_HAS_COLOR:
        color = qs.selectbox(
            label="Use this column for color:", options=[NONE_SELECTED] + cat_columns, index=0,
        )

    # Advanced settings
    dot_text = NONE_SELECTED
    dot_size = NONE_SELECTED
    max_dot_size = 60
    facet_column = NONE_SELECTED
    facet_colum_wrap = 4
    facet_row = NONE_SELECTED
    log_x = False
    log_y = False
    template = "ggplot2"
    marginals = None
    orientation = "v"
    bar_mode = "relative"
    marginal_x = NONE_SELECTED
    marginal_y = NONE_SELECTED
    points = "outliers"
    notches = False
    violin_mode = "group"
    n_bins_x = None
    n_bins_y = None
    fill_contours = False
    trend_line = NONE_SELECTED
    symbol = NONE_SELECTED
    add_boxes = False
    show_loadings = False
    corr_method = "pearson"
    matrix_diag = "Histogram"
    matrix_up = "Scatter"
    matrix_down = "Scatter"

    if show_advanced_settings:
        # Common data
        available_marginals = [NONE_SELECTED, "rug", "box", "violin", "histogram"]
        # Dot text
        if plot_type in PLOT_HAS_TEXT:
            dot_text = qs.selectbox(
                label="Text display column", options=[NONE_SELECTED] + cat_columns, index=0
            )
        # Dot size
        if plot_type in PLOT_HAS_SIZE:
            dot_size = qs.selectbox(
                label="Use this column to select what dot size represents:",
                options=[NONE_SELECTED] + num_columns,
                index=0,
            )
            max_dot_size = qs.number_input(
                label="Max dot size", min_value=11, max_value=100, value=60
            )
        if plot_type in PLOT_HAS_SHAPE:
            symbol = qs.selectbox(
                label="Select a column for the dot symbols",
                options=[NONE_SELECTED] + cat_columns,
                index=0,
            )
        if plot_type in PLOT_HAS_TREND_LINE:
            trend_line = qs.selectbox(
                label="Trend line mode",
                options=[NONE_SELECTED, "ols", "lowess"],
                format_func=lambda x: {
                    "ols": "Ordinary Least Squares ",
                    "lowess": "Locally Weighted Scatterplot Smoothing",
                }.get(x, x),
            )

        # Facet
        if plot_type in PLOT_HAS_FACET:
            facet_column = qs.selectbox(
                label="Use this column to split the plot in columns:",
                options=[NONE_SELECTED] + cat_columns,
                index=0,
            )
            if facet_column != NONE_SELECTED:
                facet_colum_wrap = qs.number_input(
                    label="Wrap columns when more than x", min_value=1, max_value=20, value=4
                )
            else:
                facet_colum_wrap = 4
            facet_row = qs.selectbox(
                label="Use this column to split the plot in lines:",
                options=[NONE_SELECTED] + cat_columns,
                index=0,
            )
        # Histogram specific parameters
        if plot_type in PLOT_HAS_MARGINAL:
            marginals = qs.selectbox(label="Marginals", options=available_marginals, index=0)
            orientation = qs.selectbox(
                label="orientation",
                options=["v", "h"],
                format_func=lambda x: {"v": "vertical", "h": "horizontal"}.get(
                    x, "unknown value"
                ),
                index=0,
            )
        if plot_type in PLOT_HAS_BAR_MODE:
            bar_mode = qs.selectbox(
                label="bar mode", options=["group", "overlay", "relative"], index=2
            )
        if plot_type in PLOT_HAS_MARGINAL_XY:
            marginal_x = qs.selectbox(
                label="Marginals for X axis", options=available_marginals, index=0
            )
            marginal_y = qs.selectbox(
                label="Marginals for Y axis", options=available_marginals, index=0
            )
        # Box plots and histograms
        if plot_type in PLOT_HAS_POINTS:
            points = qs.selectbox(
                label="Select which points are displayed",
                options=["none", "outliers", "all"],
                index=1,
            )
            points = points if points != "none" else False
        # Box plots
        if plot_type == PLOT_BOX:
            notches = qs.checkbox(label="Use notches?", value=False)
        # Violin plots
        if plot_type == PLOT_VIOLIN:
            add_boxes = qs.checkbox(label="Show boxes", value=False)
            violin_mode = qs.selectbox("Violin display mode", options=["group", "overlay"])
        # Density heat map
        if plot_type in PLOT_HAS_BINS:
            n_bins_x = qs.number_input(
                label="Number of bins in the X axis", min_value=1, max_value=1000, value=20
            )
            n_bins_y = qs.number_input(
                label="Number of bins in the Y axis", min_value=1, max_value=1000, value=20
            )
        # Density contour map
        if plot_type == PLOT_DENSITY_CONTOUR:
            fill_contours = qs.checkbox(label="Fill contours", value=False)
        # PCA loadings
        if plot_type == PLOT_PCA_2D:
            show_loadings = qs.checkbox(label="Show loadings", value=False)
        # Correlation plot
        if plot_type == PLOT_CORR_MATRIX:
            corr_method = qs.selectbox(
                label="Correlation method",
                options=["pearson", "kendall", "spearman"],
                format_func=lambda x: {
                    "pearson": "pearson : standard correlation coefficient",
                    "kendall": "kendall : Kendall Tau correlation coefficient",
                    "spearman": "spearman : Spearman rank correlation",
                }.get(x),
            )
        # Matrix plot
        if plot_type == PLOT_SCATTER_MATRIX:
            matrix_diag = qs.selectbox(
                label="diagonal", options=["Nothing", "Histogram", "Scatter"], index=1,
            )
            matrix_up = qs.selectbox(
                label="Upper triangle", options=["Nothing", "Scatter", "2D histogram"], index=1,
            )
            matrix_down = qs.selectbox(
                label="Lower triangle", options=["Nothing", "Scatter", "2D histogram"], index=1,
            )

    height = int(
        qs.selectbox(
            label="Plot resolution",
            options=[
                "400",
                "600",
                "700",
                "800",
                "900",
                "1000",
                "1200",
                "1400",
                "1600",
                "1800",
                "2000",
            ],
            index=4,
        )
    )
    # Template
    available_templates = list(pio.templates.keys())
    template = qs.selectbox(
        label="Plot template: ",
        options=available_templates,
        index=available_templates.index(pio.templates.default),
    )

    st.header(f"Plot - {plot_type}")
    if plot_mode == "Animation" and not st.button("Render"):
        st.info(
            """Since animations may tak long to initialize, 
            the rendering is started on when you click on the render button"""
        )
        return

    common_dict = {}
    if plot_type in PLOT_HAS_X:
        common_dict["x"] = filter_none(x_axis)
    if plot_type in PLOT_HAS_Y:
        common_dict["y"] = filter_none(y_axis)
    if plot_type in PLOT_HAS_COLOR:
        common_dict["color"] = filter_none(color)
    if plot_type in PLOT_HAS_TEXT:
        common_dict["text"] = filter_none(dot_text)
    if plot_mode == "Animation":
        if (
            time_column
            in df.select_dtypes(
                include=[np.datetime64, "datetime", "datetime64", "datetime64[ns, UTC]"]
            ).columns.to_list()
        ):
            df = df.assign(time_step=(df[time_column] - df[time_column].min()).dt.days)
            afc = "time_step"
        else:
            afc = time_column
        common_dict["animation_frame"] = afc
        common_dict["animation_group"] = anim_cat_group
        if plot_type not in [PLOT_PCA_3D, PLOT_PCA_2D]:
            common_dict["range_x"] = (
                None if x_axis not in num_columns else [df[x_axis].min(), df[x_axis].max()]
            )
            common_dict["range_y"] = (
                None if y_axis not in num_columns else [df[y_axis].min(), df[y_axis].max()]
            )
    common_dict["data_frame"] = df
    if plot_type in PLOT_HAS_FACET:
        common_dict = {
            **common_dict,
            "facet_row": filter_none(facet_row),
            "facet_col": filter_none(facet_column),
            "facet_col_wrap": facet_colum_wrap,
        }
    if plot_type in PLOT_HAS_LOG:
        common_dict = {**common_dict, "log_x": log_x, "log_y": log_y}
    if plot_type in PLOT_HAS_MARGINAL_XY:
        common_dict = {
            **common_dict,
            "marginal_x": filter_none(marginal_x),
            "marginal_y": filter_none(marginal_y),
        }
    if plot_type in PLOT_HAS_MARGINAL:
        common_dict["marginals"] = filter_none(marginals)
    if plot_type in PLOT_HAS_BAR_MODE:
        common_dict["barmode"] = bar_mode
    if plot_type in PLOT_HAS_SIZE:
        common_dict["size"] = filter_none(dot_size)
        common_dict["size_max"] = max_dot_size
    if plot_type in PLOT_HAS_SHAPE:
        common_dict["symbol"] = filter_none(symbol)
    if plot_type in PLOT_HAS_TREND_LINE:
        common_dict["trendline"] = filter_none(trend_line)
    if plot_type in PLOT_HAS_POINTS:
        common_dict["points"] = points
    if plot_type in PLOT_HAS_BINS:
        common_dict["nbinsx"] = n_bins_x
        common_dict["nbinsy"] = n_bins_y

    if plot_type == PLOT_SCATTER:
        fig = px.scatter(**common_dict)
    elif plot_type == PLOT_LINE:
        fig = px.line(**common_dict)
    elif plot_type == PLOT_BAR:
        fig = px.bar(**common_dict)
    elif plot_type == PLOT_HISTOGRAM:
        if orientation == "h":
            common_dict["x"] = None
            common_dict["y"] = x_axis
        fig = px.histogram(**common_dict, histfunc=hist_func, orientation=orientation,)
    elif plot_type == PLOT_BOX:
        fig = px.box(**common_dict, notched=notches)
    elif plot_type == PLOT_VIOLIN:
        fig = px.violin(**common_dict, box=add_boxes, violinmode=violin_mode)
    elif plot_type == PLOT_DENSITY_HEATMAP:
        fig = px.density_heatmap(**common_dict)
    elif plot_type == PLOT_DENSITY_CONTOUR:
        fig = px.density_contour(**common_dict)
        if fill_contours:
            fig.update_traces(contours_coloring="fill", contours_showlabels=True)
    elif plot_type == PLOT_PARALLEL_CATEGORIES:
        fig = px.parallel_categories(**common_dict)
    elif plot_type == PLOT_PARALLEL_COORDINATES:
        fig = px.parallel_coordinates(**common_dict)
    elif plot_type == PLOT_SCATTER_MATRIX:
        fig = make_subplots(
            rows=len(num_columns),
            cols=len(num_columns),
            shared_xaxes=True,
            row_titles=num_columns,
        )
        if (color_column := filter_none(color)) is not None:
            template_colors = pio.templates[template].layout["colorway"]
            if template_colors is None:
                template_colors = pio.templates[pio.templates.default].layout["colorway"]
            color_count = len(df[color_column].unique())
            if len(template_colors) >= color_count:
                pass
            else:
                template_colors = np.repeat(
                    template_colors, (color_count // len(template_colors)) + 1
                )
            template_colors = template_colors[:color_count]
        else:
            template_colors = 0
        for i, c in enumerate(num_columns):
            for j, l in enumerate(num_columns):
                if i == j:
                    if matrix_diag == "Nothing":
                        continue
                    elif matrix_diag == "Histogram":
                        mtx_plot_kind = "Histogram"
                    else:
                        mtx_plot_kind = "Scatter"
                else:
                    if (
                        (i == j)
                        or (i > j and matrix_up == "Scatter")
                        or (i < j and matrix_down == "Scatter")
                    ):
                        mtx_plot_kind = "Scatter"
                    elif (i > j and matrix_up == "Nothing") or (
                        i < j and matrix_down == "Nothing"
                    ):
                        continue
                    elif (i > j and matrix_up == "2D histogram") or (
                        i < j and matrix_down == "2D histogram"
                    ):
                        mtx_plot_kind = "2D histogram"
                    else:
                        mtx_plot_kind = "Error"

                if isinstance(template_colors, int) or mtx_plot_kind == "2D histogram":
                    if mtx_plot_kind == "Histogram":
                        add_histogram(fig=fig, x=df[c], index=i + 1)
                    elif mtx_plot_kind == "Scatter":
                        add_scatter(
                            fig=fig, x=df[c], y=df[l], row=j + 1, col=i + 1,
                        )
                    elif mtx_plot_kind == "2D histogram":
                        add_2d_hist(fig=fig, x=df[c], y=df[l], row=j + 1, col=i + 1)
                else:
                    for color_parse, cat in zip(template_colors, df[color_column].unique()):
                        df_cat = df[df[color_column] == cat]
                        if mtx_plot_kind == "Histogram":
                            add_histogram(
                                fig=fig, x=df_cat[c], index=i + 1, name=cat, marker=color_parse,
                            )
                        elif mtx_plot_kind == "Scatter":
                            add_scatter(
                                fig=fig,
                                x=df_cat[c],
                                y=df_cat[l],
                                row=j + 1,
                                col=i + 1,
                                marker=color_parse,
                            )
                fig.update_xaxes(
                    title_text=c, row=j + 1, col=i + 1,
                )
                if c == 0:
                    fig.update_yaxes(
                        title_text=l, row=j + 1, col=i + 1,
                    )
        fig.update_layout(barmode="stack")
    elif plot_type in [PLOT_PCA_2D, PLOT_PCA_3D]:
        X = df.loc[:, num_columns]
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        pca = PCA()
        x_new = pca.fit_transform(X)
        x_pca = x_new[:, 0]
        y_pca = x_new[:, 1]
        pc1_lbl = f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)"
        pc2_lbl = f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)"
        df[pc1_lbl] = x_pca * (1.0 / (x_pca.max() - x_pca.min()))
        df[pc2_lbl] = y_pca * (1.0 / (y_pca.max() - y_pca.min()))
        common_dict["x"] = pc1_lbl
        common_dict["y"] = pc2_lbl
        if plot_mode == "Animation":
            common_dict["range_x"] = [-1, 1]
            common_dict["range_y"] = [-1, 1]
        coeff = np.transpose(pca.components_[0:2, :])
        if plot_type == PLOT_PCA_3D:
            z_pca = x_new[:, 2]
            pc3_lbl = f"PC3 ({pca.explained_variance_ratio_[2] * 100:.2f}%)"
            df[pc3_lbl] = z_pca * (1.0 / (z_pca.max() - z_pca.min()))
            common_dict["z"] = pc3_lbl
            fig = px.scatter_3d(**common_dict)
        else:
            fig = px.scatter(**common_dict)
            if show_loadings:
                for i in range(coeff.shape[0]):
                    fig.add_shape(
                        type="line",
                        opacity=0.5,
                        x0=0,
                        y0=0,
                        x1=coeff[i, 0],
                        y1=coeff[i, 1],
                        line=dict(width=2),
                    )
                fig.add_trace(
                    go.Scatter(
                        x=coeff[:, 0],
                        y=coeff[:, 1],
                        mode="markers+text",
                        text=num_columns,
                        opacity=0.8,
                        name="Loadings",
                    ),
                )
    elif plot_type == PLOT_CORR_MATRIX:
        fig = px.imshow(
            df[num_columns].corr(method=corr_method).values, x=num_columns, y=num_columns
        )
    else:
        fig = None

    if fig is not None:
        fig.update_layout(height=height, template=template)
        st.plotly_chart(figure_or_data=fig, use_container_width=True)
        if (
            plot_type in [PLOT_PCA_2D, PLOT_PCA_3D]
            and plot_mode != "Animation"
            and st.checkbox(label="Show explained variance", value=False)
        ):
            df_ev = pd.DataFrame.from_dict(
                {
                    "pc": [f"PC{i}" for i in range(len(pca.explained_variance_ratio_))],
                    "exp_var_per": pca.explained_variance_ratio_ * 100,
                }
            )
            df_ev = df_ev.assign(cumulative=df_ev["exp_var_per"].cumsum())
            ev_fig = go.Figure()
            ev_fig.add_trace(go.Bar(x=df_ev["pc"], y=df_ev["exp_var_per"], name="individual",))
            ev_fig.add_trace(
                go.Scatter(x=df_ev["pc"], y=df_ev["cumulative"], name="cumulative",)
            )
            ev_fig.update_layout(
                height=height,
                title="Explained variance by different principal components",
                xaxis_title="Principal component",
                yaxis_title="Explained variance in percent",
            )
            st.plotly_chart(figure_or_data=ev_fig, use_container_width=True)


customize_plot()
