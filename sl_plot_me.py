import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from sklearn import datasets

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
def get_datafrmae(url):
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


st.title("Plot me Gently Dirk - streamlit version")

st.markdown(
    """After starting with one of the worsts attempts at a pun in the 
history of humanity, this page will ask you a series of questions in order to 
build a visualization."""
)

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
show_advanced_settings = st.checkbox(label="Show customization advanced options", value=False)


def filter_none(value):
    return None if value == NONE_SELECTED else value


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
            if selected_file != PICK_ONE:
                st.error("Unknown file loaded")
            return
    df_loaded = get_datafrmae(url=selected_file)
    if df_loaded is None:
        return
    df = df_loaded.copy()

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

        num_columns = df.select_dtypes(include=[np.number]).columns.to_list()
        cat_columns = df.select_dtypes(include=["object", "datetime"]).columns.to_list()

        st.subheader("Filter rows")
        filter_columns = st.multiselect(
            label="Select which columns you want to use to filter the rows:",
            options=cat_columns,
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

        df = df.reset_index()

        # Preview dataframe
        st.subheader("filtered data frame")
        st.dataframe(df.describe())

    num_columns = df.select_dtypes(include=[np.number]).columns.to_list()
    cat_columns = df.select_dtypes(include=["object", "datetime"]).columns.to_list()

    st.header("Detect time/date column")
    usual_time_columns = ["date", "date_time", "datetime", "timestamp", "time", "daterep"]
    time_columns = [col for col in df.columns.to_list() if col.lower() in usual_time_columns]
    if not time_columns:
        time_columns = [NONE_SELECTED]
    time_columns.extend(
        [col for col in df.columns.to_list() if col.lower() not in time_columns]
    )
    time_column = st.selectbox(label="Date/time column: ", options=time_columns, index=0,)
    if time_column != NONE_SELECTED:
        try:
            df[time_column] = pd.to_datetime(df[time_column])
        except Exception as e:
            st.error(f"Unable to set {time_column} as time reference column")
            time_column = NONE_SELECTED
        else:
            st.info("First timestamps")
            df = df.sort_values([time_column])
            st.write([str(i) for i in df[time_column].to_list()[:5]])

    qs = st.sidebar if use_side_bar else st
    if use_side_bar:
        pass
    else:
        st.header("Plot customization")

    # Select mode
    if time_column == NONE_SELECTED:
        plot_mode = "Static"
    else:
        plot_modes = [PICK_ONE, "Static", "Animation"]
        plot_mode = qs.selectbox(label="Plot mode: ", options=plot_modes, index=0,)
        if plot_mode == PICK_ONE:
            st.warning("""Please pic a PLOT MODE.""")
            return

    # Select type
    allowed_plots = [
        PICK_ONE,
        PLOT_SCATTER,
        PLOT_BAR,
        PLOT_HISTOGRAM,
        PLOT_VIOLIN,
        PLOT_BOX,
        PLOT_DENSITY_HEATMAP,
        PLOT_DENSITY_CONTOUR,
        PLOT_PARALLEL_CATEGORIES,
        PLOT_PARALLEL_COORDINATES,
        PLOT_SCATTER_MATRIX,
    ]
    if plot_mode == "Static":
        allowed_plots.append(PLOT_LINE)
    plot_type = qs.selectbox(label="Plot type: ", options=allowed_plots, index=0,)
    if plot_type == PICK_ONE:
        st.warning("""Please pic a PLOT TYPE.""")
        return

    # Customize X axis
    if plot_type in [PLOT_SCATTER, PLOT_LINE]:
        x_columns = [PICK_ONE] + df.columns.to_list()
        y_columns = [PICK_ONE] + df.columns.to_list()
    elif plot_type in [PLOT_BAR]:
        x_columns = [PICK_ONE] + cat_columns
        y_columns = [PICK_ONE] + df.columns.to_list()
    elif plot_type in [PLOT_BOX, PLOT_VIOLIN]:
        x_columns = [NONE_SELECTED] + cat_columns
        y_columns = [PICK_ONE] + df.columns.to_list()
    elif plot_type == PLOT_HISTOGRAM:
        x_columns = [PICK_ONE] + df.columns.to_list()
        y_columns = [PICK_ONE] + df.columns.to_list()
    elif plot_type in [PLOT_DENSITY_HEATMAP, PLOT_DENSITY_CONTOUR]:
        x_columns = [PICK_ONE] + num_columns
        y_columns = [PICK_ONE] + num_columns
    elif plot_type in [
        PLOT_PARALLEL_CATEGORIES,
        PLOT_PARALLEL_COORDINATES,
        PLOT_SCATTER_MATRIX,
    ]:
        x_columns = []
        y_columns = []
    else:
        st.warning("""Unknown  plot type.""")
        return
    if plot_type not in [
        PLOT_PARALLEL_CATEGORIES,
        PLOT_PARALLEL_COORDINATES,
        PLOT_SCATTER_MATRIX,
    ]:
        x_axis = qs.selectbox(
            label="X axis",
            options=x_columns,
            index=x_columns.index(time_column)
            if time_column != NONE_SELECTED in x_columns
            else 0,
        )
        if (
            show_advanced_settings
            and x_axis in num_columns
            and plot_type
            in [PLOT_SCATTER, PLOT_LINE, PLOT_DENSITY_HEATMAP, PLOT_DENSITY_CONTOUR]
        ):
            log_x = qs.checkbox(label="Log X axis?")
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
    elif plot_type in [
        PLOT_PARALLEL_CATEGORIES,
        PLOT_PARALLEL_COORDINATES,
        PLOT_SCATTER_MATRIX,
    ]:
        y_axis = NONE_SELECTED
    else:
        # Customize Y axis
        y_axis = qs.selectbox(label="Y axis", options=y_columns, index=0)
        if (
            show_advanced_settings
            and y_axis in num_columns
            and plot_type
            in [PLOT_SCATTER, PLOT_LINE, PLOT_DENSITY_HEATMAP, PLOT_DENSITY_CONTOUR]
        ):
            log_y = qs.checkbox(label="Log Y axis?")
        if y_axis == PICK_ONE:
            st.warning("""Please pic a column for the X AXIS.""")
            return

    # Color column
    if plot_type not in [
        PLOT_DENSITY_HEATMAP,
    ]:
        color = qs.selectbox(
            label="Use this column for color separation:",
            options=[NONE_SELECTED] + cat_columns,
            index=0,
        )

    if (
        plot_type
        not in [PLOT_PARALLEL_CATEGORIES, PLOT_PARALLEL_COORDINATES, PLOT_SCATTER_MATRIX]
        and show_advanced_settings
    ):
        # Common data
        available_marginals = [NONE_SELECTED, "rug", "box", "violin", "histogram"]
        # Dot size
        if plot_type == PLOT_SCATTER:
            dot_size = qs.selectbox(
                label="Use this column to select what dot size represents:",
                options=[NONE_SELECTED] + num_columns,
                index=0,
            )
            max_dot_size = qs.number_input(
                label="Max dot size", min_value=11, max_value=100, value=60
            )
            symbol = qs.selectbox(
                label="Select a column for the dot symbols",
                options=[NONE_SELECTED] + cat_columns,
                index=0,
            )
            trend_line = qs.selectbox(
                label="Trend line mode",
                options=[NONE_SELECTED, "ols", "lowess"],
                format_func=lambda x: {
                    "ols": "Ordinary Least Squares ",
                    "lowess": "Locally Weighted Scatterplot Smoothing",
                }.get(x, x),
            )

        # Facet
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
        if plot_type == PLOT_HISTOGRAM:
            marginals = qs.selectbox(label="Marginals", options=available_marginals, index=0)
            orientation = qs.selectbox(
                label="orientation",
                options=["v", "h"],
                format_func=lambda x: {"v": "vertical", "h": "horizontal"}.get(
                    x, "unknown value"
                ),
                index=0,
            )
            bar_mode = qs.selectbox(
                label="bar mode", options=["group", "overlay", "relative"], index=2
            )
        elif plot_type in [PLOT_SCATTER, PLOT_DENSITY_HEATMAP, PLOT_DENSITY_CONTOUR]:
            marginal_x = qs.selectbox(
                label="Marginals for X axis", options=available_marginals, index=0
            )
            marginal_y = qs.selectbox(
                label="Marginals for Y axis", options=available_marginals, index=0
            )
        # Box plots and histograms
        if plot_type in [PLOT_BOX, PLOT_VIOLIN]:
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
        if plot_type in [PLOT_DENSITY_HEATMAP, PLOT_DENSITY_CONTOUR]:
            n_bins_x = qs.number_input(
                label="Number of bins in the X axis", min_value=1, max_value=1000, value=20
            )
            n_bins_y = qs.number_input(
                label="Number of bins in the Y axis", min_value=1, max_value=1000, value=20
            )
        # Density contour map
        fill_contours = qs.checkbox(label="Fill contours", value=False)
    else:
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

    size = qs.selectbox(
        label="Plot resolution",
        options=["800x600", "1024x768", "1200x800", "1920x1080"],
        index=2,
    )
    width, height = [int(x) for x in size.split("x")]
    # Template
    template = qs.selectbox(
        label="Plot template: ",
        options=[
            "ggplot2",
            "seaborn",
            "simple_white",
            "plotly",
            "plotly_white",
            "plotly_dark",
            "presentation",
            "xgridoff",
            "ygridoff",
            "gridon",
            "none",
        ],
        index=4,
    )

    st.header("Plot")
    if plot_mode == "Animation" and not st.button("Render"):
        st.info(
            """Since animations may tak long to initialize, 
            the rendering is started on when you click on the render button"""
        )
        return
    common_dict = {
        "x": filter_none(x_axis),
        "y": y_axis,
        "width": width,
        "height": height,
        "template": template,
    }
    if plot_type not in [PLOT_DENSITY_HEATMAP, PLOT_PARALLEL_CATEGORIES]:
        common_dict["color"] = filter_none(color)
    if plot_mode == "Animation":
        df = df.assign(time_step=(df[time_column] - df[time_column].min()).dt.days)
        common_dict = {
            **common_dict,
            "animation_frame": "time_step",
            "range_x": [
                None
                if x_axis not in num_columns
                else max(1, df[x_axis].min())
                if log_x and (plot_type != PLOT_BAR)
                else df[x_axis].min(),
                df[x_axis].max(),
            ],
            "range_y": [
                None
                if y_axis not in num_columns
                else max(1, df[y_axis].min())
                if log_y and (plot_type != PLOT_BAR)
                else df[y_axis].min(),
                df[y_axis].max(),
            ],
        }
    common_dict = {
        **common_dict,
        "data_frame": df,
        "facet_row": filter_none(facet_row),
        "facet_col": filter_none(facet_column),
        "facet_col_wrap": facet_colum_wrap,
    }
    if plot_type in [
        PLOT_SCATTER,
        PLOT_DENSITY_HEATMAP,
        PLOT_DENSITY_CONTOUR,
        PLOT_SCATTER_MATRIX,
    ]:
        common_dict = {
            **common_dict,
            "marginal_x": filter_none(marginal_x),
            "marginal_y": filter_none(marginal_y),
        }

    if plot_type == PLOT_SCATTER:
        fig = px.scatter(
            **common_dict,
            size=filter_none(dot_size),
            size_max=max_dot_size,
            trendline=filter_none(trend_line),
            symbol=filter_none(symbol),
        )
    elif plot_type == PLOT_LINE:
        fig = px.line(**common_dict)
    elif plot_type == PLOT_BAR:
        fig = px.bar(**common_dict)
    elif plot_type == PLOT_HISTOGRAM:
        if orientation == "h":
            common_dict["x"] = None
            common_dict["y"] = x_axis
        fig = px.histogram(
            **common_dict,
            histfunc=hist_func,
            marginal=None if marginals == "none" else marginals,
            orientation=orientation,
            barmode=bar_mode,
        )
    elif plot_type == PLOT_BOX:
        fig = px.box(**common_dict, points=points, notched=notches)
    elif plot_type == PLOT_VIOLIN:
        fig = px.violin(**common_dict, points=points, box=add_boxes, violinmode=violin_mode)
    elif plot_type == PLOT_DENSITY_HEATMAP:
        fig = px.density_heatmap(**common_dict, nbinsx=n_bins_x, nbinsy=n_bins_y)
    elif plot_type == PLOT_DENSITY_CONTOUR:
        fig = px.density_contour(**common_dict, nbinsx=n_bins_x, nbinsy=n_bins_y)
        if fill_contours:
            fig.update_traces(contours_coloring="fill", contours_showlabels=True)
    elif plot_type == PLOT_PARALLEL_CATEGORIES:
        fig = px.parallel_categories(
            data_frame=df, width=width, height=height, template=template
        )
    elif plot_type == PLOT_PARALLEL_COORDINATES:
        fig = px.parallel_coordinates(
            data_frame=df, width=width, height=height, template=template
        )
    elif plot_type == PLOT_SCATTER_MATRIX:
        st.dataframe(df.head())
        fig = px.scatter_matrix(
            data_frame=df,
            width=width,
            height=height,
            template=template,
            color=filter_none(color),
        )

    else:
        fig = None

    if fig is not None:
        st.plotly_chart(figure_or_data=fig)


customize_plot()
