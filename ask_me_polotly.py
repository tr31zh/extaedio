import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from sklearn import datasets

from dw_cssegis_data import get_wrangled_cssegis_df

PICK_ONE = "Pick one..."

st.title("Ask me polotly")

st.markdown(
    """After starting with one of the worsts attempts at word play in the 
history of humanity, this page will ask you a series of questions in order to 
build a visualization."""
)

use_side_bar = st.checkbox(
    label="Put to questions to customize the plot in th sidebar? Recommended", value=True
)
show_advanced_settings = st.checkbox(label="Show customization advanced options", value=False)

url_csse = "url_csse"
url_covid_data = "https://raw.githubusercontent.com/coviddata/coviddata/master/data/sources/jhu_csse/standardized/standardized.csv"
url_ecdc_covid = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
url_iris = "url_iris"


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


_max_width_()


def format_csv_link(url):
    if url == url_csse:
        return "COVID-19 - CSSE"
    elif url == url_ecdc_covid:
        return "COVID-19 - European Centre for Disease Prevention and Control"
    elif url == url_covid_data:
        return "COVID-19 - Github coviddata"
    elif url == url_iris:
        return "Classic Iris data set"
    else:
        return url


@st.cache
def get_datafrmae(url):
    if url == PICK_ONE:
        return None
    elif isinstance(url, io.StringIO):
        return pd.read_csv(url)
    elif url == url_csse:
        return get_wrangled_cssegis_df(allow_cache=True)
    elif url == url_iris:
        iris = datasets.load_iris()
        df = pd.DataFrame(
            data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
        )
        df = (
            df.assign(species=df.target.replace(dict(enumerate(iris.target_names))))
            .assign(sepal_lenght=df["sepal length (cm)"])
            .assign(sepal_width=df["sepal width (cm)"])
            .assign(petal_lenght=df["petal length (cm)"])
            .assign(petal_width=df["petal width (cm)"])
            .drop(
                columns=[
                    "sepal length (cm)",
                    "sepal width (cm)",
                    "petal length (cm)",
                    "petal width (cm)",
                ]
            )
        )
        return df
    elif isinstance(url, str):
        return pd.read_csv(url)
    else:
        return None


def customize_plot():

    selected_file = st.selectbox(
        label="Source file: ",
        options=[
            PICK_ONE,
            "Load local file",
            url_csse,
            url_covid_data,
            url_ecdc_covid,
            url_iris,
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

    st.header("Selected data frame")
    max_dot_size = st.number_input(
        label="Lines to display", min_value=5, max_value=1000, value=5
    )
    st.dataframe(df.head(max_dot_size))

    st.header("Detect time/date column")
    usual_time_columns = ["date", "date_time", "datetime", "timestamp", "time", "daterep"]
    time_columns = [col for col in df.columns.to_list() if col.lower() in usual_time_columns]
    if not time_columns:
        time_columns = [PICK_ONE]
    time_columns.extend(
        [col for col in df.columns.to_list() if col.lower() not in time_columns]
    )
    time_column = st.selectbox(label="Date/time column: ", options=time_columns, index=0,)
    if time_column != PICK_ONE:
        try:
            df[time_column] = pd.to_datetime(df[time_column])
        except Exception as e:
            st.error(f"Unable to set {time_column} as time reference column")
            time_column = None
        else:
            st.info("First timestamps")
            df = df.sort_values([time_column])
            st.write([str(i) for i in df[time_column].to_list()[:5]])

    num_columns = df.select_dtypes(include=[np.number]).columns.to_list()
    cat_columns = df.select_dtypes(include=["object", "datetime"]).columns.to_list()

    # Filter
    st.header("Filtering")
    filter_columns = st.multiselect(
        label="Select which columns you want to filter:", options=cat_columns, default=None,
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

    # Preview dataframe
    if len(filters) > 0:
        for k, v in filters.items():
            df = df[df[k].isin(v)]
        st.subheader("filtered data frame")
        st.dataframe(df.describe())

    qs = st.sidebar if use_side_bar else st
    if use_side_bar:
        st.info("The rest of the questions are in the side bar")
    else:
        st.header("Plot customization")

    # Select mode
    if time_column == PICK_ONE:
        plot_mode = "Static"
    else:
        plot_modes = [PICK_ONE, "Static", "Animation"]
        plot_mode = qs.selectbox(label="Plot mode: ", options=plot_modes, index=0,)
        if plot_mode == PICK_ONE:
            st.warning("""Please pic a PLOT MODE.""")
            return

    # Select type
    allowed_plots = [PICK_ONE, "scatter", "bar", "histogram"]
    if plot_mode == "Static":
        allowed_plots.append("line")
    plot_type = qs.selectbox(label="Plot type: ", options=allowed_plots, index=0,)
    if plot_type == PICK_ONE:
        st.warning("""Please pic a PLOT TYPE.""")
        return

    # Customize X axis
    if plot_type in ["scatter", "line"]:
        x_columns = [PICK_ONE] + df.columns.to_list()
    elif plot_type == "bar":
        x_columns = [PICK_ONE] + cat_columns
    elif plot_type == "histogram":
        x_columns = [PICK_ONE] + df.columns.to_list()
    else:
        st.warning("""Unknown  plot type.""")
        return
    x_axis = qs.selectbox(
        label="X axis",
        options=x_columns,
        index=x_columns.index(time_column) if time_column != PICK_ONE in x_columns else 0,
    )
    if show_advanced_settings and x_axis in num_columns:
        log_x = qs.checkbox(label="Log X axis?")
    if x_axis == PICK_ONE:
        st.warning("""Please pic a column for the X AXIS.""")
        return

    if plot_type == "histogram":
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
    else:
        # Customize Y axis
        y_axis = qs.selectbox(
            label="Y axis", options=[PICK_ONE] + df.columns.to_list(), index=0
        )
        if show_advanced_settings and y_axis in num_columns:
            log_y = qs.checkbox(label="Log Y axis?")
        if y_axis == PICK_ONE:
            st.warning("""Please pic a column for the X AXIS.""")
            return

    # Color column
    color = qs.selectbox(
        label="Use this column for color separation:",
        options=[PICK_ONE] + cat_columns,
        index=0,
    )

    if show_advanced_settings:
        # Dot size
        if plot_type in [
            "scatter",
        ]:
            dot_size = qs.selectbox(
                label="Use this column to select what dot size represents:",
                options=[PICK_ONE] + num_columns,
                index=0,
            )
            max_dot_size = qs.number_input(
                label="Max dot size", min_value=11, max_value=100, value=60
            )

        # Facet
        facet_column = qs.selectbox(
            label="Use this column to split the plot in columns:",
            options=[PICK_ONE] + cat_columns,
            index=0,
        )
        if facet_column != PICK_ONE:
            facet_colum_wrap = qs.number_input(
                label="Wrap columns when more than x", min_value=1, max_value=20, value=4
            )
        else:
            facet_colum_wrap = 4
        facet_row = qs.selectbox(
            label="Use this column to split the plot in lines:",
            options=[PICK_ONE] + cat_columns,
            index=0,
        )
        # Histogram specific parameters
        if plot_type == "histogram":
            available_marginals = ["none", "rug", "box", "violin", "histogram"]
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
            index=0,
        )
    else:
        dot_size = PICK_ONE
        max_dot_size = 60
        facet_column = PICK_ONE
        facet_colum_wrap = 4
        facet_row = PICK_ONE
        log_x = False
        log_y = False
        template = "ggplot2"
        marginals = None
        orientation = "v"
        bar_mode = "relative"

    size = qs.selectbox(
        label="Plot resolution",
        options=["800x600", "1024x768", "1200x800", "1920x1080"],
        index=2,
    )
    width, height = [int(x) for x in size.split("x")]

    st.header("Plot")
    if plot_mode == "Animation" and not st.button("Render"):
        st.info(
            """Since animations may tak long to initialize, 
            the rendering is started on when you click on the render button"""
        )
        return
    common_dict = {
        "x": x_axis,
        "y": y_axis,
        "color": color if color != PICK_ONE else None,
        "width": width,
        "height": height,
        "template": template,
    }
    if plot_mode == "Animation":
        df = df.assign(time_step=(df[time_column] - df[time_column].min()).dt.days)
        common_dict = {
            **common_dict,
            "animation_frame": "time_step",
            "range_x": [
                None
                if x_axis not in num_columns
                else max(1, df[x_axis].min())
                if log_x and (plot_type != "bar")
                else df[x_axis].min(),
                df[x_axis].max(),
            ],
            "range_y": [
                None
                if y_axis not in num_columns
                else max(1, df[y_axis].min())
                if log_y and (plot_type != "bar")
                else df[y_axis].min(),
                df[y_axis].max(),
            ],
        }
    common_dict = {
        **common_dict,
        "data_frame": df,
        "facet_row": facet_row if facet_row != PICK_ONE else None,
        "facet_col": facet_column if facet_column != PICK_ONE else None,
        "facet_col_wrap": facet_colum_wrap,
    }

    if plot_type == "scatter":
        fig = px.scatter(
            **common_dict,
            size=dot_size if dot_size != PICK_ONE else None,
            size_max=max_dot_size,
        )
    elif plot_type == "line":
        fig = px.line(**common_dict)
    elif plot_type == "bar":
        fig = px.bar(**common_dict)
    elif plot_type == "histogram":
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
    else:
        fig = None

    if fig is not None:
        st.plotly_chart(figure_or_data=fig)


customize_plot()
