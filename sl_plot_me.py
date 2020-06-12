import io
import time

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

from amp_consts import (
    PICK_ONE,
    NONE_SELECTED,
    AVAILABLE_URLS,
    AVAILABLE_PLOTS,
    PLOT_SCATTER,
    PLOT_SCATTER_3D,
    PLOT_BAR,
    PLOT_HISTOGRAM,
    PLOT_VIOLIN,
    PLOT_BOX,
    PLOT_DENSITY_HEATMAP,
    PLOT_DENSITY_CONTOUR,
    PLOT_LINE,
    PLOT_PARALLEL_CATEGORIES,
    PLOT_PARALLEL_COORDINATES,
    PLOT_SCATTER_MATRIX,
    PLOT_PCA_2D,
    PLOT_PCA_3D,
    PLOT_QDA_2D,
    PLOT_LDA_2D,
    PLOT_CORR_MATRIX,
    PLOT_HAS_X,
    PLOT_HAS_Y,
    PLOT_HAS_Z,
    PLOT_HAS_COLOR,
    PLOT_HAS_TEXT,
    PLOT_HAS_BAR_MODE,
    PLOT_HAS_BINS,
    PLOT_HAS_ANIM,
    PLOT_HAS_FACET,
    PLOT_HAS_LOG,
    PLOT_HAS_MARGINAL,
    PLOT_HAS_MARGINAL_XY,
    PLOT_HAS_POINTS,
    PLOT_HAS_SHAPE,
    PLOT_HAS_SIZE,
    PLOT_HAS_TREND_LINE,
    PLOT_HAS_CUSTOM_HOVER_DATA,
    PLOT_HAS_TARGET,
    PLOT_HAS_IGNORE_COLUMNS,
)
from amp_functs import get_dataframe_from_url, format_csv_link, build_plot


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


@st.cache
def get_df_from_url(url):
    return get_dataframe_from_url(url)


st.title("Ex Taedio")

st.markdown("""Build plots the (kind of) easy way.""")

st.header("Settings")

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


def customize_plot():

    st.header("Load dataframe, usually a CSV file")
    selected_file = st.selectbox(
        label="Source file: ", options=AVAILABLE_URLS, index=0, format_func=format_csv_link,
    )
    if selected_file == "Load local file":
        selected_file = st.file_uploader(label="Select file to upload")
        if selected_file is None:
            return
    df_loaded = get_df_from_url(url=selected_file)
    if df_loaded is None:
        return
    df = df_loaded.copy().reset_index(drop=True)

    st.header("Sort columns")
    sort_columns = st.multiselect(label="Sort by", options=df.columns.to_list())
    invert_sort = st.checkbox(label="Reverse sort?", value=False)
    if sort_columns:
        df = df.sort_values(sort_columns, ascending=not invert_sort)

    if show_dw_options:
        # Filter
        st.header("Filtering")

        st.subheader("Selected data frame")
        line_display_count = st.number_input(
            label="Lines to display", min_value=5, max_value=1000, value=5
        )
        st.dataframe(df.head(line_display_count))

        st.subheader("Dataframe description")
        st.dataframe(df.describe())
        st.write(df.dtypes)

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
        st.subheader("filtered data frame numeric data description")
        st.dataframe(df.describe())

    qs = st.sidebar if use_side_bar else st
    if use_side_bar:
        pass
    else:
        st.header("Plot customization")

    num_columns = df.select_dtypes(include=[np.number]).columns.to_list()
    cat_columns = df.select_dtypes(include=["object", "datetime"]).columns.to_list()
    supervision_columns = df.select_dtypes(include=["object", "number"]).columns.to_list()
    all_columns = df.columns.to_list()
    plot_data_dict = {}

    # Select type
    plot_type = qs.selectbox(label="Plot type: ", options=AVAILABLE_PLOTS, index=0,)
    st.header(f"Plot - {plot_type}")
    if plot_type == PLOT_SCATTER:
        doc, _ = px.scatter.__doc__.split("\nParameters")
        st.write(doc.strip())
    elif plot_type == PLOT_SCATTER_3D:
        doc, _ = px.scatter_3d.__doc__.split("\nParameters")
        st.write(doc.strip())
    elif plot_type == PLOT_LINE:
        doc, _ = px.line.__doc__.split("\nParameters")
        st.write(doc.strip())
    elif plot_type == PLOT_BAR:
        doc, _ = px.bar.__doc__.split("\nParameters")
        st.write(doc.strip())
    elif plot_type == PLOT_HISTOGRAM:
        doc, _ = px.histogram.__doc__.split("\nParameters")
        st.write(doc.strip())
    elif plot_type == PLOT_BOX:
        doc, _ = px.box.__doc__.split("\nParameters")
        st.write(doc.strip())
    elif plot_type == PLOT_VIOLIN:
        doc, _ = px.violin.__doc__.split("\nParameters")
        st.write(doc.strip())
    elif plot_type == PLOT_DENSITY_HEATMAP:
        doc, _ = px.density_heatmap.__doc__.split("\nParameters")
        st.write(doc.strip())
    elif plot_type == PLOT_DENSITY_CONTOUR:
        doc, _ = px.density_contour.__doc__.split("\nParameters")
        st.write(doc.strip())
    elif plot_type == PLOT_PARALLEL_CATEGORIES:
        doc, _ = px.parallel_categories.__doc__.split("\nParameters")
        st.write(doc.strip())
    elif plot_type == PLOT_PARALLEL_COORDINATES:
        doc, _ = px.parallel_coordinates.__doc__.split("\nParameters")
        st.write(doc.strip())
    elif plot_type == PLOT_SCATTER_MATRIX:
        st.write("Plot a scatter mattrix for all selected columns")
    elif plot_type in [PLOT_PCA_2D]:
        st.write("Plot 2D PCA")
    elif plot_type in [PLOT_PCA_3D]:
        st.write("Plot 3D PCA")
    elif plot_type == PLOT_CORR_MATRIX:
        st.write("Plot correlation matrix")
    elif plot_type == PLOT_LDA_2D:
        st.write(PLOT_LDA_2D)
    elif plot_type == PLOT_QDA_2D:
        st.write(PLOT_QDA_2D + "On 2D PCA")

    # Select mode
    is_anim = plot_type in PLOT_HAS_ANIM and qs.checkbox(label="Build animation", value=False)
    if is_anim:
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
        plot_data_dict["time_column"] = qs.selectbox(
            label="Date/time column: ", options=time_columns, index=0,
        )
        if plot_data_dict["time_column"] != PICK_ONE:
            new_time_column = plot_data_dict["time_column"] + "_" + "pmgd"
        else:
            qs.warning("""Time column needed for animations.""")
            return
        if plot_data_dict["time_column"] != PICK_ONE and (
            plot_data_dict["time_column"] in usual_time_columns
            or qs.checkbox(
                label="Convert to date?",
                value=plot_data_dict["time_column"] in usual_time_columns,
            )
        ):
            try:
                cf_columns = [c.casefold() for c in df.columns.to_list()]
                if (
                    (plot_data_dict["time_column"].lower() in ["year", "month", "day"])
                    and (len(set(("year", "month", "day")).intersection(set(cf_columns))) > 0)
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
                    df[new_time_column] = pd.to_datetime(df[plot_data_dict["time_column"]])
            except Exception as e:
                qs.error(
                    f"Unable to set {plot_data_dict['time_column']} as time reference column because {repr(e)}"
                )
                plot_data_dict["time_column"] = PICK_ONE
            else:
                df = df.sort_values([plot_data_dict["time_column"]])
        else:
            df[new_time_column] = df[plot_data_dict["time_column"]]
        plot_data_dict["time_column"] = new_time_column
        qs.info(f"Frames: {len(df[new_time_column].unique())}")
        plot_data_dict["animation_group"] = qs.selectbox(
            label="Animation category group", options=[NONE_SELECTED] + cat_columns, index=0
        )

    # Customize X axis
    if plot_type in PLOT_HAS_X:
        if plot_type in [PLOT_SCATTER, PLOT_LINE]:
            x_columns = [PICK_ONE] + all_columns
            y_columns = [PICK_ONE] + all_columns
        elif plot_type in [PLOT_SCATTER_3D]:
            x_columns = [PICK_ONE] + all_columns
            y_columns = [PICK_ONE] + all_columns
            z_columns = [PICK_ONE] + all_columns
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
        plot_data_dict["x"] = qs.selectbox(label="X axis", options=x_columns, index=0,)
        if (
            show_advanced_settings
            and plot_data_dict["x"] in num_columns
            and plot_type
            not in [PLOT_PARALLEL_CATEGORIES, PLOT_PARALLEL_COORDINATES, PLOT_SCATTER_MATRIX]
        ):
            plot_data_dict["log_x"] = qs.checkbox(label="Log X axis?")
        else:
            plot_data_dict["log_x"] = False
        if plot_data_dict["x"] == PICK_ONE:
            qs.warning("""Please pic a column for the X AXIS.""")
            return

    if plot_type == PLOT_HISTOGRAM:
        hist_modes = ["count", "sum", "avg", "min", "max"]
        plot_data_dict["histfunc"] = qs.selectbox(
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
    elif plot_type in PLOT_HAS_Y:
        # Customize Y axis
        plot_data_dict["y"] = qs.selectbox(label="Y axis", options=y_columns, index=0)
        if (
            show_advanced_settings
            and plot_data_dict["y"] in num_columns
            and plot_type
            not in [PLOT_PARALLEL_CATEGORIES, PLOT_PARALLEL_COORDINATES, PLOT_SCATTER_MATRIX]
        ):
            plot_data_dict["log_y"] = qs.checkbox(label="Log Y axis?")
        else:
            plot_data_dict["log_y"] = False
        if plot_data_dict["y"] == PICK_ONE:
            qs.warning("""Please pic a column for the Y AXIS.""")
            return

    if plot_type == PLOT_SCATTER_3D:
        plot_data_dict["z"] = qs.selectbox(label="Z axis", options=z_columns, index=0,)
        if show_advanced_settings and plot_data_dict["z"] in num_columns:
            plot_data_dict["log_z"] = qs.checkbox(label="Log Z axis?")
        else:
            plot_data_dict["log_z"] = False
        if plot_data_dict["z"] == PICK_ONE:
            qs.warning("""Please pic a column for the Z AXIS.""")
            return

    # Target for supervised machine learning
    if plot_type in PLOT_HAS_TARGET:
        plot_data_dict["target"] = qs.selectbox(
            label="Target:", options=[PICK_ONE] + supervision_columns, index=0,
        )
        if plot_data_dict["target"] == PICK_ONE:
            qs.warning("""Please select target for supervised machine learning.""")
            return
        elif plot_data_dict["target"] in df.select_dtypes(include=[np.float]).columns.to_list():
            qs.info("Non discrete columns will be rounded")

    # Color column
    if plot_type in PLOT_HAS_COLOR:
        plot_data_dict["color"] = qs.selectbox(
            label="Use this column for color:",
            options=[NONE_SELECTED] + all_columns,
            index=0
            if plot_type not in PLOT_HAS_TARGET
            else all_columns.index(plot_data_dict["target"]) + 1,
        )

    if show_advanced_settings:
        # Common data
        available_marginals = [NONE_SELECTED, "rug", "box", "violin", "histogram"]
        # Dot text
        if plot_type in PLOT_HAS_TEXT:
            plot_data_dict["text"] = qs.selectbox(
                label="Text display column", options=[NONE_SELECTED] + cat_columns, index=0
            )
        # Dot size
        if plot_type in PLOT_HAS_SIZE:
            plot_data_dict["size"] = qs.selectbox(
                label="Use this column to select what dot size represents:",
                options=[NONE_SELECTED] + num_columns,
                index=0,
            )
            plot_data_dict["size_max"] = qs.number_input(
                label="Max dot size", min_value=11, max_value=100, value=60
            )
        if plot_type in PLOT_HAS_SHAPE:
            plot_data_dict["symbol"] = qs.selectbox(
                label="Select a column for the dot symbols",
                options=[NONE_SELECTED] + cat_columns,
                index=0,
            )
        if plot_type in PLOT_HAS_TREND_LINE:
            plot_data_dict["trendline"] = qs.selectbox(
                label="Trend line mode",
                options=[NONE_SELECTED, "ols", "lowess"],
                format_func=lambda x: {
                    "ols": "Ordinary Least Squares ",
                    "lowess": "Locally Weighted Scatterplot Smoothing",
                }.get(x, x),
            )

        # Facet
        if plot_type in PLOT_HAS_FACET:
            plot_data_dict["facet_col"] = qs.selectbox(
                label="Use this column to split the plot in columns:",
                options=[NONE_SELECTED] + cat_columns,
                index=0,
            )
            if plot_data_dict["facet_col"] != NONE_SELECTED:
                plot_data_dict["facet_col_wrap"] = qs.number_input(
                    label="Wrap columns when more than x", min_value=1, max_value=20, value=4
                )
            else:
                plot_data_dict["facet_col_wrap"] = 4
            plot_data_dict["facet_row"] = qs.selectbox(
                label="Use this column to split the plot in lines:",
                options=[NONE_SELECTED] + cat_columns,
                index=0,
            )
        # Histogram specific parameters
        if plot_type in PLOT_HAS_MARGINAL and is_anim:
            plot_data_dict["marginal"] = qs.selectbox(
                label="Marginal", options=available_marginals, index=0
            )
            plot_data_dict["orientation"] = qs.selectbox(
                label="orientation",
                options=["v", "h"],
                format_func=lambda x: {"v": "vertical", "h": "horizontal"}.get(
                    x, "unknown value"
                ),
                index=0,
            )
        if plot_type in PLOT_HAS_BAR_MODE:
            plot_data_dict["barmode"] = qs.selectbox(
                label="bar mode", options=["group", "overlay", "relative"], index=2
            )
        if plot_type in PLOT_HAS_MARGINAL_XY and is_anim:
            plot_data_dict["marginal_x"] = qs.selectbox(
                label="Marginals for X axis", options=available_marginals, index=0
            )
            plot_data_dict["marginal_y"] = qs.selectbox(
                label="Marginals for Y axis", options=available_marginals, index=0
            )
        # Box plots and histograms
        if plot_type in PLOT_HAS_POINTS:
            plot_data_dict["points"] = qs.selectbox(
                label="Select which points are displayed",
                options=["none", "outliers", "all"],
                index=1,
            )
            plot_data_dict["points"] = (
                plot_data_dict["points"] if plot_data_dict["points"] != "none" else False
            )
        # Box plots
        if plot_type == PLOT_BOX:
            plot_data_dict["notched"] = qs.checkbox(label="Use notches?", value=False)
        # Violin plots
        if plot_type == PLOT_VIOLIN:
            plot_data_dict["box"] = qs.checkbox(label="Show boxes", value=False)
            plot_data_dict["violinmode"] = qs.selectbox(
                "Violin display mode", options=["group", "overlay"]
            )
        # Density heat map
        if plot_type in PLOT_HAS_BINS:
            plot_data_dict["nbinsx"] = qs.number_input(
                label="Number of bins in the X axis", min_value=1, max_value=1000, value=20
            )
            plot_data_dict["nbinsy"] = qs.number_input(
                label="Number of bins in the Y axis", min_value=1, max_value=1000, value=20
            )
        # Density contour map
        if plot_type == PLOT_DENSITY_CONTOUR:
            plot_data_dict["fill_contours"] = qs.checkbox(label="Fill contours", value=False)
        # PCA loadings
        if plot_type == PLOT_PCA_2D:
            plot_data_dict["show_loadings"] = qs.checkbox(label="Show loadings", value=False)
        # Ignored columns
        if plot_type in PLOT_HAS_IGNORE_COLUMNS:
            plot_data_dict["ignore_columns"] = qs.multiselect(
                label="Ignore columns:",
                options=all_columns,
                default=[plot_data_dict["target"]] if plot_type in PLOT_HAS_TARGET else [],
            )
            qs.info(
                "Ignored columns will be omitted when calculating LDA, but available for display"
            )
        # Correlation plot
        if plot_type == PLOT_CORR_MATRIX:
            plot_data_dict["corr_method"] = qs.selectbox(
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
            plot_data_dict["matrix_diag"] = qs.selectbox(
                label="diagonal", options=["Nothing", "Histogram", "Scatter"], index=1,
            )
            plot_data_dict["matrix_up"] = qs.selectbox(
                label="Upper triangle", options=["Nothing", "Scatter", "2D histogram"], index=1,
            )
            plot_data_dict["matrix_down"] = qs.selectbox(
                label="Lower triangle", options=["Nothing", "Scatter", "2D histogram"], index=1,
            )

        # Hover data
        if plot_type in PLOT_HAS_CUSTOM_HOVER_DATA:
            plot_data_dict["hover_name"] = qs.selectbox(
                label="Hover name:", options=[NONE_SELECTED] + cat_columns, index=0,
            )
            plot_data_dict["hover_data"] = qs.multiselect(
                label="Add columns to hover data", options=df.columns.to_list(), default=[],
            )
    else:
        if plot_type == PLOT_SCATTER_MATRIX:
            plot_data_dict["matrix_diag"] = "Histogram"
            plot_data_dict["matrix_up"] = "Scatter"
            plot_data_dict["matrix_down"] = "Scatter"
        if plot_type == PLOT_PCA_2D:
            plot_data_dict["show_loadings"] = False
        if plot_type == PLOT_CORR_MATRIX:
            plot_data_dict["corr_method"] = "pearson"
        if plot_type in PLOT_HAS_TARGET:
            plot_data_dict["ignore_columns"] = (
                [plot_data_dict["target"]] if plot_type in PLOT_HAS_TARGET else []
            )

    plot_data_dict["height"] = int(
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
    plot_data_dict["template"] = qs.selectbox(
        label="Plot template: ",
        options=available_templates,
        index=available_templates.index(pio.templates.default),
    )

    if is_anim and not st.button("Render"):
        st.info(
            """Since animations may tak long to initialize, 
            the rendering is started on when you click on the render button"""
        )
        return

    progress = st.progress(0)

    def update_progress(step, total):
        progress.progress(min(100, int(step / total * 100)))
        time.sleep(1)

    fig = build_plot(
        is_anim=is_anim, plot_type=plot_type, df=df, progress=update_progress, **plot_data_dict,
    )

    if fig is not None:
        st.plotly_chart(figure_or_data=fig, use_container_width=True)
        if (
            plot_type in [PLOT_PCA_2D, PLOT_PCA_3D]
            and not is_anim
            and st.checkbox(label="Show explained variance", value=False)
        ):
            X = df.loc[:, num_columns]
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            pca = PCA()
            x_new = pca.fit_transform(X)
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
                height=plot_data_dict["height"],
                template=plot_data_dict["template"],
                title="Explained variance by different principal components",
                xaxis_title="Principal component",
                yaxis_title="Explained variance in percent",
            )
            st.plotly_chart(figure_or_data=ev_fig, use_container_width=True)
    else:
        st.warning("No fig")


customize_plot()
