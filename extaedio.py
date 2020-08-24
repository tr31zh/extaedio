import io
import os

import base64

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

import amp_consts
from amp_functs import get_dataframe_from_url, format_csv_link, build_plot, get_plot_help


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


def customize_plot():

    step = 1

    st.title("Ex Taedio")

    st.markdown(
        """Welcome to Ex Taedio, a dashboard to help you generate plots from CSV files.  
        Click [here](https://github.com/tr31zh/ask_me_polotly) for the source code and help."""
    )

    show_info = st.checkbox(
        label="Show information panels (blue panels with hints and tips).", value=False
    )

    adv_mode = st.checkbox(label="Advanced mode", value=False)
    st.info(
        """
        **Advanced mode** will add:
        - Options to customize display
        - Option to enable data wrangling (filtering columns and rows)
        - Option to add advanced plots to list
        - Option to add advanced customization to plots
        """
    )

    if adv_mode:
        st.header(f"Step {step} - Set display options")
        step += 1

        if show_info:
            st.info(
                """
                - **Plot settings to side bar**:Put the plot setttings in the sidebar instead 
                of with all the other settings (Recommended).  
                - **Force wide display**: Set the main UI to occupy all the available space,
                can also be set from the settings. This overrides the settings if checked.
                """
            )
        use_side_bar = st.checkbox(label="Plot settings to side bar", value=True)
        if st.checkbox(label="Force wide display", value=True,):
            _max_width_()
    else:
        use_side_bar = True
        _max_width_()

    st.header(f"Step {step} - Load dataframe in CSV format")
    step += 1
    if show_info:
        st.info(
            f"""
            Select **{amp_consts.URL_LOCAL_FILE}** to load a file from your file system.  
            Select **{amp_consts.URL_DISTANT_FILE}** to paste an URL of a distant CSV.  
            The other options are CSVs that can be used to learn how to use the dashboard.
            """
        )
    selected_file = st.selectbox(
        label="Source file: ",
        options=amp_consts.AVAILABLE_URLS,
        index=0,
        format_func=format_csv_link,
    )
    if selected_file == amp_consts.URL_LOCAL_FILE:
        selected_file = st.file_uploader(label="Select file to upload")
        if selected_file is None:
            return
    elif selected_file == amp_consts.URL_DISTANT_FILE:
        selected_file = st.text_input(label="Paste web URL", value="")
        if not (st.button(label="Download file", key="grab_file") and selected_file):
            return
    df_loaded = get_df_from_url(url=selected_file)
    if df_loaded is None:
        return
    df = df_loaded.copy().reset_index(drop=True)

    if adv_mode:
        st.header(f"Step {step} - Set advanced settings")
        step += 1
        if show_info:
            st.info(
                """
            If activated, this settings can quickly become overwhelming.  
            - **Show advanced plots.**: Expand the list of available plots.
            - **Show dataframe customization options**: Add widgets to sort, filter and clean the dataframe.  
            - **Show plot customization advanced options**: Add widgets to further customize the plots.  
            - **Defer rendering**: Only generate plot when user presses the render button.
            Usefull if the rendering takes too long when changing a parameter.
            """
            )

        show_advanced_plots = st.checkbox(label="Show advanced plots.", value=False)
        show_dw_options = st.checkbox(label="Show dataframe customization options.")
        show_advanced_settings = st.checkbox(label="Show plot advanced options", value=False)
        defer_render = st.checkbox(label="Defer rendering", value=False)

        if show_dw_options:
            # Sorting
            st.header(f"Step {step} - Sort columns")
            step += 1
            if show_info:
                st.info(
                    """
                Select columns to sort the dataframe, if multiple columns are selected,
                sort will be applied in the displayed order.
                """
                )
            sort_columns = st.multiselect(label="Sort by", options=df.columns.to_list())
            invert_sort = st.checkbox(label="Reverse sort?", value=False)
            if sort_columns:
                df = df.sort_values(sort_columns, ascending=not invert_sort)
            # Filter
            st.header(f"Step {step} - Filter rows")
            step += 1
            if show_info:
                st.info("Some options to modify the dataframe")

            st.subheader("Selected data frame")
            if show_info:
                st.info("Displays n lines of the original dataframe")
            line_display_count = st.number_input(
                label="Lines to display", min_value=5, max_value=1000, value=5
            )
            st.dataframe(df.head(line_display_count))

            st.subheader("Dataframe description")
            if show_info:
                st.info(
                    """
                    Display info about the dataframe's numerical 
                    columns and types associated to all columns
                    """
                )
            st.dataframe(df.describe())
            st.write(df.dtypes)

            st.subheader("Select columns to keep")
            if show_info:
                st.info(
                    """
                    Remove all columns that will not be needed, 
                    the lower the number of columns the faster the dashboard will run
                    """
                )
            kept_columns = st.multiselect(
                label="Columns to keep",
                options=df.columns.to_list(),
                default=df.columns.to_list(),
            )
            df = df[kept_columns]

            st.subheader("Filter rows")
            if show_info:
                st.info(
                    """Select which columns will be filtered by values.  
                    Only date and string columns can be filtered at the moment"""
                )
            filter_columns = st.multiselect(
                label="Select which columns you want to use to filter the rows:",
                options=df.select_dtypes(include=["object", "datetime"]).columns.to_list(),
                default=None,
            )
            if filter_columns and show_info:
                st.info(
                    """For each selected column select all the values that will be included.
                    Less rows means faster dashboard"""
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

            st.subheader("Bin columns")
            if show_info:
                st.info(
                    """Select which columns will be binned.  
                    Numeric columns only"""
                )
            bin_columns = st.multiselect(
                label="Select which columns you want replace by bins:",
                options=df.select_dtypes(include=[np.number]).columns.to_list(),
                default=None,
            )
            if bin_columns and show_info:
                st.info("""For each selected column select the bin value""")
            binners = {}
            for column in bin_columns:
                st.subheader(f"{column}: ")
                binners[column] = st.number_input(
                    label="Bin count:",
                    min_value=1,
                    max_value=len(df[column].unique()),
                    value=10,
                )
            if len(binners) > 0:
                for k, v in binners.items():
                    df[k] = pd.cut(df[k], v)

            st.subheader("Clean up")
            if st.checkbox(label="Remove duplicates", value=False):
                df = df.drop_duplicates()
            if show_info:
                st.info(
                    "Some plots like PCA won't work if NA values are present in the dataframe"
                )
            if st.checkbox(label="Remove rows with NA values", value=False):
                df = df.dropna(axis="index")

            df = df.reset_index(drop=True)

            # Preview dataframe
            st.subheader("filtered data frame numeric data description and column types")
            if show_info:
                st.info("Display info about the filtered dataframe's numerical ")
            st.dataframe(df.describe())
            st.write(df.dtypes)
    else:
        show_advanced_settings = False
        defer_render = False
        show_advanced_plots = False

    qs = st.sidebar if use_side_bar else st
    if use_side_bar:
        pass
    else:
        st.header(f"Step {step} - Plot customization")
        step += 1

    num_columns = df.select_dtypes(include=[np.number]).columns.to_list()
    cat_columns = df.select_dtypes(include=["object", "datetime"]).columns.to_list()
    supervision_columns = df.select_dtypes(include=["object", "number"]).columns.to_list()
    all_columns = df.columns.to_list()

    qs.subheader("Plot selection")

    # Select type
    plot_type = qs.selectbox(
        label="Plot type: ",
        options=amp_consts.ALL_PLOTS if show_advanced_plots else amp_consts.BASIC_PLOTS,
        index=0,
    )
    st.header(
        f"Step {step} - Plot {plot_type}{' customization (Widgets in sidebar)' if use_side_bar else ''}"
    )
    step += 1

    st.write(get_plot_help(plot_type))
    if plot_type in [amp_consts.PLOT_LDA_2D, amp_consts.PLOT_NCA]:
        qs.warning("If plotting fails, make sure that no variable is colinear with your target")

    plot_data_dict = {}

    # Select mode
    is_anim = plot_type in amp_consts.PLOT_HAS_ANIM and qs.checkbox(
        label="Build animation", value=False
    )
    if is_anim:
        if show_info:
            qs.info(
                """Animation will be rendered when the **render** 
                button bellow the plot title is pressed
                """
            )
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
            time_columns = [amp_consts.PICK_ONE]
        time_columns.extend(
            [col for col in df.columns.to_list() if col.lower() not in time_columns]
        )
        plot_data_dict["time_column"] = qs.selectbox(
            label="Date/time column: ", options=time_columns, index=0,
        )
        if plot_data_dict["time_column"] != amp_consts.PICK_ONE:
            new_time_column = plot_data_dict["time_column"] + "_" + "pmgd"
        else:
            qs.warning("""Time column needed for animations.""")
            return
        if plot_data_dict["time_column"] != amp_consts.PICK_ONE and (
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
                    try:
                        df[new_time_column] = pd.to_datetime(df[plot_data_dict["time_column"]])
                    except Exception as e:
                        qs.warning(
                            "Failed to convert column to time, switching to category mode."
                        )
                        df[new_time_column] = (
                            df[plot_data_dict["time_column"]].astype("category").cat.codes
                        )
            except Exception as e:
                qs.error(
                    f"Unable to set {plot_data_dict['time_column']} as time reference column because {repr(e)}"
                )
                plot_data_dict["time_column"] = amp_consts.PICK_ONE
            else:
                df = df.sort_values([plot_data_dict["time_column"]])
        else:
            df[new_time_column] = df[plot_data_dict["time_column"]]
        plot_data_dict["time_column"] = new_time_column
        qs.info(f"Frames: {len(df[new_time_column].unique())}")
        plot_data_dict["animation_group"] = qs.selectbox(
            label="Animation category group",
            options=[amp_consts.NONE_SELECTED] + cat_columns,
            index=0,
        )
        if show_info:
            qs.info(
                """
                Select the main column of your dataframe as the category group.  
                For example if using *gapminder* select country.  
                If no column is selected the animation may jitter.
                """
            )

    qs.subheader("Basic options")

    # Customize X axis
    if plot_type in amp_consts.PLOT_HAS_X:
        if plot_type in [amp_consts.PLOT_SCATTER, amp_consts.PLOT_LINE]:
            x_columns = [amp_consts.PICK_ONE] + all_columns
            y_columns = [amp_consts.PICK_ONE] + all_columns
        elif plot_type in [amp_consts.PLOT_SCATTER_3D]:
            x_columns = [amp_consts.PICK_ONE] + all_columns
            y_columns = [amp_consts.PICK_ONE] + all_columns
            z_columns = [amp_consts.PICK_ONE] + all_columns
        elif plot_type in [amp_consts.PLOT_BAR]:
            x_columns = [amp_consts.PICK_ONE] + cat_columns
            y_columns = [amp_consts.PICK_ONE] + all_columns
        elif plot_type in [amp_consts.PLOT_BOX, amp_consts.PLOT_VIOLIN]:
            x_columns = [amp_consts.NONE_SELECTED] + cat_columns
            y_columns = [amp_consts.PICK_ONE] + all_columns
        elif plot_type == amp_consts.PLOT_HISTOGRAM:
            x_columns = [amp_consts.PICK_ONE] + all_columns
            y_columns = [amp_consts.PICK_ONE] + all_columns
        elif plot_type in [amp_consts.PLOT_DENSITY_HEATMAP, amp_consts.PLOT_DENSITY_CONTOUR]:
            x_columns = [amp_consts.PICK_ONE] + num_columns
            y_columns = [amp_consts.PICK_ONE] + num_columns
        plot_data_dict["x"] = qs.selectbox(label="X axis", options=x_columns, index=0,)
        if (
            show_advanced_settings
            and plot_data_dict["x"] in num_columns
            and plot_type
            not in [
                amp_consts.PLOT_PARALLEL_CATEGORIES,
                amp_consts.PLOT_PARALLEL_COORDINATES,
                amp_consts.PLOT_SCATTER_MATRIX,
            ]
        ):
            plot_data_dict["log_x"] = qs.checkbox(label="Log X axis?")
        else:
            plot_data_dict["log_x"] = False
        if plot_data_dict["x"] == amp_consts.PICK_ONE:
            qs.warning("""Please pic a column for the X AXIS.""")
            return

    if plot_type == amp_consts.PLOT_HISTOGRAM:
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
    elif plot_type in amp_consts.PLOT_HAS_Y:
        # Customize Y axis
        plot_data_dict["y"] = qs.selectbox(label="Y axis", options=y_columns, index=0)
        if (
            show_advanced_settings
            and plot_data_dict["y"] in num_columns
            and plot_type
            not in [
                amp_consts.PLOT_PARALLEL_CATEGORIES,
                amp_consts.PLOT_PARALLEL_COORDINATES,
                amp_consts.PLOT_SCATTER_MATRIX,
            ]
        ):
            plot_data_dict["log_y"] = qs.checkbox(label="Log Y axis?")
        else:
            plot_data_dict["log_y"] = False
        if plot_data_dict["y"] == amp_consts.PICK_ONE:
            qs.warning("""Please pic a column for the Y AXIS.""")
            return

    if plot_type == amp_consts.PLOT_SCATTER_3D:
        plot_data_dict["z"] = qs.selectbox(label="Z axis", options=z_columns, index=0,)
        if show_advanced_settings and plot_data_dict["z"] in num_columns:
            plot_data_dict["log_z"] = qs.checkbox(label="Log Z axis?")
        else:
            plot_data_dict["log_z"] = False
        if plot_data_dict["z"] == amp_consts.PICK_ONE:
            qs.warning("""Please pic a column for the Z AXIS.""")
            return

    # Target for supervised machine learning
    if plot_type in amp_consts.PLOT_HAS_TARGET:
        plot_data_dict["target"] = qs.selectbox(
            label="Target:", options=[amp_consts.PICK_ONE] + supervision_columns, index=0,
        )
        if plot_data_dict["target"] == amp_consts.PICK_ONE:
            qs.warning("""Please select target for supervised machine learning.""")
            return
        elif (
            plot_data_dict["target"] in df.select_dtypes(include=[np.float]).columns.to_list()
            and show_info
        ):
            qs.info("Non discrete columns will be rounded")

    # Color column
    if plot_type in amp_consts.PLOT_HAS_COLOR:
        plot_data_dict["color"] = qs.selectbox(
            label="Use this column for color:",
            options=[amp_consts.NONE_SELECTED] + all_columns,
            index=0
            if plot_type not in amp_consts.PLOT_HAS_TARGET
            else all_columns.index(plot_data_dict["target"]) + 1,
        )

    # Ignored columns
    if plot_type in amp_consts.PLOT_HAS_IGNORE_COLUMNS:
        plot_data_dict["ignore_columns"] = qs.multiselect(
            label="Ignore this columns when building the model:",
            options=all_columns,
            default=[plot_data_dict["target"]]
            if plot_type in amp_consts.PLOT_HAS_TARGET
            else [],
        )
        if show_info:
            qs.info(
                """
                **Ignored columns** will be omitted when calculating LDA, 
                but available for display.  
                Use this to avoid giving the answer to the question when building models.
                """
            )

    if show_advanced_settings:
        qs.subheader("Advanced options:")
        # Common data
        available_marginals = [amp_consts.NONE_SELECTED, "rug", "box", "violin", "histogram"]
        # Solver selection
        if plot_type in amp_consts.PLOT_HAS_SOLVER:
            solvers = ["svd", "eigen"]
            plot_data_dict["solver"] = qs.selectbox(
                label="Solver",
                options=solvers,
                index=0,
                format_func=lambda x: {
                    "svd": "Singular value decomposition",
                    "eigen": "Eigenvalue decomposition",
                }.get(x, "svd"),
            )
        # About NCA
        if plot_type in amp_consts.PLOT_HAS_NCOMP:
            plot_data_dict["n_components"] = qs.number_input(
                label="Number of components", min_value=2, max_value=len(num_columns), value=2
            )
        if plot_type in amp_consts.PLOT_HAS_INIT:
            plot_data_dict["init"] = qs.selectbox(
                label="Linear transformation init",
                options=["auto", "pca", "lda", "identity", "random"],
                index=0,
            )
            if plot_data_dict["init"] == "auto":
                qs.info(
                    """
                    Depending on n_components, the most reasonable initialization will be 
                    chosen. If n_components <= n_classes we use ‘lda’, as it uses labels 
                    information. If not, but n_components < min(n_features, n_samples), 
                    we use ‘pca’, as it projects data in meaningful directions 
                    (those of higher variance). Otherwise, we just use ‘identity’.
                    """
                )
            elif plot_data_dict["init"] == "pca":
                qs.info(
                    """n_components principal components of the inputs passed to 
                    fit will be used to initialize the transformation."""
                )
            elif plot_data_dict["init"] == "lda":
                qs.info(
                    """
                    min(n_components, n_classes) most discriminative components of
                    the inputs passed to fit will be used to initialize the transformation. 
                    (If n_components > n_classes, the rest of the components will be zero.)
                    """
                )
            elif plot_data_dict["init"] == "identity":
                qs.info(
                    """
                    If n_components is strictly smaller than the dimensionality 
                    of the inputs passed to 
                    fit, the identity matrix will be truncated to the first n_components rows.
                    """
                )
            elif plot_data_dict["init"] == "random":
                qs.info(
                    """
                    The initial transformation will be a random array of shape 
                    (n_components, n_features). 
                    Each value is sampled from the standard normal distribution.
                    """
                )
        # Dot text
        if plot_type in amp_consts.PLOT_HAS_TEXT:
            plot_data_dict["text"] = qs.selectbox(
                label="Text display column",
                options=[amp_consts.NONE_SELECTED] + cat_columns,
                index=0,
            )
        # Dot size
        if plot_type in amp_consts.PLOT_HAS_SIZE:
            plot_data_dict["size"] = qs.selectbox(
                label="Use this column to select what dot size represents:",
                options=[amp_consts.NONE_SELECTED] + num_columns,
                index=0,
            )
            plot_data_dict["size_max"] = qs.number_input(
                label="Max dot size", min_value=11, max_value=100, value=60
            )
        if plot_type in amp_consts.PLOT_HAS_SHAPE:
            plot_data_dict["symbol"] = qs.selectbox(
                label="Select a column for the dot symbols",
                options=[amp_consts.NONE_SELECTED] + cat_columns,
                index=0,
            )
        if plot_type in amp_consts.PLOT_HAS_TREND_LINE:
            plot_data_dict["trendline"] = qs.selectbox(
                label="Trend line mode",
                options=[amp_consts.NONE_SELECTED, "ols", "lowess"],
                format_func=lambda x: {
                    "ols": "Ordinary Least Squares ",
                    "lowess": "Locally Weighted Scatterplot Smoothing",
                }.get(x, x),
            )

        # Facet
        if plot_type in amp_consts.PLOT_HAS_FACET:
            plot_data_dict["facet_col"] = qs.selectbox(
                label="Use this column to split the plot in columns:",
                options=[amp_consts.NONE_SELECTED] + cat_columns,
                index=0,
            )
            if plot_data_dict["facet_col"] != amp_consts.NONE_SELECTED:
                plot_data_dict["facet_col_wrap"] = qs.number_input(
                    label="Wrap columns when more than x", min_value=1, max_value=20, value=4
                )
            else:
                plot_data_dict["facet_col_wrap"] = 4
            plot_data_dict["facet_row"] = qs.selectbox(
                label="Use this column to split the plot in lines:",
                options=[amp_consts.NONE_SELECTED] + cat_columns,
                index=0,
            )
        # Histogram specific parameters
        if plot_type in amp_consts.PLOT_HAS_MARGINAL and is_anim:
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
        if plot_type in amp_consts.PLOT_HAS_BAR_MODE:
            plot_data_dict["barmode"] = qs.selectbox(
                label="bar mode", options=["group", "overlay", "relative"], index=2
            )
        if plot_type in amp_consts.PLOT_HAS_MARGINAL_XY and is_anim:
            plot_data_dict["marginal_x"] = qs.selectbox(
                label="Marginals for X axis", options=available_marginals, index=0
            )
            plot_data_dict["marginal_y"] = qs.selectbox(
                label="Marginals for Y axis", options=available_marginals, index=0
            )
        # Box plots and histograms
        if plot_type in amp_consts.PLOT_HAS_POINTS:
            plot_data_dict["points"] = qs.selectbox(
                label="Select which points are displayed",
                options=["none", "outliers", "all"],
                index=1,
            )
            plot_data_dict["points"] = (
                plot_data_dict["points"] if plot_data_dict["points"] != "none" else False
            )
        # Box plots
        if plot_type == amp_consts.PLOT_BOX:
            plot_data_dict["notched"] = qs.checkbox(label="Use notches?", value=False)
        # Violin plots
        if plot_type == amp_consts.PLOT_VIOLIN:
            plot_data_dict["box"] = qs.checkbox(label="Show boxes", value=False)
            plot_data_dict["violinmode"] = qs.selectbox(
                "Violin display mode", options=["group", "overlay"]
            )
        # Density heat map
        if plot_type in amp_consts.PLOT_HAS_BINS:
            plot_data_dict["nbinsx"] = qs.number_input(
                label="Number of bins in the X axis", min_value=1, max_value=1000, value=20
            )
            plot_data_dict["nbinsy"] = qs.number_input(
                label="Number of bins in the Y axis", min_value=1, max_value=1000, value=20
            )
        # Density contour map
        if plot_type == amp_consts.PLOT_DENSITY_CONTOUR:
            plot_data_dict["fill_contours"] = qs.checkbox(label="Fill contours", value=False)
        # PCA loadings
        if plot_type == amp_consts.PLOT_PCA_2D:
            plot_data_dict["show_loadings"] = qs.checkbox(label="Show loadings", value=False)
        # Correlation plot
        if plot_type == amp_consts.PLOT_CORR_MATRIX:
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
        if plot_type == amp_consts.PLOT_SCATTER_MATRIX:
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
        if plot_type in amp_consts.PLOT_HAS_CUSTOM_HOVER_DATA:
            plot_data_dict["hover_name"] = qs.selectbox(
                label="Hover name:", options=[amp_consts.NONE_SELECTED] + cat_columns, index=0,
            )
            plot_data_dict["hover_data"] = qs.multiselect(
                label="Add columns to hover data", options=df.columns.to_list(), default=[],
            )
    else:
        if plot_type == amp_consts.PLOT_SCATTER_MATRIX:
            plot_data_dict["matrix_diag"] = "Histogram"
            plot_data_dict["matrix_up"] = "Scatter"
            plot_data_dict["matrix_down"] = "Scatter"
        if plot_type == amp_consts.PLOT_PCA_2D:
            plot_data_dict["show_loadings"] = False
        if plot_type == amp_consts.PLOT_CORR_MATRIX:
            plot_data_dict["corr_method"] = "pearson"

    if adv_mode:
        plot_data_dict["height"] = int(
            qs.selectbox(
                label="Plot height in pixels",
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
            label="Plot template (theme): ",
            options=available_templates,
            index=available_templates.index(pio.templates.default),
        )
    else:
        plot_data_dict["height"] = 900
        plot_data_dict["template"] = pio.templates.default

    if (is_anim or defer_render) and not st.button(label="Render", key="render_plot"):
        if is_anim and show_info:
            st.info(
                """Since animations may tak long to initialize, 
                the rendering starts only when you click on the **render** button above"""
            )
        return

    if plot_type in amp_consts.PLOT_HAS_PROGRESS_DISPLAY:
        progress = st.progress(0)

        def update_progress(step, total):
            progress.progress(min(100, int(step / total * 100)))

    else:
        update_progress = None

    if plot_type in amp_consts.PLOT_NEEDS_NA_DROP and df.isnull().values.any():
        st.markdown("NA values found in will be removed:")
        [
            st.markdown(f"- {c} has {df[c].isnull().sum()} NA values")
            for c in all_columns
            if df[c].isnull().values.any()
        ]
        df = df.dropna(axis="index")

    fig_data = build_plot(
        is_anim=is_anim, plot_type=plot_type, df=df, progress=update_progress, **plot_data_dict,
    )

    if fig_data:
        if "figure" in fig_data:
            html = fig_data.get("figure", None).to_html()
            b64 = base64.b64encode(html.encode()).decode()
            href = f"""<a href="data:file/html;base64,{b64}">
            Download plot as HTML file</a> 
            (Does not work on windows) 
            - right-click and save as &lt;some_name&gt;.html"""
            st.markdown(href, unsafe_allow_html=True)
            st.plotly_chart(
                figure_or_data=fig_data.get("figure", None), use_container_width=True
            )

        if "model_data" in fig_data:
            model_data = fig_data.get("model_data", None)
            if hasattr(model_data, "explained_variance_ratio_") and st.checkbox(
                label="Show explained variance", value=False
            ):
                df_ev = pd.DataFrame.from_dict(
                    {
                        "pc": [
                            f"PC{i}" for i in range(len(model_data.explained_variance_ratio_))
                        ],
                        "exp_var_per": model_data.explained_variance_ratio_ * 100,
                    }
                )
                df_ev = df_ev.assign(cumulative=df_ev["exp_var_per"].cumsum())
                ev_fig = go.Figure()
                ev_fig.add_trace(
                    go.Bar(x=df_ev["pc"], y=df_ev["exp_var_per"], name="individual",)
                )
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
            if (
                hasattr(model_data, "components_")
                and "column_names" in fig_data
                and st.checkbox(
                    label=(
                        lambda x: {
                            amp_consts.PLOT_PCA_2D: "Components details",
                            amp_consts.PLOT_PCA_3D: "Components details",
                            amp_consts.PLOT_NCA: "Linear transformation learned",
                        }.get(x, "Error")
                    )(plot_type),
                    value=False,
                )
            ):
                st.write(
                    pd.DataFrame.from_dict(
                        {
                            f"PC{i+1}": pc_data
                            for i, pc_data in enumerate(model_data.components_)
                        }
                    ).set_index(pd.Series(fig_data["column_names"]))
                )
            if (
                hasattr(model_data, "means_")
                and "class_names" in fig_data
                and "column_names" in fig_data
                and st.checkbox(label="Class means", value=False)
            ):
                st.write(
                    pd.DataFrame.from_dict(
                        {
                            col: pc_data
                            for col, pc_data in zip(
                                fig_data["column_names"], model_data.means_.T
                            )
                        }
                    ).set_index(pd.Series(fig_data["class_names"]))
                )
    else:
        st.warning("No plot")


customize_plot()
