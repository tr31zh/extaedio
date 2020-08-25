from collections import defaultdict
from datetime import datetime as dt
import base64

import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.io as pio


import amp_consts
from amp_consts import PICK_ONE
from amp_functs import (
    get_dataframe_from_url,
    format_csv_link,
    build_plot,
    get_plot_help_digest,
    get_plot_docstring,
)


def print_param_help(parent, param_name, params_doc):
    if isinstance(params_doc, str):
        parent.markdown(params_doc)
    else:
        for k, v in params_doc.items():
            p, *_ = k.split(":")
            if p == param_name:
                parent.markdown("".join(v))
                break
        else:
            parent.warning(f"Missing doc for {param_name}")
    parent.markdown("<hr>", unsafe_allow_html=True)


def init_param(
    parent,
    widget_type: str,
    param_name: str,
    widget_params: dict,
    show_help: bool,
    params_doc: dict,
    overrides: dict = {},
):
    if overrides and param_name in overrides:
        ret = overrides[param_name]
        parent.markdown(f"**{param_name}** <- {overrides[param_name]}")
    elif widget_type:
        f = getattr(parent, widget_type)
        ret = None if f is None else f(**widget_params)
    else:
        ret = None

    if ret == amp_consts.PICK_ONE:
        parent.warning(
            f"Please pic a column for the {widget_params.get('label', 'previous option')}."
        )

    if (show_help == "all") or ((show_help == "mandatory") and (ret == amp_consts.PICK_ONE)):
        print_param_help(parent=parent, param_name=param_name, params_doc=params_doc)

    return ret


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
        Click [here](https://github.com/tr31zh/ask_me_polotly/blob/master/README.pdf) for help
        and [here](https://github.com/tr31zh/ask_me_polotly) for the source code."""
    )

    param_help_level = st.selectbox(
        label="Show help related to plot options:",
        options=["none", "mandatory", "all"],
        format_func=lambda x: {
            "none": "Never",
            "mandatory": "When waiting for non optional parameters (recommended)",
            "all": "Always",
        }.get(x, "all"),
        index=1,
    )
    st.markdown("")

    show_info = st.checkbox(
        label="Show information panels (blue panels with hints and tips).", value=False
    )

    adv_mode = st.checkbox(label="Advanced mode", value=False)
    if show_info:
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
        show_dw_options = st.checkbox(
            label="Show dataframe customization options - sort, filter, clean."
        )
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

    st.write(get_plot_help_digest(plot_type))
    if plot_type in [amp_consts.PLOT_LDA_2D, amp_consts.PLOT_NCA]:
        qs.warning("If plotting fails, make sure that no variable is colinear with your target")

    params = get_plot_docstring(plot_type).split("\nParameters")[1].split("\n")[2:]
    params_dict = defaultdict(list)
    current_key = ""
    for l in params:
        if l.startswith("    "):
            params_dict[current_key].append(l.replace("    ", " "))
        else:
            current_key = l
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
        plot_data_dict["time_column"] = init_param(
            parent=qs,
            widget_type="selectbox",
            param_name="time_column",
            widget_params=dict(label="Date/time column: ", options=time_columns, index=0),
            show_help=param_help_level,
            params_doc="A column from the dataframe used as key to build frames",
            overrides={},
        )
        if plot_data_dict["time_column"] != amp_consts.PICK_ONE:
            new_time_column = plot_data_dict["time_column"] + "_" + "pmgd"
        else:
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
        plot_data_dict["animation_group"] = init_param(
            parent=qs,
            widget_type="selectbox",
            param_name="animation_group",
            widget_params=dict(
                label="Animation category group",
                options=[amp_consts.NONE_SELECTED] + cat_columns,
                index=0,
            ),
            show_help=param_help_level,
            params_doc=params_dict,
            overrides={},
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

    if plot_type in amp_consts.PLOT_HAS_X:
        if plot_type in [amp_consts.PLOT_SCATTER, amp_consts.PLOT_LINE]:
            x_columns = [amp_consts.PICK_ONE] + all_columns
        elif plot_type in [amp_consts.PLOT_SCATTER_3D]:
            x_columns = [amp_consts.PICK_ONE] + all_columns
        elif plot_type in [amp_consts.PLOT_BAR]:
            x_columns = [amp_consts.PICK_ONE] + cat_columns
        elif plot_type in [amp_consts.PLOT_BOX, amp_consts.PLOT_VIOLIN]:
            x_columns = [amp_consts.NONE_SELECTED] + cat_columns
        elif plot_type == amp_consts.PLOT_HISTOGRAM:
            x_columns = [amp_consts.PICK_ONE] + all_columns
        elif plot_type in [amp_consts.PLOT_DENSITY_HEATMAP, amp_consts.PLOT_DENSITY_CONTOUR]:
            x_columns = [amp_consts.PICK_ONE] + num_columns
        else:
            x_columns = []
    else:
        x_columns = []

    if plot_type in amp_consts.PLOT_HAS_Y:
        if plot_type in [amp_consts.PLOT_SCATTER, amp_consts.PLOT_LINE]:
            y_columns = [amp_consts.PICK_ONE] + all_columns
        elif plot_type in [amp_consts.PLOT_SCATTER_3D]:
            y_columns = [amp_consts.PICK_ONE] + all_columns
        elif plot_type in [amp_consts.PLOT_BAR]:
            y_columns = [amp_consts.PICK_ONE] + all_columns
        elif plot_type in [amp_consts.PLOT_BOX, amp_consts.PLOT_VIOLIN]:
            y_columns = [amp_consts.PICK_ONE] + all_columns
        elif plot_type == amp_consts.PLOT_HISTOGRAM:
            y_columns = [amp_consts.PICK_ONE] + all_columns
        elif plot_type in [amp_consts.PLOT_DENSITY_HEATMAP, amp_consts.PLOT_DENSITY_CONTOUR]:
            y_columns = [amp_consts.PICK_ONE] + num_columns
        else:
            y_columns = []
    else:
        y_columns = []

    if plot_type in amp_consts.PLOT_HAS_Z:
        if plot_type in [amp_consts.PLOT_SCATTER_3D]:
            z_columns = [amp_consts.PICK_ONE] + all_columns
        else:
            z_columns = []
    else:
        z_columns = []

    # Customize X axis
    if plot_type in amp_consts.PLOT_HAS_X:
        plot_data_dict["x"] = init_param(
            parent=qs,
            widget_type="selectbox",
            param_name="x",
            widget_params=dict(label="X axis", options=x_columns, index=0),
            show_help=param_help_level,
            params_doc=params_dict,
            overrides={},
        )
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
            plot_data_dict["log_x"] = init_param(
                parent=qs,
                widget_type="checkbox",
                param_name="log_x",
                widget_params=dict(label="Log X axis?"),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        if plot_data_dict["x"] == amp_consts.PICK_ONE:
            return

    if plot_type == amp_consts.PLOT_HISTOGRAM:
        plot_data_dict["histfunc"] = init_param(
            parent=qs,
            widget_type="selectbox",
            param_name="histfunc",
            widget_params=dict(
                label="Histogram function",
                options=["count", "sum", "avg", "min", "max"],
                format_func=lambda x: {
                    "count": "Count",
                    "sum": "Sum",
                    "avg": "Average",
                    "min": "Minimum",
                    "max": "Maximum",
                }.get(x, "Unknown histogram mode"),
            ),
            show_help=param_help_level,
            params_doc=params_dict,
            overrides={},
        )
    elif plot_type in amp_consts.PLOT_HAS_Y:
        # Customize Y axis
        plot_data_dict["y"] = init_param(
            parent=qs,
            widget_type="selectbox",
            param_name="y",
            widget_params=dict(label="Y axis", options=y_columns, index=0),
            show_help=param_help_level,
            params_doc=params_dict,
            overrides={},
        )
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
            plot_data_dict["log_y"] = init_param(
                parent=qs,
                widget_type="checkbox",
                param_name="log_y",
                widget_params=dict(label="Log Y axis?"),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        if plot_data_dict["y"] == amp_consts.PICK_ONE:
            return

    if plot_type == amp_consts.PLOT_SCATTER_3D:
        plot_data_dict["z"] = init_param(
            parent=qs,
            widget_type="selectbox",
            param_name="z",
            widget_params=dict(label="Z axis", options=z_columns, index=0),
            show_help=param_help_level,
            params_doc=params_dict,
            overrides={},
        )
        if show_advanced_settings and plot_data_dict["z"] in num_columns:
            plot_data_dict["log_z"] = init_param(
                parent=qs,
                widget_type="checkbox",
                param_name="log_z",
                widget_params=dict(label="Log Z axis?"),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        else:
            plot_data_dict["log_z"] = False
        if plot_data_dict["z"] == amp_consts.PICK_ONE:
            return

    # Target for supervised machine learning
    if plot_type in amp_consts.PLOT_HAS_TARGET:
        plot_data_dict["target"] = init_param(
            parent=qs,
            widget_type="selectbox",
            param_name="target",
            widget_params=dict(
                label="ML target:", options=[amp_consts.PICK_ONE] + supervision_columns, index=0
            ),
            show_help=param_help_level,
            params_doc=params_dict,
            overrides={},
        )
        if plot_data_dict["target"] == amp_consts.PICK_ONE:
            return
        elif (
            plot_data_dict["target"] in df.select_dtypes(include=[np.float]).columns.to_list()
            and show_info
        ):
            qs.info("Non discrete columns will be rounded")

    # Color column
    if plot_type in amp_consts.PLOT_HAS_COLOR:
        plot_data_dict["color"] = init_param(
            parent=qs,
            widget_type="selectbox",
            param_name="color",
            widget_params=dict(
                label="Use this column for color:",
                options=[amp_consts.NONE_SELECTED] + all_columns,
                index=0
                if plot_type not in amp_consts.PLOT_HAS_TARGET
                else all_columns.index(plot_data_dict["target"]) + 1,
            ),
            show_help=param_help_level,
            params_doc=params_dict,
            overrides={},
        )

    # Ignored columns
    if plot_type in amp_consts.PLOT_HAS_IGNORE_COLUMNS:
        plot_data_dict["ignore_columns"] = init_param(
            parent=qs,
            widget_type="multiselect",
            param_name="ignore_columns",
            widget_params=dict(
                label="Ignore this columns when building the model:",
                options=all_columns,
                default=[plot_data_dict["target"]]
                if plot_type in amp_consts.PLOT_HAS_TARGET
                else [],
            ),
            show_help=param_help_level,
            params_doc="""
                This columns will be omitted when building the model, 
                but available for display.  
                Use this to avoid giving the answer to the question when building models.
                """,
            overrides={},
        )
    if show_advanced_settings:
        qs.subheader("Advanced options:")
        # Common data
        available_marginals = [amp_consts.NONE_SELECTED, "rug", "box", "violin", "histogram"]
        # Solver selection
        if plot_type in amp_consts.PLOT_HAS_SOLVER:
            plot_data_dict["solver"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="solver",
                widget_params=dict(
                    label="Solver",
                    options=["svd", "eigen"],
                    index=0,
                    format_func=lambda x: {
                        "svd": "Singular value decomposition",
                        "eigen": "Eigenvalue decomposition",
                    }.get(x, "svd"),
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        # About NCA
        if plot_type in amp_consts.PLOT_HAS_NCOMP:
            plot_data_dict["n_components"] = init_param(
                parent=qs,
                widget_type="number_input",
                param_name="n_components",
                widget_params=dict(
                    label="Number of components",
                    min_value=2,
                    max_value=len(num_columns),
                    value=2,
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        if plot_type in amp_consts.PLOT_HAS_INIT:
            plot_data_dict["init"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="init",
                widget_params=dict(
                    label="Linear transformation init",
                    options=["auto", "pca", "lda", "identity", "random"],
                    index=0,
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
            if plot_data_dict["init"] == "auto":
                qs.markdown(
                    """
                    Depending on n_components, the most reasonable initialization will be 
                    chosen. If n_components <= n_classes we use ‘lda’, as it uses labels 
                    information. If not, but n_components < min(n_features, n_samples), 
                    we use ‘pca’, as it projects data in meaningful directions 
                    (those of higher variance). Otherwise, we just use ‘identity’.
                    """
                )
            elif plot_data_dict["init"] == "pca":
                qs.markdown(
                    """n_components principal components of the inputs passed to 
                    fit will be used to initialize the transformation."""
                )
            elif plot_data_dict["init"] == "lda":
                qs.markdown(
                    """
                    min(n_components, n_classes) most discriminative components of
                    the inputs passed to fit will be used to initialize the transformation. 
                    (If n_components > n_classes, the rest of the components will be zero.)
                    """
                )
            elif plot_data_dict["init"] == "identity":
                qs.markdown(
                    """
                    If n_components is strictly smaller than the dimensionality 
                    of the inputs passed to 
                    fit, the identity matrix will be truncated to the first n_components rows.
                    """
                )
            elif plot_data_dict["init"] == "random":
                qs.markdown(
                    """
                    The initial transformation will be a random array of shape 
                    (n_components, n_features). 
                    Each value is sampled from the standard normal distribution.
                    """
                )
            qs.markdown("<hr>", unsafe_allow_html=True)
        # Dot text
        if plot_type in amp_consts.PLOT_HAS_TEXT:
            plot_data_dict["text"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="text",
                widget_params=dict(
                    label="Text display column",
                    options=[amp_consts.NONE_SELECTED] + cat_columns,
                    index=0,
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        # Dot size
        if plot_type in amp_consts.PLOT_HAS_SIZE:
            plot_data_dict["size"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="size",
                widget_params=dict(
                    label="Use this column to select what dot size represents:",
                    options=[amp_consts.NONE_SELECTED] + num_columns,
                    index=0,
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
            plot_data_dict["size_max"] = init_param(
                parent=qs,
                widget_type="number_input",
                param_name="size_max",
                widget_params=dict(label="Max dot size", min_value=11, max_value=100, value=60),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        if plot_type in amp_consts.PLOT_HAS_SHAPE:
            plot_data_dict["symbol"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="symbol",
                widget_params=dict(
                    label="Select a column for the dot symbols",
                    options=[amp_consts.NONE_SELECTED] + cat_columns,
                    index=0,
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        if plot_type in amp_consts.PLOT_HAS_TREND_LINE:
            plot_data_dict["trendline"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="trendline",
                widget_params=dict(
                    label="Trend line mode",
                    options=[amp_consts.NONE_SELECTED, "ols", "lowess"],
                    format_func=lambda x: {
                        "ols": "Ordinary Least Squares ",
                        "lowess": "Locally Weighted Scatterplot Smoothing",
                    }.get(x, x),
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )

        # Facet
        if plot_type in amp_consts.PLOT_HAS_FACET:
            plot_data_dict["facet_col"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="facet_col",
                widget_params=dict(
                    label="Use this column to split the plot in columns:",
                    options=[amp_consts.NONE_SELECTED] + cat_columns,
                    index=0,
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
            if plot_data_dict["facet_col"] != amp_consts.NONE_SELECTED:
                plot_data_dict["facet_col_wrap"] = init_param(
                    parent=qs,
                    widget_type="number_input",
                    param_name="facet_col_wrap",
                    widget_params=dict(
                        label="Wrap columns when more than x",
                        min_value=1,
                        max_value=20,
                        value=4,
                    ),
                    show_help=param_help_level,
                    params_doc=params_dict,
                    overrides={},
                )
            else:
                plot_data_dict["facet_col_wrap"] = 4
            plot_data_dict["facet_row"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="facet_row",
                widget_params=dict(
                    label="Use this column to split the plot in lines:",
                    options=[amp_consts.NONE_SELECTED] + cat_columns,
                    index=0,
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        # Histogram specific parameters
        if plot_type in amp_consts.PLOT_HAS_MARGINAL and is_anim:
            plot_data_dict["marginal"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="marginal",
                widget_params=dict(label="Marginal", options=available_marginals, index=0),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
            plot_data_dict["orientation"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="orientation",
                widget_params=dict(
                    label="orientation",
                    options=["v", "h"],
                    format_func=lambda x: {"v": "vertical", "h": "horizontal"}.get(
                        x, "unknown value"
                    ),
                    index=0,
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        if plot_type in amp_consts.PLOT_HAS_BAR_MODE:
            plot_data_dict["barmode"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="barmode",
                widget_params=dict(
                    label="bar mode", options=["group", "overlay", "relative"], index=2
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        if plot_type in amp_consts.PLOT_HAS_MARGINAL_XY and is_anim:
            plot_data_dict["marginal_x"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="marginal_x",
                widget_params=dict(
                    label="Marginals for X axis", options=available_marginals, index=0
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
            plot_data_dict["marginal_y"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="marginal_y",
                widget_params=dict(
                    label="Marginals for Y axis", options=available_marginals, index=0
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        # Box plots and histograms
        if plot_type in amp_consts.PLOT_HAS_POINTS:
            plot_data_dict["points"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="points",
                widget_params=dict(
                    label="Select which points are displayed",
                    options=["none", "outliers", "suspectedoutliers", "all"],
                    index=1,
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
            plot_data_dict["points"] = (
                plot_data_dict["points"] if plot_data_dict["points"] != "none" else False
            )
        # Box plots
        if plot_type == amp_consts.PLOT_BOX:
            plot_data_dict["notched"] = init_param(
                parent=qs,
                widget_type="checkbox",
                param_name="notched",
                widget_params=dict(label="Use notches?", value=False),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        # Violin plots
        if plot_type == amp_consts.PLOT_VIOLIN:
            plot_data_dict["box"] = init_param(
                parent=qs,
                widget_type="checkbox",
                param_name="box",
                widget_params=dict(label="Show boxes", value=False),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
            plot_data_dict["violinmode"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="violinmode",
                widget_params=dict(label="Violin display mode", options=["group", "overlay"]),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        # Density heat map
        if plot_type in amp_consts.PLOT_HAS_BINS:
            plot_data_dict["nbinsx"] = init_param(
                parent=qs,
                widget_type="number_input",
                param_name="nbinsx",
                widget_params=dict(
                    label="Number of bins in the X axis", min_value=1, max_value=1000, value=20
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
            plot_data_dict["nbinsy"] = init_param(
                parent=qs,
                widget_type="number_input",
                param_name="nbinsy",
                widget_params=dict(
                    label="Number of bins in the Y axis", min_value=1, max_value=1000, value=20
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        # Density contour map
        if plot_type == amp_consts.PLOT_DENSITY_CONTOUR:
            plot_data_dict["fill_contours"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="fill_contours",
                widget_params=dict(label="Fill contours", value=False),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        # PCA loadings
        if plot_type == amp_consts.PLOT_PCA_2D:
            plot_data_dict["show_loadings"] = init_param(
                parent=qs,
                widget_type="checkbox",
                param_name="show_loadings",
                widget_params=dict(label="Show loadings", value=False),
                show_help=param_help_level,
                params_doc="Display PCA loadings for each attribute",
                overrides={},
            )
        # Correlation plot
        if plot_type == amp_consts.PLOT_CORR_MATRIX:
            plot_data_dict["corr_method"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="corr_method",
                widget_params=dict(
                    label="Correlation method",
                    options=["pearson", "kendall", "spearman"],
                    format_func=lambda x: {
                        "pearson": "pearson : standard correlation coefficient",
                        "kendall": "kendall : Kendall Tau correlation coefficient",
                        "spearman": "spearman : Spearman rank correlation",
                    }.get(x),
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        # Matrix plot
        if plot_type == amp_consts.PLOT_SCATTER_MATRIX:
            plot_data_dict["matrix_diag"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="matrix_diag",
                widget_params=dict(
                    label="diagonal", options=["Nothing", "Histogram", "Scatter"], index=1
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
            plot_data_dict["matrix_up"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="matrix_up",
                widget_params=dict(
                    label="Upper triangle",
                    options=["Nothing", "Scatter", "2D histogram"],
                    index=1,
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
            plot_data_dict["matrix_down"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="matrix_down",
                widget_params=dict(
                    label="Lower triangle",
                    options=["Nothing", "Scatter", "2D histogram"],
                    index=1,
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )

        # Hover data
        if plot_type in amp_consts.PLOT_HAS_CUSTOM_HOVER_DATA:
            plot_data_dict["hover_name"] = init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="hover_name",
                widget_params=dict(
                    label="Hover name:",
                    options=[amp_consts.NONE_SELECTED] + cat_columns,
                    index=0,
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
            plot_data_dict["hover_data"] = init_param(
                parent=qs,
                widget_type="multiselect",
                param_name="hover_data",
                widget_params=dict(
                    label="Add columns to hover data", options=df.columns.to_list(), default=[]
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
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
            init_param(
                parent=qs,
                widget_type="selectbox",
                param_name="height",
                widget_params=dict(
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
                ),
                show_help=param_help_level,
                params_doc=params_dict,
                overrides={},
            )
        )
        # Template
        available_templates = list(pio.templates.keys())
        plot_data_dict["template"] = init_param(
            parent=qs,
            widget_type="selectbox",
            param_name="template",
            widget_params=dict(
                label="Plot template (theme): ",
                options=available_templates,
                index=available_templates.index(pio.templates.default),
            ),
            show_help=param_help_level,
            params_doc=params_dict,
            overrides={},
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
