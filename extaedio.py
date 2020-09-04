from collections import defaultdict
from datetime import datetime as dt
import base64
import json
import datetime

import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.io as pio

import amp_consts
import amp_st_functs
from amp_functs import (
    build_plot,
    get_plot_help_digest,
    get_plot_docstring,
)


def get_final_index(default_index: int, options: list, key: str, overrides: dict) -> int:
    if key in overrides:
        return options.index(overrides[key])
    else:
        return default_index


class ParamInitializer(object):
    def __init__(self, parent, params_doc, overrides, show_help) -> None:
        self._parent = parent
        self._params_doc = params_doc
        self._overrides = overrides
        self._show_help = show_help

    def __call__(
        self,
        param_name,
        widget_params,
        widget_type="selectbox",
        doc_override=None,
        lock: bool = False,
    ):
        if self._overrides and lock:
            if param_name in self._overrides:
                self._parent.markdown(
                    f"**{param_name}** <- {self._overrides[param_name]}"
                )
                self._parent.info(f"**{param_name}** is locked in report mode")
                ret = self._overrides[param_name]
            else:
                return None
        elif widget_type:
            f = getattr(self._parent, widget_type)
            if param_name in self._overrides:
                if "options" in widget_params and "index" in widget_params:
                    widget_params["index"] = get_final_index(
                        default_index=widget_params["index"],
                        options=widget_params["options"],
                        key=param_name,
                        overrides=self._overrides,
                    )
                elif "value" in widget_params:
                    widget_params["value"] = self._overrides[param_name]

            ret = None if f is None else f(**widget_params)
        else:
            ret = None

        if ret == amp_consts.PICK_ONE:
            self._parent.warning(
                f"Please pic a column for the {widget_params.get('label', 'previous parameter')}."
            )

        if (self._show_help == "all") or (
            (self._show_help == "mandatory") and (ret == amp_consts.PICK_ONE)
        ):
            self.print_help(
                param_name=param_name,
                params_doc=doc_override if doc_override is not None else self._params_doc,
            )

        return ret

    def print_help(self, param_name, params_doc):
        if isinstance(params_doc, str):
            self._parent.markdown(params_doc)
        else:
            for k, v in params_doc.items():
                p, *_ = k.split(":")
                if p == param_name:
                    self._parent.markdown("".join(v))
                    break
            else:
                self._parent.warning(f"Missing doc for {param_name}")
        self._parent.markdown("___")


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
def wrangle_the_data(df, dw_options):

    # Sort
    if dw_options["sort_columns"]:
        df = df.sort_values(
            dw_options["sort_columns"], ascending=not dw_options["invert_sort"],
        )

    # Filter columns
    df = df[dw_options["kept_columns"]]

    # Filter rows
    if len(dw_options["filters"]) > 0:
        for k, v in dw_options["filters"].items():
            df = df[df[k].isin(v)]

    # Bin columns
    if len(dw_options["binners"]) > 0:
        for k, v in dw_options["binners"].items():
            df[k] = pd.cut(df[k], v)

    # Clean
    if dw_options["remove_duplicates"] is True:
        df = df.drop_duplicates()
    if dw_options["remove_na"] is True:
        df = df.dropna(axis="index")
    return df


def customize_plot():

    step = 1

    st.title("Ex Taedio")

    st.markdown(
        "Welcome to Ex Taedio, a dashboard to help you generate plots from CSV files."
    )

    reporting_mode = st.checkbox(label="Switch to report mode", value=False)
    st.info(
        """
        **About report mode:**  Report mode allows you to reproduce a plot made 
        by somebody else by loading a JSON configuration file that you got from another user.
        """
    )
    if reporting_mode:
        _max_width_()
        report_path = st.file_uploader(
            label="Select JSON file containing plot parameters and options"
        )
        if report_path is None:
            return
        report = json.loads(report_path.getvalue())
    else:
        report = {}

    st.header(f"Step {step} - How do we get started:")
    step += 1
    ui_modes = [
        "no_choice_all_help",
        "no_choice_some_help",
        "some_choice_some_help",
        "all_choices",
    ]
    ui_mode = st.radio(
        label="User interface mode",
        options=ui_modes,
        index=get_final_index(
            default_index=1, options=ui_modes, key="ui_mode", overrides=report
        ),
        format_func=lambda x: {
            "no_choice_all_help": "Beginner",
            "no_choice_some_help": "Normal",
            "some_choice_some_help": "Advanced",
            "all_choices": "Expert",
        }.get(x, "Unknown"),
    )

    if ui_mode == "some_choice_some_help":
        st.header(f"Step {step} - Some basic configuration")
        step += 1
    elif ui_mode == "all_choices":
        st.header(f"Step {step} - Advanced settings")
        step += 1
    else:
        pass

    if ui_mode == "no_choice_some_help":
        param_help_level = "mandatory"
    elif ui_mode == "no_choice_all_help":
        param_help_level = "all"
    else:
        help_levels = ["none", "mandatory", "all"]
        param_help_level = st.selectbox(
            label="Show help related to plot parameters:",
            options=help_levels,
            format_func=lambda x: {
                "none": "Never",
                "mandatory": "When waiting for non optional parameters (recommended)",
                "all": "Always",
            }.get(x, "all"),
            index=get_final_index(
                default_index=0 if ui_mode == "all_choices" else 1,
                options=help_levels,
                key="param_help_level",
                overrides=report,
            ),
        )

    if ui_mode == "no_choice_some_help":
        real_time_render = False
    elif ui_mode == "no_choice_all_help":
        real_time_render = False
    else:
        real_time_render = st.checkbox(
            label="""
            Realtime rendering: build plot as soon as a parameter is changed, may render the UI unresponsive. Disabled for animations
            """,
            value=report.get("real_time_render", False),
        )

    if ui_mode == "no_choice_some_help":
        show_info = False
    elif ui_mode == "no_choice_all_help":
        show_info = "True"
    else:
        show_info = st.checkbox(
            label="Show information panels (blue panels with hints and tips).",
            value=report.get("show_info", False),
        )

    if ui_mode == "no_choice_some_help":
        use_side_bar = True
    elif ui_mode == "no_choice_all_help":
        use_side_bar = True
    else:
        use_side_bar = st.checkbox(
            label="Plot settings to side bar", value=report.get("use_side_bar", True)
        )
        if show_info:
            st.info(
                """
                **Plot settings to side bar**:Put the plot setttings in the sidebar instead 
                of with all the other settings (Recommended).
                """
            )

    if ui_mode == "no_choice_some_help":
        show_advanced_plots = False
    elif ui_mode == "no_choice_all_help":
        show_advanced_plots = False
    else:
        show_advanced_plots = st.checkbox(
            label="Show advanced plots.", value=report.get("show_advanced_plots", False)
        )
        if show_info:
            st.info(
                """**Show advanced plots.**: Expand the list of available plots.
            Includes among others PCA3D, LDA, density plots, ..."""
            )

    if ui_mode == "no_choice_some_help":
        show_dw_options = False
    elif ui_mode == "no_choice_all_help":
        show_dw_options = False
    elif ui_mode == "some_choice_some_help":
        show_dw_options = False
    elif ui_mode == "all_choices":
        if report:
            st.warning(
                "Avoid modifying the dataframe while in report mode, some columns may be needed later"
            )
        show_dw_options = st.checkbox(
            label="Show dataframe customization options.",
            value=report.get("show_dw_options", False),
        )
        if show_info:
            st.info(
                "**Show dataframe customization options**: Add widgets to sort, filter and clean the dataframe."
            )
    else:
        show_dw_options = False
        st.error(f"Unknown UI mode: {ui_mode}")

    if ui_mode == "no_choice_some_help":
        show_advanced_settings = False
    elif ui_mode == "no_choice_all_help":
        show_advanced_settings = False
    else:
        show_advanced_settings = st.checkbox(
            label="Show plot advanced parameters",
            value=report.get("show_advanced_settings", False),
        )
        if show_info:
            st.info(
                """**Show plot customization advanced parameters**: 
                Add widgets to further customize the plots.  
                Usefull if the rendering takes too long when changing a parameter.
                """
            )

    if ui_mode == "all_choices":
        force_wide_mode = st.checkbox(
            label="Force wide display", value=report.get("force_wide_mode", True),
        )
    else:
        force_wide_mode = True
    if force_wide_mode:
        _max_width_()
    if ui_mode == "all_choices" and show_info:
        st.info(
            """
            **Force wide display**: Set the main UI to occupy all the available space,
            can also be set from the settings. This if checked, overrides the settings.
            """
        )

    if "dataframe" in report:
        df = pd.DataFrame.from_dict(report.get("dataframe", {}))
        selected_file = None
        dw_options = {}
    else:
        df = amp_st_functs.load_dataframe(step=step, show_info=show_info)
        step += 1
    if df is None:
        return

    dw_options = {}

    if show_dw_options:
        st.header(f"Step {step} - Data wrangling")
        st.subheader("Source dataframe")
        st.markdown("dataframe first rows")
        line_display_count = st.number_input(
            label="Lines to display", min_value=5, max_value=1000, value=5
        )
        st.dataframe(df.head(line_display_count))
        st.markdown("Data frame numerical columns description")
        st.dataframe(df.describe())
        st.markdown("Dataframe's column types")
        st.write(df.dtypes)

        st.subheader(f"Sort")
        if show_info:
            st.info(
                """
            Select columns to sort the dataframe, if multiple columns are selected,
            sort will be applied in the displayed order.
            """
            )
        dw_options["sort_columns"] = st.multiselect(
            label="Sort by", options=df.columns.to_list()
        )
        dw_options["invert_sort"] = st.checkbox(label="Reverse sort?", value=False)

        st.subheader("Filter columns")
        dw_options["kept_columns"] = st.multiselect(
            label="Columns to keep",
            options=df.columns.to_list(),
            default=df.columns.to_list(),
        )

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
        dw_options["filters"] = {}
        for column in filter_columns:
            st.subheader(f"{column}: ")
            select_all = st.checkbox(label=f"{column} Select all:")
            elements = list(df[column].unique())
            dw_options["filters"][column] = st.multiselect(
                label=f"Select which {column} to include",
                options=elements,
                default=None if not select_all else elements,
            )

        st.subheader("Bin numerical columns")
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
        dw_options["binners"] = {}
        for column in bin_columns:
            st.subheader(f"{column}: ")
            dw_options["binners"][column] = st.number_input(
                label="Bin count:",
                min_value=1,
                max_value=len(df[column].unique()),
                value=10,
            )

        st.subheader("Clean")
        if show_info:
            st.info(
                "Some plots like PCA won't work if NA values are present in the dataframe"
            )
        dw_options["remove_na"] = st.checkbox(
            label="Remove rows with NA values", value=False
        )
        dw_options["remove_duplicates"] = st.checkbox(
            label="Remove duplicates", value=False
        )

        df = wrangle_the_data(df=df, dw_options=dw_options)

        df = df.reset_index(drop=True)

        st.subheader("Transformed dataframe")
        st.markdown("dataframe first rows")
        st.dataframe(df.head(line_display_count))
        st.markdown("Data frame numerical columns description")
        st.dataframe(df.describe())
        st.markdown("Dataframe's column types")
        st.write(df.dtypes)

    qs = st.sidebar if use_side_bar else st
    if use_side_bar:
        pass
    else:
        st.header(f"Step {step} - Plot customization")
        step += 1

    qs.subheader("Plot selection")

    # Select type
    plot_options = amp_consts.ALL_PLOTS if show_advanced_plots else amp_consts.BASIC_PLOTS
    plot_type = qs.selectbox(
        label="Plot type: ",
        options=plot_options,
        index=get_final_index(
            default_index=0, options=plot_options, key="plot_type", overrides=report
        ),
    )
    st.header(
        f"Step {step} - &darr; Plot {plot_type}{' customization (Widgets in sidebar)' if use_side_bar else ''} &darr;"
    )
    step += 1

    st.write(get_plot_help_digest(plot_type))
    if plot_type in [amp_consts.PLOT_LDA_2D, amp_consts.PLOT_NCA]:
        qs.warning(
            "If plotting fails, make sure that no variable is colinear with your target"
        )
    comment = report.get("comment", "")
    param_help_level = "mandatory"
    if comment:
        st.subheader("A word from the creator")
        st.markdown(comment)

    params = get_plot_docstring(plot_type).split("\nParameters")[1].split("\n")[2:]
    params_dict = defaultdict(list)
    current_key = ""
    for l in params:
        if l.startswith("    "):
            params_dict[current_key].append(l.replace("    ", " "))
        else:
            current_key = l
    plot_data_dict = {}

    param_initializer = ParamInitializer(
        parent=qs,
        params_doc=params_dict,
        overrides=report.get("params", {}),
        show_help=param_help_level,
    )

    # Select mode
    is_anim = (
        report.get("is_anim", False)
        if report
        else plot_type in amp_consts.PLOT_HAS_ANIM
        and qs.checkbox(label="Build animation", value=False)
    )

    if is_anim:
        if report:
            plot_data_dict["time_column"] = param_initializer(
                widget_params={},
                param_name="time_column",
                doc_override="A column from the dataframe used as key to build frames",
                lock=True,
            )
            plot_data_dict["animation_group"] = param_initializer(
                param_name="animation_group",
                widget_params=dict(
                    label="Animation category group",
                    options=[amp_consts.NONE_SELECTED]
                    + df.select_dtypes(include=["object", "datetime"]).columns.to_list(),
                    index=0,
                ),
            )
        else:
            amp_st_functs.set_anim_data(
                df=df,
                show_info=show_info,
                plot_data_dict=plot_data_dict,
                param_initializer=param_initializer,
                qs=qs,
            )

    qs.subheader("Basic parameters")
    num_columns = df.select_dtypes(include=[np.number]).columns.to_list()
    cat_columns = df.select_dtypes(include=["object", "datetime"]).columns.to_list()
    supervision_columns = df.select_dtypes(include=["object", "number"]).columns.to_list()
    all_columns = df.columns.to_list()

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
        elif plot_type in [
            amp_consts.PLOT_DENSITY_HEATMAP,
            amp_consts.PLOT_DENSITY_CONTOUR,
        ]:
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
        elif plot_type in [
            amp_consts.PLOT_DENSITY_HEATMAP,
            amp_consts.PLOT_DENSITY_CONTOUR,
        ]:
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
        plot_data_dict["x"] = param_initializer(
            param_name="x",
            widget_params=dict(label="X axis", options=x_columns, index=0),
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
            plot_data_dict["log_x"] = param_initializer(
                widget_type="checkbox",
                param_name="log_x",
                widget_params=dict(label="Log X axis?"),
            )
        if plot_data_dict["x"] == amp_consts.PICK_ONE:
            return

    if plot_type == amp_consts.PLOT_HISTOGRAM:
        plot_data_dict["histfunc"] = param_initializer(
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
        )
    elif plot_type in amp_consts.PLOT_HAS_Y:
        # Customize Y axis
        plot_data_dict["y"] = param_initializer(
            param_name="y",
            widget_params=dict(label="Y axis", options=y_columns, index=0),
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
            plot_data_dict["log_y"] = param_initializer(
                widget_type="checkbox",
                param_name="log_y",
                widget_params=dict(label="Log Y axis?"),
            )
        if plot_data_dict["y"] == amp_consts.PICK_ONE:
            return

    if plot_type == amp_consts.PLOT_SCATTER_3D:
        plot_data_dict["z"] = param_initializer(
            param_name="z",
            widget_params=dict(label="Z axis", options=z_columns, index=0),
        )
        if show_advanced_settings and plot_data_dict["z"] in num_columns:
            plot_data_dict["log_z"] = param_initializer(
                widget_type="checkbox",
                param_name="log_z",
                widget_params=dict(label="Log Z axis?"),
            )
        else:
            plot_data_dict["log_z"] = False
        if plot_data_dict["z"] == amp_consts.PICK_ONE:
            return

    # Target for supervised machine learning
    if plot_type in amp_consts.PLOT_HAS_TARGET:
        plot_data_dict["target"] = param_initializer(
            param_name="target",
            widget_params=dict(
                label="ML target:",
                options=[amp_consts.PICK_ONE] + supervision_columns,
                index=0,
            ),
        )
        if plot_data_dict["target"] == amp_consts.PICK_ONE:
            return
        elif (
            plot_data_dict["target"]
            in df.select_dtypes(include=[np.float]).columns.to_list()
            and show_info
        ):
            qs.info("Non discrete columns will be rounded")

    # Color column
    if plot_type in amp_consts.PLOT_HAS_COLOR:
        plot_data_dict["color"] = param_initializer(
            param_name="color",
            widget_params=dict(
                label="Use this column for color:",
                options=[amp_consts.NONE_SELECTED] + all_columns,
                index=0
                if plot_type not in amp_consts.PLOT_HAS_TARGET
                else all_columns.index(plot_data_dict["target"]) + 1,
            ),
        )

    # Ignored columns
    if plot_type in amp_consts.PLOT_HAS_IGNORE_COLUMNS:
        plot_data_dict["ignore_columns"] = param_initializer(
            widget_type="multiselect",
            param_name="ignore_columns",
            widget_params=dict(
                label="Ignore this columns when building the model:",
                options=all_columns,
                default=[plot_data_dict["target"]]
                if plot_type in amp_consts.PLOT_HAS_TARGET
                else [],
            ),
            doc_override="""
                This columns will be omitted when building the model, 
                but available for display.  
                Use this to avoid giving the answer to the question when building models.
                """,
        )
    if show_advanced_settings:
        qs.subheader("Advanced parameters:")
        # Common data
        available_marginals = [
            amp_consts.NONE_SELECTED,
            "rug",
            "box",
            "violin",
            "histogram",
        ]
        # Solver selection
        if plot_type in amp_consts.PLOT_HAS_SOLVER:
            plot_data_dict["solver"] = param_initializer(
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
            )
        # About NCA
        if plot_type in amp_consts.PLOT_HAS_NCOMP:
            plot_data_dict["n_components"] = param_initializer(
                widget_type="number_input",
                param_name="n_components",
                widget_params=dict(
                    label="Number of components",
                    min_value=2,
                    max_value=len(num_columns),
                    value=2,
                ),
            )
        if plot_type in amp_consts.PLOT_HAS_INIT:
            plot_data_dict["init"] = param_initializer(
                param_name="init",
                widget_params=dict(
                    label="Linear transformation init",
                    options=["auto", "pca", "lda", "identity", "random"],
                    index=0,
                ),
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
            qs.markdown("___")
        # Dot text
        if plot_type in amp_consts.PLOT_HAS_TEXT:
            plot_data_dict["text"] = param_initializer(
                param_name="text",
                widget_params=dict(
                    label="Text display column",
                    options=[amp_consts.NONE_SELECTED] + cat_columns,
                    index=0,
                ),
            )
        # Dot size
        if plot_type in amp_consts.PLOT_HAS_SIZE:
            plot_data_dict["size"] = param_initializer(
                param_name="size",
                widget_params=dict(
                    label="Use this column to select what dot size represents:",
                    options=[amp_consts.NONE_SELECTED] + num_columns,
                    index=0,
                ),
            )
            plot_data_dict["size_max"] = param_initializer(
                widget_type="number_input",
                param_name="size_max",
                widget_params=dict(
                    label="Max dot size", min_value=11, max_value=100, value=60
                ),
            )
        if plot_type in amp_consts.PLOT_HAS_SHAPE:
            plot_data_dict["symbol"] = param_initializer(
                param_name="symbol",
                widget_params=dict(
                    label="Select a column for the dot symbols",
                    options=[amp_consts.NONE_SELECTED] + cat_columns,
                    index=0,
                ),
            )
        if plot_type in amp_consts.PLOT_HAS_TREND_LINE:
            plot_data_dict["trendline"] = param_initializer(
                param_name="trendline",
                widget_params=dict(
                    label="Trend line mode",
                    options=[amp_consts.NONE_SELECTED, "ols", "lowess"],
                    format_func=lambda x: {
                        "ols": "Ordinary Least Squares ",
                        "lowess": "Locally Weighted scatterplot Smoothing",
                    }.get(x, x),
                ),
            )

        # Facet
        if plot_type in amp_consts.PLOT_HAS_FACET:
            plot_data_dict["facet_col"] = param_initializer(
                param_name="facet_col",
                widget_params=dict(
                    label="Use this column to split the plot in columns:",
                    options=[amp_consts.NONE_SELECTED] + cat_columns,
                    index=0,
                ),
            )
            if plot_data_dict["facet_col"] != amp_consts.NONE_SELECTED:
                plot_data_dict["facet_col_wrap"] = param_initializer(
                    widget_type="number_input",
                    param_name="facet_col_wrap",
                    widget_params=dict(
                        label="Wrap columns when more than x",
                        min_value=1,
                        max_value=20,
                        value=4,
                    ),
                )
            else:
                plot_data_dict["facet_col_wrap"] = 4
            plot_data_dict["facet_row"] = param_initializer(
                param_name="facet_row",
                widget_params=dict(
                    label="Use this column to split the plot in lines:",
                    options=[amp_consts.NONE_SELECTED] + cat_columns,
                    index=0,
                ),
            )
        # Histogram specific parameters
        if plot_type in amp_consts.PLOT_HAS_MARGINAL and is_anim:
            plot_data_dict["marginal"] = param_initializer(
                param_name="marginal",
                widget_params=dict(
                    label="Marginal", options=available_marginals, index=0
                ),
            )
            plot_data_dict["orientation"] = param_initializer(
                param_name="orientation",
                widget_params=dict(
                    label="orientation",
                    options=["v", "h"],
                    format_func=lambda x: {"v": "vertical", "h": "horizontal"}.get(
                        x, "unknown value"
                    ),
                    index=0,
                ),
            )
        if plot_type in amp_consts.PLOT_HAS_BAR_MODE:
            plot_data_dict["barmode"] = param_initializer(
                param_name="barmode",
                widget_params=dict(
                    label="bar mode", options=["group", "overlay", "relative"], index=2
                ),
            )
        if plot_type in amp_consts.PLOT_HAS_MARGINAL_XY and is_anim:
            plot_data_dict["marginal_x"] = param_initializer(
                param_name="marginal_x",
                widget_params=dict(
                    label="Marginals for X axis", options=available_marginals, index=0
                ),
            )
            plot_data_dict["marginal_y"] = param_initializer(
                param_name="marginal_y",
                widget_params=dict(
                    label="Marginals for Y axis", options=available_marginals, index=0
                ),
            )
        # Box plots and histograms
        if plot_type in amp_consts.PLOT_HAS_POINTS:
            plot_data_dict["points"] = param_initializer(
                param_name="points",
                widget_params=dict(
                    label="Select which points are displayed",
                    options=["none", "outliers", "suspectedoutliers", "all"],
                    index=1,
                ),
            )
            plot_data_dict["points"] = (
                plot_data_dict["points"] if plot_data_dict["points"] != "none" else False
            )
        # Box plots
        if plot_type == amp_consts.PLOT_BOX:
            plot_data_dict["notched"] = param_initializer(
                param_name="notched",
                widget_type="checkbox",
                widget_params=dict(label="Use notches?", value=False),
            )
        # Violin plots
        if plot_type == amp_consts.PLOT_VIOLIN:
            plot_data_dict["box"] = param_initializer(
                param_name="box", widget_params=dict(label="Show boxes", value=False),
            )
            plot_data_dict["violinmode"] = param_initializer(
                param_name="violinmode",
                widget_params=dict(
                    label="Violin display mode", options=["group", "overlay"]
                ),
            )
        # Density heat map
        if plot_type in amp_consts.PLOT_HAS_BINS:
            plot_data_dict["nbinsx"] = param_initializer(
                widget_type="number_input",
                param_name="nbinsx",
                widget_params=dict(
                    label="Number of bins in the X axis",
                    min_value=1,
                    max_value=1000,
                    value=20,
                ),
            )
            plot_data_dict["nbinsy"] = param_initializer(
                widget_type="number_input",
                param_name="nbinsy",
                widget_params=dict(
                    label="Number of bins in the Y axis",
                    min_value=1,
                    max_value=1000,
                    value=20,
                ),
            )
        # Density contour map
        if plot_type == amp_consts.PLOT_DENSITY_CONTOUR:
            plot_data_dict["fill_contours"] = param_initializer(
                param_name="fill_contours",
                widget_params=dict(label="Fill contours", value=False),
            )
        # PCA loadings
        if plot_type in amp_consts.PLOT_HAS_LOADINGS:
            plot_data_dict["show_loadings"] = param_initializer(
                param_name="show_loadings",
                widget_type="checkbox",
                widget_params=dict(label="Show loadings", value=False),
            )
        # Correlation plot
        if plot_type == amp_consts.PLOT_CORR_MATRIX:
            plot_data_dict["corr_method"] = param_initializer(
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
            )
        # Matrix plot
        if plot_type == amp_consts.PLOT_SCATTER_MATRIX:
            plot_data_dict["matrix_diag"] = param_initializer(
                param_name="matrix_diag",
                widget_params=dict(
                    label="diagonal",
                    options=["Nothing", "Histogram", "scatter"],
                    index=1,
                ),
            )
            plot_data_dict["matrix_up"] = param_initializer(
                param_name="matrix_up",
                widget_params=dict(
                    label="Upper triangle",
                    options=["Nothing", "scatter", "2D histogram"],
                    index=1,
                ),
            )
            plot_data_dict["matrix_down"] = param_initializer(
                param_name="matrix_down",
                widget_params=dict(
                    label="Lower triangle",
                    options=["Nothing", "scatter", "2D histogram"],
                    index=1,
                ),
            )

        # Hover data
        if plot_type in amp_consts.PLOT_HAS_CUSTOM_HOVER_DATA:
            plot_data_dict["hover_name"] = param_initializer(
                param_name="hover_name",
                widget_params=dict(
                    label="Hover name:",
                    options=[amp_consts.NONE_SELECTED] + cat_columns,
                    index=0,
                ),
            )
            plot_data_dict["hover_data"] = param_initializer(
                widget_type="multiselect",
                param_name="hover_data",
                widget_params=dict(
                    label="Add columns to hover data",
                    options=df.columns.to_list(),
                    default=[],
                ),
            )
    else:
        if plot_type == amp_consts.PLOT_SCATTER_MATRIX:
            plot_data_dict["matrix_diag"] = "Histogram"
            plot_data_dict["matrix_up"] = "scatter"
            plot_data_dict["matrix_down"] = "scatter"
        if plot_type == amp_consts.PLOT_PCA_2D:
            plot_data_dict["show_loadings"] = False
        if plot_type == amp_consts.PLOT_CORR_MATRIX:
            plot_data_dict["corr_method"] = "pearson"

    if show_advanced_settings:
        plot_data_dict["height"] = int(
            param_initializer(
                param_name="height",
                widget_params=dict(
                    label="Plot height in pixels",
                    options=[
                        400,
                        600,
                        700,
                        800,
                        900,
                        1000,
                        1200,
                        1400,
                        1600,
                        1800,
                        2000,
                    ],
                    index=4,
                ),
            )
        )
        # Template
        available_templates = list(pio.templates.keys())
        plot_data_dict["template"] = param_initializer(
            param_name="template",
            widget_params=dict(
                label="Plot template (theme): ",
                options=available_templates,
                index=available_templates.index(pio.templates.default),
            ),
        )
    else:
        plot_data_dict["height"] = 900
        plot_data_dict["template"] = pio.templates.default

    if not reporting_mode:
        report_comment = st.text_area(
            label="Add a comment to your report (markdown accepted)"
        )
    else:
        report_comment = ""

    if plot_type in amp_consts.PLOT_HAS_MODEL_DATA:
        show_variance_ratio = st.checkbox(
            label="Show explained variance",
            value=report.get("show_variance_ratio", False),
        )
        show_additional_model_data = st.checkbox(
            label=(
                lambda x: {
                    amp_consts.PLOT_PCA_2D: "Components details",
                    amp_consts.PLOT_PCA_3D: "Components details",
                    amp_consts.PLOT_NCA: "Linear transformation learned",
                }.get(x, "Error")
            )(plot_type),
            value=report.get("show_additional_model_data", False),
        )
    else:
        show_variance_ratio = False
        show_additional_model_data = False

    if (
        not reporting_mode
        and (is_anim or not real_time_render)
        and not st.button(label="Render", key="render_plot")
    ):
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
        for c in all_columns:
            if df[c].isnull().values.any():
                st.markdown(f"- {c} has {df[c].isnull().sum()} NA values")
        df = df.dropna(axis="index")

    fig_data = build_plot(
        is_anim=is_anim,
        plot_type=plot_type,
        df=df.copy(),
        progress=update_progress,
        **plot_data_dict,
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

            b64 = base64.b64encode(
                json.dumps(
                    {
                        "params": plot_data_dict,
                        "plot_type": plot_type,
                        "is_anim": is_anim,
                        "dataframe": df.to_dict(),
                        "comment": report_comment,
                        "ui_mode": ui_mode,
                        "param_help_level": param_help_level,
                        "real_time_render": real_time_render,
                        "show_info": show_info,
                        "use_side_bar": use_side_bar,
                        "show_advanced_plots": show_advanced_plots,
                        "show_dw_options": show_dw_options,
                        "show_advanced_settings": show_advanced_settings,
                        "force_wide_mode": force_wide_mode,
                        "show_variance_ratio": show_variance_ratio,
                        "show_additional_model_data": show_additional_model_data,
                    },
                    default=lambda x: str(x)
                    if isinstance(x, datetime.datetime)
                    else None,
                ).encode("utf-8")
            ).decode()
            href = f"""<a href="data:file/json;base64,{b64}" download="plot_configuration.json">
            Download plot parameters as JSON file</a> 
            - right-click and save as &lt;some_name&gt;.html"""
            st.markdown(href, unsafe_allow_html=True)
            st.markdown("")

        if "model_data" in fig_data:
            model_data = fig_data.get("model_data", None)
            if hasattr(model_data, "explained_variance_ratio_") and show_variance_ratio:
                df_ev = pd.DataFrame.from_dict(
                    {
                        "pc": [
                            f"PC{i}"
                            for i in range(len(model_data.explained_variance_ratio_))
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
                and show_additional_model_data
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
