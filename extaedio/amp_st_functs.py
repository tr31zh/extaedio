import pandas as pd
import re
import io

import streamlit as st

import amp_consts
import amp_functs


@st.cache
def get_df_from_url(url):
    return amp_functs.get_dataframe_from_url(url)


def load_dataframe(step: int, show_info):
    st.header(f"Step {step} - Load CSV file (dataframe")

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
        format_func=amp_functs.format_csv_link,
    )
    if selected_file == amp_consts.URL_LOCAL_FILE:
        # st.set_option("deprecation.showfileUploaderEncoding", False)
        selected_file = st.file_uploader(
            label="Select file to upload",
            type=["csv"],
        )
        if selected_file is None:
            return None
        return pd.read_csv(selected_file).copy().reset_index(drop=True)
    elif selected_file == amp_consts.URL_DISTANT_FILE:
        selected_file = st.text_input(label="Paste web URL", value="")
        st.write(selected_file)
        if not (st.button(label="Download file", key="grab_file") and selected_file):
            return None
    df_loaded = get_df_from_url(url=selected_file)
    if df_loaded is None:
        return None
    return df_loaded.copy().reset_index(drop=True)


def set_anim_data(df, show_info: bool, plot_data_dict: dict, param_initializer, qs):
    if show_info:
        qs.info(
            """Animation will be rendered when the **render** 
            button bellow the plot title is pressed
            """
        )

    time_col_match = "date|time|day|year|month"
    time_columns = [
        c
        for c in df.columns.to_list()
        if any(re.findall(pattern=time_col_match, string=c, flags=re.IGNORECASE))
    ]
    if not time_columns:
        time_columns = [amp_consts.PICK_ONE]
    time_columns.extend(
        [col for col in df.columns.to_list() if col.lower() not in time_columns]
    )
    time_columns.sort(key=lambda x: 0 if x.lower() == "date" else 1)
    plot_data_dict["time_column"] = param_initializer(
        widget_params=dict(label="Date/time column: ", options=time_columns, index=0),
        param_name="time_column",
        doc_override="A column from the dataframe used as key to build frames",
    )
    if plot_data_dict["time_column"] != amp_consts.PICK_ONE:
        new_time_column = plot_data_dict["time_column"] + "_" + "pmgd"
    else:
        return
    is_default_time_col = any(
        re.findall(
            pattern=time_col_match,
            string=plot_data_dict["time_column"],
            flags=re.IGNORECASE,
        )
    )
    if plot_data_dict["time_column"] != amp_consts.PICK_ONE and (
        is_default_time_col
        or qs.checkbox(
            label="Convert to date?",
            value=is_default_time_col,
        )
    ):
        try:
            cf_columns = [c.casefold() for c in df.columns.to_list()]
            if (
                (plot_data_dict["time_column"].lower() in ["year", "month", "day"])
                and (
                    len(set(("year", "month", "day")).intersection(set(cf_columns))) > 0
                )
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
                        + df[src_columns[cf_columns.index("day".casefold())]].astype(
                            "str"
                        )
                    )
                else:
                    date_series = date_series + "-01"
                df[new_time_column] = pd.to_datetime(date_series)
            else:
                try:
                    df[new_time_column] = pd.to_datetime(
                        df[plot_data_dict["time_column"]]
                    )
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
    if plot_data_dict["time_column"] != amp_consts.PICK_ONE:
        plot_data_dict["time_column"] = new_time_column
        qs.info(f"Frames: {len(df[new_time_column].unique())}")
        plot_data_dict["animation_group"] = param_initializer(
            param_name="animation_group",
            widget_params=dict(
                label="Animation category group",
                options=[amp_consts.NONE_SELECTED]
                + df.select_dtypes(include=["object", "datetime"]).columns.to_list(),
                index=0,
            ),
        )
    if show_info:
        qs.info(
            """
            Select the main column of your dataframe as the category group.  
            For example if using *gapminder* select country.  
            If no column is selected the animation may jitter.
            """
        )
