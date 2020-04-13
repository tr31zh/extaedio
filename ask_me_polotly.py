import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Ask me polotly")

st.markdown(
    """After starting with one of the worsts attempts at word play in the 
history of humanity, this page will ask you a sereiess of questions in order to 
build a visualization."""
)

st.info("By the way, some questions are on the side bar")

url_csse = "url_csse"


def format_csv_link(url):
    if url == url_csse:
        return "CSSE VOVID-19"
    else:
        return url


def customize_plot():

    selected_file = st.selectbox(
        label="Source file: ",
        options=["Pick one...", url_csse,],
        index=0,
        format_func=format_csv_link,
    )

    if selected_file == "Pick one...":
        return
    elif selected_file == url_csse:
        from dw_cssegis_data import get_wrangled_cssegis_df

        df = get_wrangled_cssegis_df(allow_cache=True)

    st.header("Selected data frame")
    st.dataframe(df.describe())

    st.header("Plotting options")

    # Select mode
    plot_mode = st.selectbox(
        label="Plot mode: ", options=["Pick one...", "Static", "Animation"], index=0,
    )
    if plot_mode == "Pick one...":
        return

    # Select type
    allowed_plots = [
        "Pick one...",
        "scatter",
        "bar",
    ]
    if plot_mode == "Static":
        allowed_plots.append("line")
    plot_type = st.selectbox(label="Plot type: ", options=allowed_plots, index=0,)
    if plot_type == "Pick one...":
        return

    # Filter
    st.header("Filtering")
    filter_columns = st.multiselect(
        label="Select which columns you want to filter:",
        options=(df.select_dtypes(include=["object", "datetime"]).columns.to_list()),
        default=None,
    )
    filters = {}
    for column in filter_columns:
        st.subheader(f"{column}: ")
        sellect_all = st.checkbox(label=f"{column} Select all:")
        elements = list(df[column].unique())
        filters[column] = st.multiselect(
            label=f"Select wich {column} to include",
            options=elements,
            default=None if not sellect_all else elements,
        )

    # Preview dataframe
    if len(filters) > 0:
        for k, v in filters.items():
            df = df[df[k].isin(v)]
        st.subheader("filtered data frame")
        st.dataframe(df.describe())

    st.header("Plot customization")
    # Customize X axis
    if plot_type in ["scatter", "line"]:
        x_axis = st.selectbox(
            label="X axis", options=["Pick one..."] + df.columns.to_list(), index=0
        )
    elif plot_type == "bar":
        x_axis = st.selectbox(
            label="X axis",
            options=["Pick one..."]
            + df.select_dtypes(include=["object", "datetime"]).columns.to_list(),
            index=0,
        )
    if x_axis == "Pick one...":
        return

    # Customize Y axis
    y_axis = st.selectbox(
        label="Y axis", options=["Pick one..."] + df.columns.to_list(), index=0
    )
    if y_axis == "Pick one...":
        return

    # Color column
    color = st.selectbox(
        label="Use this column for color separation:",
        options=["Pick one..."]
        + df.select_dtypes(include=["object", "datetime"]).columns.to_list(),
        index=0,
    )

    if st.button(label="Render plot"):
        common_dict = {
            "data_frame": df,
            "x": x_axis,
            "y": y_axis,
            "color": color if color != "Pick one..." else None,
            "width": 800,
            "height": 600,
        }
        if plot_type == "scatter":
            fig = px.scatter(**common_dict)
        elif plot_type == "line":
            fig = px.line(**common_dict)
        elif plot_type == "bar":
            fig = px.bar(**common_dict)
        else:
            fig = None

        if fig is not None:
            st.plotly_chart(figure_or_data=fig)


customize_plot()
