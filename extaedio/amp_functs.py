import io
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler

from dw_cssegis_data import get_wrangled_cssegis_df

import amp_consts


def format_csv_link(url):
    if url == amp_consts.URL_CSSE:
        return "COVID-19 - CSSE, modified version"
    elif url == amp_consts.URL_ECDC_COVID:
        return "COVID-19 - European Centre for Disease Prevention and Control"
    elif url == amp_consts.URL_COVID_DATA:
        return "COVID-19 - Github coviddata"
    elif url == amp_consts.URL_IRIS:
        return "Iris plotly express data set"
    elif url == amp_consts.URL_CARSHARE:
        return "Carshare plotly express data set"
    elif url == amp_consts.URL_GAPMINDER:
        return "Gapminder plotly express data set"
    elif url == amp_consts.URL_TIPS:
        return "Tips plotly express data set"
    elif url == amp_consts.URL_WIND:
        return "Wind plotly express data set"
    else:
        return url


def get_plot_docstring(plot_type) -> str:
    if plot_type == amp_consts.PLOT_SCATTER:
        return px.scatter.__doc__
    elif plot_type == amp_consts.PLOT_SCATTER_3D:
        return px.scatter_3d.__doc__
    elif plot_type == amp_consts.PLOT_LINE:
        return px.line.__doc__
    elif plot_type == amp_consts.PLOT_BAR:
        return px.bar.__doc__
    elif plot_type == amp_consts.PLOT_HISTOGRAM:
        return px.histogram.__doc__
    elif plot_type == amp_consts.PLOT_BOX:
        return px.box.__doc__
    elif plot_type == amp_consts.PLOT_VIOLIN:
        return px.violin.__doc__
    elif plot_type == amp_consts.PLOT_DENSITY_HEATMAP:
        return px.density_heatmap.__doc__
    elif plot_type == amp_consts.PLOT_DENSITY_CONTOUR:
        return px.density_contour.__doc__
    elif plot_type == amp_consts.PLOT_PARALLEL_CATEGORIES:
        return px.parallel_categories.__doc__
    elif plot_type == amp_consts.PLOT_PARALLEL_COORDINATES:
        return px.parallel_coordinates.__doc__
    else:
        return px.scatter.__doc__

    return doc


def get_plot_help_digest(plot_type) -> str:
    if plot_type == amp_consts.PLOT_SCATTER:
        doc, _ = px.scatter.__doc__.split("\nParameters")
    elif plot_type == amp_consts.PLOT_SCATTER_3D:
        doc, _ = px.scatter_3d.__doc__.split("\nParameters")
    elif plot_type == amp_consts.PLOT_LINE:
        doc, _ = px.line.__doc__.split("\nParameters")
    elif plot_type == amp_consts.PLOT_BAR:
        doc, _ = px.bar.__doc__.split("\nParameters")
    elif plot_type == amp_consts.PLOT_HISTOGRAM:
        doc, _ = px.histogram.__doc__.split("\nParameters")
    elif plot_type == amp_consts.PLOT_BOX:
        doc, _ = px.box.__doc__.split("\nParameters")
    elif plot_type == amp_consts.PLOT_VIOLIN:
        doc, _ = px.violin.__doc__.split("\nParameters")
    elif plot_type == amp_consts.PLOT_DENSITY_HEATMAP:
        doc, _ = px.density_heatmap.__doc__.split("\nParameters")
    elif plot_type == amp_consts.PLOT_DENSITY_CONTOUR:
        doc, _ = px.density_contour.__doc__.split("\nParameters")
    elif plot_type == amp_consts.PLOT_PARALLEL_CATEGORIES:
        doc, _ = px.parallel_categories.__doc__.split("\nParameters")
    elif plot_type == amp_consts.PLOT_PARALLEL_COORDINATES:
        doc, _ = px.parallel_coordinates.__doc__.split("\nParameters")
    elif plot_type == amp_consts.PLOT_SCATTER_MATRIX:
        doc = "Plot a scatter mattrix for all selected columns"
    elif plot_type in [amp_consts.PLOT_PCA_2D]:
        doc = """
        **Principal component analysis (2 dimensions)**  
        Given a collection of points in two, three, or higher dimensional space, 
        a "best fitting" line can be defined as one that minimizes the average squared distance 
        from a point to the line. The next best-fitting line can be similarly chosen from 
        directions perpendicular to the first. Repeating this process yields an orthogonal 
        basis in which different individual dimensions of the data are uncorrelated. 
        These basis vectors are called principal components, and several related procedures 
        principal component analysis (PCA).
        """
    elif plot_type in [amp_consts.PLOT_PCA_3D]:
        doc = """
        **Principal component analysis (3 dimensions)**  
        Given a collection of points in two, three, or higher dimensional space, 
        a "best fitting" line can be defined as one that minimizes the average squared distance 
        from a point to the line. The next best-fitting line can be similarly chosen from 
        directions perpendicular to the first. Repeating this process yields an orthogonal 
        basis in which different individual dimensions of the data are uncorrelated. 
        These basis vectors are called principal components, and several related procedures 
        principal component analysis (PCA).
        """
    elif plot_type == amp_consts.PLOT_CORR_MATRIX:
        doc = "Plot correlation matrix"
    elif plot_type == amp_consts.PLOT_LDA_2D:
        doc = """
        **Linear discriminant analysis (LDA)**  
        A generalization of Fisher's linear discriminant, a method used in statistics, 
        pattern recognition, and machine learning to find a linear combination of features 
        that characterizes or separates two or more classes of objects or events. The resulting 
        combination may be used as a linear classifier, or, more commonly, for dimensionality 
        reduction before later classification.
        """
    elif plot_type == amp_consts.PLOT_QDA_2D:
        doc = amp_consts.PLOT_QDA_2D + "On 2D PCA"
    elif plot_type == amp_consts.PLOT_NCA:
        doc = """
        **Neighborhood components analysis**  
        A supervised learning method for classifying multivariate data into distinct classes 
        according to a given distance metric over the data. Functionally, it serves the same 
        purposes as the K-nearest neighbors algorithm, and makes direct use of a related concept 
        termed stochastic nearest neighbors.
        """
    else:
        doc = "Unknown"

    return doc


def filter_none(value):
    return None if value == amp_consts.NONE_SELECTED else value


def add_histogram(fig, x, index, name="", marker=None, legend=False):
    fig.add_trace(
        go.Histogram(
            x=x,
            showlegend=legend,
            marker={} if marker is None else {"color": marker},
            name=name,
        ),
        row=index,
        col=index,
    )


def add_scatter(fig, x, y, row, col, marker=None, opacity=0.5, legend=False, name="²"):
    fig.add_scatter(
        x=x,
        y=y,
        mode="markers",
        marker={} if marker is None else {"color": marker},
        opacity=opacity,
        showlegend=legend,
        name=name,
        row=row,
        col=col,
    )


def add_2d_hist(fig, x, y, row, col, legend=False):
    fig.add_trace(
        go.Histogram2dContour(
            x=x,
            y=y,
            reversescale=True,
            xaxis="x",
            yaxis="y",
            showlegend=legend,
        ),
        row=row,
        col=col,
    )


def get_dataframe_from_url(url):
    if url == amp_consts.PICK_ONE:
        return None
    elif isinstance(url, io.StringIO):
        return pd.read_csv(url)
    elif url == amp_consts.URL_CSSE:
        return get_wrangled_cssegis_df(allow_cache=False)
    elif url == amp_consts.URL_IRIS:
        return px.data.iris()
    elif url == amp_consts.URL_CARSHARE:
        return px.data.carshare()
    elif url == amp_consts.URL_GAPMINDER:
        return px.data.gapminder()
    elif url == amp_consts.URL_TIPS:
        return px.data.tips()
    elif url == amp_consts.URL_WIND:
        return px.data.wind()
    elif url == amp_consts.URL_ELECTION:
        return px.data.election()
    elif isinstance(url, str):
        try:
            return pd.read_csv(url)
        except:
            return None
    elif isinstance(url, bytes):
        try:
            return pd.read_csv(io.BytesIO(url), encoding="utf8")
        except:
            return None
    else:
        return None


def build_plot(is_anim, plot_type, df, progress=None, **kwargs) -> dict:

    params = dict(**kwargs)
    for k, v in params.items():
        if v == amp_consts.NONE_SELECTED:
            params[k] = filter_none(params[k])
    num_columns = df.select_dtypes(include=[np.number]).columns.to_list()

    if is_anim:
        time_column = params.pop("time_column", "")
        if (
            time_column
            in df.select_dtypes(
                include=[np.datetime64, "datetime", "datetime64", "datetime64[ns, UTC]"]
            ).columns.to_list()
        ):
            df["time_step"] = df[time_column].dt.strftime("%Y/%m/%d %H:%M:%S")
            afc = "time_step"
        else:
            afc = time_column
        params["animation_frame"] = afc
        df = df.sort_values([afc])
        if plot_type not in [
            amp_consts.PLOT_PCA_3D,
            amp_consts.PLOT_PCA_2D,
            amp_consts.PLOT_LDA_2D,
            amp_consts.PLOT_QDA_2D,
            amp_consts.PLOT_NCA,
        ]:
            x = params.get("x")
            params["range_x"] = (
                None if x not in num_columns else [df[x].min(), df[x].max()]
            )
            y = params.get("y")
            params["range_y"] = (
                None if y not in num_columns else [df[y].min(), df[y].max()]
            )
            if plot_type in [amp_consts.PLOT_SCATTER_3D, amp_consts.PLOT_PCA_3D]:
                z = params.get("z")
                params["range_z"] = (
                    None if z not in num_columns else [df[z].min(), df[z].max()]
                )

    params["data_frame"] = df

    fig = None
    model_data = None
    column_names = None
    class_names = None

    if plot_type == amp_consts.PLOT_SCATTER:
        fig = px.scatter(**params)
    elif plot_type == amp_consts.PLOT_SCATTER_3D:
        fig = px.scatter_3d(**params)
    elif plot_type == amp_consts.PLOT_LINE:
        fig = px.line(**params)
    elif plot_type == amp_consts.PLOT_BAR:
        fig = px.bar(**params)
    elif plot_type == amp_consts.PLOT_HISTOGRAM:
        if "orientation" in params and params.get("orientation") == "h":
            params["x"], params["y"] = None, params["x"]
        fig = px.histogram(**params)
    elif plot_type == amp_consts.PLOT_BOX:
        fig = px.box(**params)
    elif plot_type == amp_consts.PLOT_VIOLIN:
        fig = px.violin(**params)
    elif plot_type == amp_consts.PLOT_DENSITY_HEATMAP:
        fig = px.density_heatmap(**params)
    elif plot_type == amp_consts.PLOT_DENSITY_CONTOUR:
        fc = params.pop("fill_contours") is True
        fig = px.density_contour(**params)
        if fc:
            fig.update_traces(contours_coloring="fill", contours_showlabels=True)
    elif plot_type == amp_consts.PLOT_PARALLEL_CATEGORIES:
        fig = px.parallel_categories(**params)
    elif plot_type == amp_consts.PLOT_PARALLEL_COORDINATES:
        fig = px.parallel_coordinates(**params)
    elif plot_type == amp_consts.PLOT_SCATTER_MATRIX:
        fig = make_subplots(
            rows=len(num_columns),
            cols=len(num_columns),
            shared_xaxes=True,
            row_titles=num_columns,
        )
        color_column = params.get("color")
        if color_column is not None:
            template_colors = pio.templates[params.get("template")].layout["colorway"]
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
        legend_added = False
        step = 0
        total = len(num_columns) ** 2
        matrix_diag = params["matrix_diag"]
        matrix_up = params["matrix_up"]
        matrix_down = params["matrix_down"]
        for i, c in enumerate(num_columns):
            for j, l in enumerate(num_columns):
                progress(step, total)
                step += 1
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
                            fig=fig,
                            x=df[c],
                            y=df[l],
                            row=j + 1,
                            col=i + 1,
                        )
                    elif mtx_plot_kind == "2D histogram":
                        add_2d_hist(fig=fig, x=df[c], y=df[l], row=j + 1, col=i + 1)
                else:
                    for color_parse, cat in zip(
                        template_colors, df[color_column].unique()
                    ):
                        df_cat = df[df[color_column] == cat]
                        if mtx_plot_kind == "Histogram":
                            add_histogram(
                                fig=fig,
                                x=df_cat[c],
                                index=i + 1,
                                name=cat,
                                marker=color_parse,
                                legend=not legend_added,
                            )
                        elif mtx_plot_kind == "Scatter":
                            add_scatter(
                                fig=fig,
                                x=df_cat[c],
                                y=df_cat[l],
                                row=j + 1,
                                col=i + 1,
                                name=cat,
                                marker=color_parse,
                                legend=not legend_added,
                            )
                    legend_added = True
                fig.update_xaxes(
                    title_text=c,
                    row=j + 1,
                    col=i + 1,
                )
                if c == 0:
                    fig.update_yaxes(
                        title_text=l,
                        row=j + 1,
                        col=i + 1,
                    )
        fig.update_layout(barmode="stack")
    elif plot_type in [amp_consts.PLOT_PCA_2D, amp_consts.PLOT_PCA_3D]:
        X = df.loc[:, num_columns]
        ignored_columns = params.pop("ignore_columns", [])
        if ignored_columns:
            X = X.drop(
                list(set(ignored_columns).intersection(set(X.columns.to_list()))), axis=1
            )
        column_names = X.columns.to_list()
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        model_data = PCA()
        x_new = model_data.fit_transform(X)
        pc1_lbl = f"PC1 ({model_data.explained_variance_ratio_[0] * 100:.2f}%)"
        pc2_lbl = f"PC2 ({model_data.explained_variance_ratio_[1] * 100:.2f}%)"
        x = x_new[:, 0]
        y = x_new[:, 1]
        df[pc1_lbl] = x * (1.0 / (x.max() - x.min()))
        df[pc2_lbl] = y * (1.0 / (y.max() - y.min()))
        params["x"] = pc1_lbl
        params["y"] = pc2_lbl
        if is_anim:
            params["range_x"] = [-1, 1]
            params["range_y"] = [-1, 1]
        sl = params.pop("show_loadings") is True
        if plot_type in [amp_consts.PLOT_PCA_3D]:
            z = x_new[:, 2]
            pc3_lbl = f"PC3 ({model_data.explained_variance_ratio_[2] * 100:.2f}%)"
            df[pc3_lbl] = z * (1.0 / (z.max() - z.min()))
            params["z"] = pc3_lbl
            if is_anim:
                params["range_z"] = [-1, 1]
            fig = px.scatter_3d(**params)
            if sl:
                loadings = np.transpose(model_data.components_[0:3, :])
                m = 1 / np.amax(loadings)
                loadings = loadings * m
                xc, yc, zc = [], [], []
                for i in range(loadings.shape[0]):
                    xc.extend([0, loadings[i, 0], None])
                    yc.extend([0, loadings[i, 1], None])
                    zc.extend([0, loadings[i, 2], None])
                fig.add_trace(
                    go.Scatter3d(
                        x=xc,
                        y=yc,
                        z=zc,
                        mode="lines",
                        name="Loadings",
                        showlegend=False,
                        line=dict(color="black"),
                        opacity=0.3,
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=loadings[:, 0],
                        y=loadings[:, 1],
                        z=loadings[:, 2],
                        mode="text",
                        text=num_columns,
                        opacity=0.7,
                        name="Loadings",
                    ),
                )
        else:
            fig = px.scatter(**params)
            if sl:
                loadings = np.transpose(model_data.components_[0:2, :])
                m = 1 / np.amax(loadings)
                loadings = loadings * m
                xc, yc = [], []
                for i in range(loadings.shape[0]):
                    xc.extend([0, loadings[i, 0], None])
                    yc.extend([0, loadings[i, 1], None])
                fig.add_trace(
                    go.Scatter(
                        x=xc,
                        y=yc,
                        mode="lines",
                        name="Loadings",
                        showlegend=False,
                        line=dict(color="black"),
                        opacity=0.3,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=loadings[:, 0],
                        y=loadings[:, 1],
                        mode="text",
                        text=num_columns,
                        opacity=0.7,
                        name="Loadings",
                    ),
                )
    elif plot_type in [amp_consts.PLOT_LDA_2D, amp_consts.PLOT_QDA_2D]:
        X = df.loc[:, num_columns]
        ignored_columns = params.pop("ignore_columns", [])
        if ignored_columns:
            X = X.drop(
                list(set(ignored_columns).intersection(set(X.columns.to_list()))), axis=1
            )
        column_names = X.columns.to_list()
        if params["target"] in df.select_dtypes(include=["object"]).columns.to_list():
            t = df[params["target"]].astype("category").cat.codes
        elif params["target"] in df.select_dtypes(include=[np.float]).columns.to_list():
            t = df[params["target"]].astype("int")
        else:
            t = df[params["target"]]
        class_names = df[params["target"]].unique()
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        if plot_type == amp_consts.PLOT_LDA_2D:
            model_data = LinearDiscriminantAnalysis(solver=params.pop("solver", "svd"))
        elif plot_type == amp_consts.PLOT_QDA_2D:
            model_data = QuadraticDiscriminantAnalysis(store_covariance=True)
        x_new = model_data.fit(X, y=t).transform(X)
        pc1_lbl = f"PC1 ({model_data.explained_variance_ratio_[0] * 100:.2f}%)"
        pc2_lbl = f"PC2 ({model_data.explained_variance_ratio_[1] * 100:.2f}%)"
        x = x_new[:, 0]
        y = x_new[:, 1]
        df[pc1_lbl] = x * (1.0 / (x.max() - x.min()))
        df[pc2_lbl] = y * (1.0 / (y.max() - y.min()))
        params["x"] = pc1_lbl
        params["y"] = pc2_lbl
        if is_anim:
            params["range_x"] = [-1, 1]
            params["range_y"] = [-1, 1]
        params.pop("target")
        sl = params.pop("show_loadings") is True
        fig = px.scatter(**params)
        if sl:
            loadings = np.transpose(model_data.coef_[0:2, :])
            m = 1 / np.amax(loadings)
            loadings = loadings * m
            xc, yc = [], []
            for i in range(loadings.shape[0]):
                xc.extend([0, loadings[i, 0], None])
                yc.extend([0, loadings[i, 1], None])
            fig.add_trace(
                go.Scatter(
                    x=xc,
                    y=yc,
                    mode="lines",
                    name="Loadings",
                    showlegend=False,
                    line=dict(color="black"),
                    opacity=0.3,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=loadings[:, 0],
                    y=loadings[:, 1],
                    mode="text",
                    text=num_columns,
                    opacity=0.7,
                    name="Loadings",
                ),
            )
    elif plot_type in [amp_consts.PLOT_NCA]:
        X = df.loc[:, num_columns]
        ignored_columns = params.pop("ignore_columns", [])
        if ignored_columns:
            X = X.drop(
                list(set(ignored_columns).intersection(set(X.columns.to_list()))), axis=1
            )
        column_names = X.columns.to_list()
        if params["target"] in df.select_dtypes(include=["object"]).columns.to_list():
            t = df[params["target"]].astype("category").cat.codes
        elif params["target"] in df.select_dtypes(include=[np.float]).columns.to_list():
            t = df[params["target"]].astype("int")
        else:
            t = df[params["target"]]
        class_names = df[params["target"]].unique()
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        model_data = NeighborhoodComponentsAnalysis(
            init=params.pop("init", "auto"),
            n_components=min(len(column_names), params.pop("n_components", 2)),
        )
        x_new = model_data.fit(X, y=t).transform(X)
        df["x_nca"] = x_new[:, 0]
        df["y_nca"] = x_new[:, 1]
        params["x"] = "x_nca"
        params["y"] = "y_nca"
        if is_anim:
            params["range_x"] = [-1, 1]
            params["range_y"] = [-1, 1]
        params.pop("target")
        fig = px.scatter(**params)
    elif plot_type == amp_consts.PLOT_CORR_MATRIX:
        fig = px.imshow(
            df[num_columns].corr(method=params.get("corr_method")).values,
            x=num_columns,
            y=num_columns,
        )
    else:
        fig = None

    if plot_type in amp_consts.PLOT_IS_3D:
        fig.update_layout(scene={"aspectmode": "cube"})

    if fig is not None:
        fig.update_layout(
            height=params["height"],
            template=params["template"],
            legend={"traceorder": "normal"},
        )

    return {
        k: v
        for k, v in zip(
            ["figure", "model_data", "column_names", "class_names"],
            [fig, model_data, column_names, class_names],
        )
        if v is not None
    }
