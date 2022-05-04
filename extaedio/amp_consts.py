PICK_ONE = "Pick one..."
NONE_SELECTED = "None selected"

URL_CSSE = "url_csse"
URL_COVID_DATA = "https://raw.githubusercontent.com/coviddata/coviddata/master/data/sources/jhu_csse/standardized/standardized.csv"
URL_ECDC_COVID = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
URL_IRIS = "url_iris"
URL_TIPS = "url_tips"
URL_GAPMINDER = "url_gapminder"
URL_WIND = "url_wind"
URL_ELECTION = "url_election"
URL_CARSHARE = "url_carshare"
URL_LOCAL_FILE = "Browse computer"
URL_DISTANT_FILE = "Distant URL"

AVAILABLE_URLS = [
    PICK_ONE,
    URL_LOCAL_FILE,
    URL_DISTANT_FILE,
    URL_CSSE,
    URL_GAPMINDER,
    URL_IRIS,
    URL_CARSHARE,
    URL_ELECTION,
    URL_TIPS,
    URL_WIND,
    URL_COVID_DATA,
    URL_ECDC_COVID,
]

PLOT_SCATTER = "Scatter"
PLOT_SCATTER_3D = "Scatter 3D"
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
PLOT_PCA_SCATTER = "PCA scatter matrix"
PLOT_LDA_2D = "Linear Discriminant Analysis"
PLOT_QDA_2D = "Quadratic Discriminant Analysis"
PLOT_CORR_MATRIX = "Correlation matrix"
PLOT_NCA = "Neighborhood Component Analysis"

BASIC_PLOTS = [
    PLOT_SCATTER,
    PLOT_BAR,
    PLOT_HISTOGRAM,
    PLOT_VIOLIN,
    PLOT_BOX,
    PLOT_PCA_2D,
]

ADVANCED_PLOTS = [
    PLOT_SCATTER_3D,
    PLOT_LINE,
    PLOT_DENSITY_HEATMAP,
    PLOT_DENSITY_CONTOUR,
    PLOT_PARALLEL_CATEGORIES,
    PLOT_PARALLEL_COORDINATES,
    PLOT_SCATTER_MATRIX,
    PLOT_PCA_3D,
    PLOT_LDA_2D,
    PLOT_NCA,
    # PLOT_QDA_2D,
    PLOT_CORR_MATRIX,
    PLOT_PCA_SCATTER,
]

ALL_PLOTS = BASIC_PLOTS + ADVANCED_PLOTS

PLOT_HAS_X = [
    PLOT_SCATTER,
    PLOT_SCATTER_3D,
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
    PLOT_SCATTER_3D,
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
    PLOT_SCATTER_3D,
]
PLOT_HAS_COLOR = [
    PLOT_SCATTER,
    PLOT_SCATTER_3D,
    PLOT_PCA_2D,
    PLOT_PCA_SCATTER,
    PLOT_PCA_3D,
    PLOT_LDA_2D,
    PLOT_NCA,
    PLOT_QDA_2D,
    PLOT_LINE,
    PLOT_BAR,
    PLOT_HISTOGRAM,
    PLOT_BOX,
    PLOT_VIOLIN,
    PLOT_DENSITY_CONTOUR,
    PLOT_SCATTER_MATRIX,
]
PLOT_HAS_TEXT = [
    PLOT_SCATTER,
    PLOT_SCATTER_3D,
    PLOT_PCA_2D,
    PLOT_PCA_3D,
    PLOT_LDA_2D,
    PLOT_NCA,
    PLOT_QDA_2D,
    PLOT_LINE,
    PLOT_BAR,
]
PLOT_HAS_FACET = [
    PLOT_SCATTER,
    PLOT_PCA_2D,
    PLOT_LDA_2D,
    PLOT_NCA,
    PLOT_QDA_2D,
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
    PLOT_SCATTER_3D,
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
    PLOT_LDA_2D,
    PLOT_NCA,
    PLOT_QDA_2D,
    PLOT_DENSITY_HEATMAP,
    PLOT_DENSITY_CONTOUR,
]
PLOT_HAS_BAR_MODE = [
    PLOT_HISTOGRAM,
]
PLOT_HAS_SIZE = [
    PLOT_SCATTER,
    PLOT_SCATTER_3D,
    PLOT_PCA_2D,
    PLOT_PCA_3D,
    PLOT_LDA_2D,
    PLOT_NCA,
    PLOT_QDA_2D,
]

PLOT_HAS_SHAPE = [
    PLOT_SCATTER,
    PLOT_SCATTER_3D,
    PLOT_PCA_2D,
    PLOT_PCA_3D,
    PLOT_LDA_2D,
    PLOT_NCA,
    PLOT_QDA_2D,
]

PLOT_HAS_TREND_LINE = [
    PLOT_SCATTER,
    PLOT_DENSITY_CONTOUR,
]

PLOT_HAS_POINTS = [
    PLOT_BOX,
    PLOT_VIOLIN,
]

PLOT_HAS_ANIM = [
    PLOT_SCATTER,
    PLOT_SCATTER_3D,
    PLOT_BAR,
    PLOT_HISTOGRAM,
    PLOT_VIOLIN,
    PLOT_BOX,
    PLOT_DENSITY_HEATMAP,
    PLOT_DENSITY_CONTOUR,
    PLOT_PCA_2D,
    PLOT_PCA_3D,
    PLOT_LDA_2D,
    PLOT_NCA,
    PLOT_QDA_2D,
]

PLOT_HAS_CUSTOM_HOVER_DATA = [
    PLOT_SCATTER,
    PLOT_SCATTER_3D,
    PLOT_LINE,
    PLOT_BAR,
    PLOT_BOX,
    PLOT_VIOLIN,
    PLOT_PCA_2D,
    PLOT_PCA_3D,
    PLOT_LDA_2D,
    PLOT_NCA,
    PLOT_QDA_2D,
]

PLOT_IS_3D = [
    PLOT_SCATTER_3D,
    PLOT_PCA_3D,
]

PLOT_HAS_TARGET = [
    PLOT_LDA_2D,
    PLOT_NCA,
    PLOT_QDA_2D,
]

PLOT_HAS_LOADINGS = [
    PLOT_PCA_2D,
    PLOT_PCA_3D,
    PLOT_LDA_2D,
]

PLOT_HAS_IGNORE_COLUMNS = [
    PLOT_LDA_2D,
    PLOT_NCA,
    PLOT_QDA_2D,
    PLOT_PCA_2D,
    PLOT_PCA_SCATTER,
    PLOT_PCA_3D,
]

PLOT_HAS_PROGRESS_DISPLAY = [
    PLOT_SCATTER_MATRIX,
]

PLOT_HAS_SOLVER = [
    PLOT_LDA_2D,
]

PLOT_HAS_NCOMP = [
    PLOT_NCA,
]

PLOT_HAS_INIT = [
    PLOT_NCA,
]

PLOT_NEEDS_NA_DROP = [
    PLOT_PCA_2D,
    PLOT_PCA_SCATTER,
    PLOT_PCA_3D,
]

PLOT_HAS_MODEL_DATA = [
    PLOT_PCA_2D,
    PLOT_PCA_SCATTER,
    PLOT_PCA_3D,
    PLOT_LDA_2D,
    PLOT_QDA_2D,
    PLOT_NCA,
]

PLOT_HAS_COMPONENT_LIMIT = [
    PLOT_PCA_SCATTER,
]
