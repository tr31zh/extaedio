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

AVAILABLE_URLS = [
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
]

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
PLOT_PCA_2D = "PCA (2D)"
PLOT_PCA_3D = "PCA (3D)"
PLOT_CORR_MATRIX = "Correlation matrix"

PLOT_HAS_X = [
    PLOT_SCATTER,
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
]
PLOT_HAS_COLOR = [
    PLOT_SCATTER,
    PLOT_PCA_2D,
    PLOT_PCA_3D,
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
    PLOT_PCA_2D,
    PLOT_PCA_3D,
    PLOT_LINE,
    PLOT_BAR,
    PLOT_PCA_2D,
    PLOT_PCA_3D,
]
PLOT_HAS_FACET = [
    PLOT_SCATTER,
    PLOT_PCA_2D,
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
    PLOT_DENSITY_HEATMAP,
    PLOT_DENSITY_CONTOUR,
]
PLOT_HAS_BAR_MODE = [
    PLOT_HISTOGRAM,
]
PLOT_HAS_SIZE = [
    PLOT_SCATTER,
    PLOT_PCA_2D,
    PLOT_PCA_3D,
]
PLOT_HAS_SHAPE = [
    PLOT_SCATTER,
    PLOT_PCA_2D,
    PLOT_PCA_3D,
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
    PLOT_BAR,
    PLOT_HISTOGRAM,
    PLOT_VIOLIN,
    PLOT_BOX,
    PLOT_DENSITY_HEATMAP,
    PLOT_DENSITY_CONTOUR,
    PLOT_PCA_2D,
]

AVAILABLE_PLOTS = [
    PLOT_SCATTER,
    PLOT_LINE,
    PLOT_BAR,
    PLOT_HISTOGRAM,
    PLOT_BOX,
    PLOT_VIOLIN,
    PLOT_DENSITY_HEATMAP,
    PLOT_DENSITY_CONTOUR,
    PLOT_PARALLEL_CATEGORIES,
    PLOT_PARALLEL_COORDINATES,
    PLOT_SCATTER_MATRIX,
    PLOT_PCA_2D,
    PLOT_PCA_3D,
    PLOT_CORR_MATRIX,
]

PLOTS_DICT = {}
for plot in AVAILABLE_PLOTS:
    PLOTS_DICT[plot] = {
        "name": plot,
        "allow_anim": plot in PLOT_HAS_ANIM,
        "has_x": plot in PLOT_HAS_X,
        "has_y": plot in PLOT_HAS_Y,
        "has_z": plot in PLOT_HAS_Z,
        "has_color": plot in PLOT_HAS_COLOR,
        "has_text": plot in PLOT_HAS_TEXT,
        "has_facet": plot in PLOT_HAS_FACET,
        "has_log": plot in PLOT_HAS_LOG,
        "has_bins": plot in PLOT_HAS_BINS,
        "has_marginals": plot in PLOT_HAS_MARGINAL,
        "has_marginals_xy": plot in PLOT_HAS_MARGINAL_XY,
        "has_bar_mode": plot in PLOT_HAS_BAR_MODE,
        "has_size": plot in PLOT_HAS_SIZE,
        "has_shape": plot in PLOT_HAS_SHAPE,
        "has_trend_line": plot in PLOT_HAS_TREND_LINE,
        "has_points": plot in PLOT_HAS_POINTS,
    }
