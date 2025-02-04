import base64
import csv
import datetime
import io
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal, NamedTuple, TypedDict

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, State, callback, dcc, html
from scipy import stats

type PlotlyTemplate = Literal[
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
]


class PlotlyTheme(StrEnum):
    GGPLOT2 = "ggplot2"
    SEABORN = "seaborn"
    SIMPLE_WHITE = "simple_white"
    PLOTLY = "plotly"
    PLOTLY_WHITE = "plotly_white"
    PLOTLY_DARK = "plotly_dark"
    PRESENTATION = "presentation"
    XGRIDOFF = "xgridoff"
    YGRIDOFF = "ygridoff"
    GRIDON = "gridon"
    NONE = "none"


class QualitativeColors(NamedTuple):
    """NamedTuple holding the available qualitative color palettes."""

    Plotly = (
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    )
    D3 = "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"
    G10 = "#3366CC", "#DC3912", "#FF9900", "#109618", "#990099", "#0099C6", "#DD4477", "#66AA00", "#B82E2E", "#316395"
    T10 = "#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"
    Alphabet = (
        "#AA0DFE",
        "#3283FE",
        "#85660D",
        "#782AB6",
        "#565656",
        "#1C8356",
        "#16FF32",
        "#F7E1A0",
        "#E2E2E2",
        "#1CBE4F",
        "#C4451C",
        "#DEA0FD",
        "#FE00FA",
        "#325A9B",
        "#FEAF16",
        "#F8A19F",
        "#90AD1C",
        "#F6222E",
        "#1CFFCE",
        "#2ED9FF",
        "#B10DA1",
        "#C075A6",
        "#FC1CBF",
        "#B00068",
        "#FBE426",
        "#FA0087",
    )
    Dark24 = (
        "#2E91E5",
        "#E15F99",
        "#1CA71C",
        "#FB0D0D",
        "#DA16FF",
        "#222A2A",
        "#B68100",
        "#750D86",
        "#EB663B",
        "#511CFB",
        "#00A08B",
        "#FB00D1",
        "#FC0080",
        "#B2828D",
        "#6C7C32",
        "#778AAE",
        "#862A16",
        "#A777F1",
        "#620042",
        "#1616A7",
        "#DA60CA",
        "#6C4516",
        "#0D2A63",
        "#AF0038",
    )
    Light24 = (
        "#FD3216",
        "#00FE35",
        "#6A76FC",
        "#FED4C4",
        "#FE00CE",
        "#0DF9FF",
        "#F6F926",
        "#FF9616",
        "#479B55",
        "#EEA6FB",
        "#DC587D",
        "#D626FF",
        "#6E899C",
        "#00B5F7",
        "#B68E00",
        "#C9FBE5",
        "#FF0092",
        "#22FFA7",
        "#E3EE9E",
        "#86CE00",
        "#BC7196",
        "#7E7DCD",
        "#FC6955",
        "#E48F72",
    )


class LinearFitInfoDict(TypedDict):
    name: str
    respiration_period: int
    fit_id: int
    temperature_mean: float
    temperature_std_dev: float
    change_per_minute: float
    slope: float
    slope_std_err: float
    r_squared: float
    r_value: float
    p_value: float
    intercept: float
    intercept_std_err: float


class LinregressResultDict(TypedDict):
    slope: float
    intercept: float
    rvalue: float
    pvalue: float
    stderr: float
    intercept_stderr: float
    rsquared: float


class LinregressResult(NamedTuple):
    slope: float
    intercept: float
    rvalue: float
    pvalue: float
    stderr: float
    intercept_stderr: float

    def to_dict(self) -> LinregressResultDict:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "rvalue": self.rvalue,
            "pvalue": self.pvalue,
            "stderr": self.stderr,
            "intercept_stderr": self.intercept_stderr,
            "rsquared": self.rvalue**2,
        }

    def to_df(self) -> pl.DataFrame:
        return pl.DataFrame([self.to_dict()])


class SelectedPoint(TypedDict):
    curveNumber: int
    pointNumber: int
    pointIndex: int
    x: float
    y: float
    customdata: list[Any]


class SelectedRange(TypedDict):
    x: list[float]
    y: list[float]


class SelectedData(TypedDict):
    points: list[SelectedPoint]
    range: SelectedRange


class UploadedData(TypedDict):
    name: str
    data: str


class ResultRow(TypedDict):
    source_file: str
    start_index: int
    end_index: int
    slope: float
    rsquared: float


@dataclass(slots=True)
class LayoutOpts:
    """
    Dataclass to hold layout options for the Plotly plot.

    Parameters
    ----------
    title : str
        The title of the plot.
    xaxis_title : str
        The title of the x-axis.
    yaxis_title : str
        The title of the y-axis.
    theme : PlotlyTemplate, optional
        The theme to use for the plot. Defaults to "simple_white".
    colors : Sequence[str], optional
        The colors to use for the fits. Defaults to QualitativeColors.Plotly.
    width : int, optional
        The width of the plot. Defaults to 2100.
    height : int, optional
        The height of the plot. Defaults to 1000.
    font_size : int, optional
        The font size of the annotations in pixels. Defaults to 12.
    """

    title: str = "Oxygen Saturation vs Time"
    xaxis_title: str = "Time (s)"
    yaxis_title: str = "O2 Saturation (%)"
    theme: PlotlyTemplate = "simple_white"
    colors: Sequence[str] = QualitativeColors.Plotly
    width: int = 2100
    height: int = 1000
    font_size: int = 12

    def set_colors(self, name: Literal["Plotly", "D3", "G10", "T10", "Alphabet", "Dark24", "Light24"]) -> None:
        self.colors = getattr(QualitativeColors, name)


class DataSegmentDict(TypedDict):
    segment_id: str
    start_index: int
    end_index: int
    data: str  # JSON string of df, read with pl.read_json(io.StringIO(data))
    fit_result: LinregressResultDict
    fig: go.Figure
    x_col: str
    y0_col: str
    y1_col: str | None
    name: str
    formatted_results: str


class DataSegment:
    all_segments: ClassVar[list[DataSegmentDict]] = []
    source_name: ClassVar[str] = ""
    source_data: ClassVar[pl.DataFrame] = pl.DataFrame()
    source_fig: ClassVar[go.Figure] = go.Figure()
    x_col: ClassVar[str] = ""
    y0_col: ClassVar[str] = ""
    y1_col: ClassVar[str | None] = None
    _source_set: ClassVar[bool] = False
    _layout_opts: ClassVar[LayoutOpts] = LayoutOpts()

    @classmethod
    def set_source(
        cls,
        source_name: str,
        source_data: pl.DataFrame,
        x_col: str,
        y0_col: str,
        y1_col: str | None = None,
        layout_opts: LayoutOpts | None = None,
    ) -> None:
        layout_opts = layout_opts or LayoutOpts()
        cls._layout_opts = layout_opts
        cls.source_name = Path(source_name).stem
        cls.source_data = source_data
        cls.x_col = x_col
        cls.y0_col = y0_col
        cls.y1_col = y1_col
        cls.make_base_fig()
        cls._source_set = True

    @classmethod
    def make_base_fig(
        cls,
    ) -> None:
        cls.all_segments = []
        point_color = "lightgray" if cls.y1_col is None else cls.source_data.get_column(cls.y1_col)
        fig = go.Figure()
        fig.add_scattergl(
            x=cls.source_data.get_column(cls.x_col),
            y=cls.source_data.get_column(cls.y0_col),
            mode="markers",
            marker=dict(color=point_color, symbol="circle-open-dot", colorscale="Plasma", opacity=0.2, size=3),
        )
        fig.update_layout(
            clickmode="event+select",
            template=cls._layout_opts.theme,
            height=cls._layout_opts.height,
            dragmode="select",
            showlegend=False,
        )
        cls.source_fig = fig

    def __init__(self, start_index: int, end_index: int) -> None:
        if not self._source_set:
            raise ValueError("DataSegment must be initialized after calling DataSegment.set_source")
        self.start_index = start_index
        self.end_index = end_index
        self.data = self.source_data.slice(self.start_index, self.end_index - self.start_index + 1)
        res: Any = stats.linregress(self.data.get_column(self.x_col), self.data.get_column(self.y0_col))
        self.fit_result = LinregressResult(
            slope=res.slope,
            intercept=res.intercept,
            rvalue=res.rvalue,
            pvalue=res.pvalue,
            stderr=res.stderr,
            intercept_stderr=res.intercept_stderr,
        )
        self.data = self.data.with_columns(
            pl.lit(self.segment_id).alias("segment_id"),
            (self.fit_result.slope * pl.col(self.x_col) + self.fit_result.intercept).alias("fitted"),
        )
        DataSegment.all_segments.append(self.serialize())
        DataSegment.all_segments.sort(key=lambda s: s["start_index"])

    @property
    def segment_id(self) -> str:
        return f"{self.start_index}-{self.end_index}"

    @property
    def name(self) -> str:
        return f"{self.source_name}_{self.segment_id}"

    @property
    def x_data(self) -> pl.Series:
        return self.data.get_column(self.x_col)

    @property
    def y0_data(self) -> pl.Series:
        return self.data.get_column(self.y0_col)

    @property
    def y1_data(self) -> pl.Series | None:
        return None if self.y1_col is None else self.data.get_column(self.y1_col)

    @property
    def y0_fitted(self) -> pl.Series:
        return self.data.get_column("fitted")

    @property
    def slope(self) -> float:
        return self.fit_result.slope

    @property
    def intercept(self) -> float:
        return self.fit_result.intercept

    @property
    def r_value(self) -> float:
        return self.fit_result.rvalue

    @property
    def p_value(self) -> float:
        return self.fit_result.pvalue

    @property
    def stderr(self) -> float:
        return self.fit_result.stderr

    @property
    def intercept_stderr(self) -> float:
        return self.fit_result.intercept_stderr

    @property
    def r_squared(self) -> float:
        return self.r_value**2

    def plot(self, add: bool = True) -> go.Figure:
        if add:
            self.source_fig.add_scattergl(
                x=self.x_data,
                y=self.y0_fitted,
                mode="lines",
                line=dict(color="red", width=3),
                name=f"Segment {self.segment_id}",
                hoverinfo="name",
            )
            return self.source_fig
        else:
            fig = go.Figure()
            point_color = self.y1_data or "lightgray"
            fig.add_scattergl(
                x=self.x_data,
                y=self.y0_data,
                mode="markers",
                marker=dict(color=point_color),
                name=f"Segment {self.segment_id}, raw values",
                hoverinfo="name",
            )
            fig.add_scattergl(
                x=self.x_data,
                y=self.y0_fitted,
                mode="lines",
                line=dict(color="red", width=3),
                name=f"Segment {self.segment_id}, fitted '{self.y0_col}' values",
                hoverinfo="name",
            )
            return fig

    def serialize(self) -> DataSegmentDict:
        return {
            "segment_id": self.segment_id,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "data": self.data.write_json(),
            "fit_result": self.fit_result.to_dict(),
            "fig": self.plot(),
            "x_col": self.x_col,
            "y0_col": self.y0_col,
            "y1_col": self.y1_col,
            "name": self.name,
            "formatted_results": self.fit_result.to_df().write_json(),
        }


app = Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR])

server = app.server
app.config.suppress_callback_exceptions = True


# -----------------------------------------------------------------------------
# Style dictionaries for reuse
# -----------------------------------------------------------------------------
upload_style = {
    "width": "100%",
    "height": "60px",
    "lineHeight": "60px",
    "borderWidth": "1px",
    "borderStyle": "dashed",
    "borderRadius": "5px",
    "textAlign": "center",
    "margin": "10px",
}

upload_link_style = {
    "color": "blue",
    "textDecoration": "underline",
    "cursor": "pointer",
}

flex_container_style = {
    "display": "flex",
    "gap": "10px",
    "align-items": "center",
    "margin-top": "10px",
}

container_style = {"padding": "10px"}

# -----------------------------------------------------------------------------
# Column definitions for the segment results grid
# -----------------------------------------------------------------------------
segment_grid_columns: list[dict[str, Any]] = [
    {"field": "source_file", "headerName": "source_file", "checkboxSelection": True},
    {"field": "start_index", "headerName": "start_index"},
    {"field": "end_index", "headerName": "end_index"},
    {"field": "slope", "headerName": "slope"},
    {"field": "rsquared", "headerName": "rsquared"},
]

# -----------------------------------------------------------------------------
# App layout
# -----------------------------------------------------------------------------
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                # Left column: Card with file upload, dropdowns, and buttons
                dbc.Col(
                    dbc.Card(
                        [
                            # Inputs for setting skip_rows and separator for the data upload
                            html.Div(
                                [
                                    dbc.Label("Skip Rows"),
                                    dbc.Input(id="skip-rows", type="number", value=0, min=0, style={"flex": "1"}),
                                    dbc.Label("Column Separator"),
                                    dcc.Dropdown(
                                        id="separator",
                                        options=[
                                            {"label": "Detect Automatically", "value": "auto"},
                                            {"label": "Comma (,)", "value": ","},
                                            {"label": "Semicolon (;)", "value": ";"},
                                            {"label": "Tab (\\t)", "value": "\t"},
                                            {"label": "Pipe (|)", "value": "|"},
                                        ],
                                        value="auto",
                                        style={"flex": "1"},
                                    ),
                                ],
                                style=container_style,
                            ),
                            # File upload section
                            html.Div(
                                [
                                    dcc.Upload(
                                        id="upload-data",
                                        children=html.Div(
                                            [
                                                "Drag and Drop or ",
                                                html.A("Select File", style=upload_link_style),
                                            ]
                                        ),
                                        multiple=False,
                                        style=upload_style,
                                    ),
                                    dbc.Label("Current File: -", id="current-file-label"),
                                ]
                            ),
                            # Dropdowns for selecting x and y columns
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id="x-data",
                                        placeholder="Select column for x-axis",
                                        style={"flex": "1"},
                                    ),
                                    dcc.Dropdown(
                                        id="y-data",
                                        multi=True,
                                        placeholder="Select column(s) for y-axis",
                                        style={"flex": "1"},
                                    ),
                                ],
                                style=flex_container_style,
                            ),
                            # Dropdown for plot template and control buttons
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id="plot-template",
                                        options=[t.value for t in PlotlyTheme],
                                        value="simple_white",
                                        style={"flex": "1"},
                                    ),
                                    dbc.Button("Plot", id="plot-button", n_clicks=0),
                                    dbc.Button("Add Segment", id="add-segment-button", n_clicks=0),
                                    dbc.Button("Clear Segments", id="clear-segments-button", n_clicks=0),
                                    dbc.Button("Save Segments", id="save-segments-button", n_clicks=0),
                                ],
                                style=flex_container_style,
                            ),
                        ],
                        body=True,
                    ),
                    width=4,
                ),
                # Right column: Data grid for segment results
                dbc.Col(
                    html.Div(
                        dag.AgGrid(
                            id="segment-result-grid",
                            columnSize="responsiveSizeToFit",
                            columnDefs=segment_grid_columns,
                            rowData=[],
                            csvExportParams={"fileName": "results.csv"},
                            dashGridOptions={
                                "rowSelection": "multiple",
                                "suppressRowClickSelection": True,
                                "animateRows": False,
                            },
                        ),
                        id="segment-table",
                    ),
                    width=8,
                ),
            ]
        ),
        # Row for the graph output
        dbc.Row(
            dbc.Col(dcc.Graph(id="output-graph"), width=12),
            style={"margin-top": "10px"},
        ),
        # Row for data upload output (if any)
        dbc.Row(
            dbc.Col(id="output-data-upload", width=12),
            style={"margin-top": "10px"},
        ),
        # Hidden stores for intermediate data
        dcc.Store(id="uploaded-data"),
        dcc.Store(id="data-segments"),
    ],
    fluid=True,
    style=container_style,
)


def detect_delimiter(decoded_string: str, max_rows: int = 3, skip_rows: int = 0) -> str:
    """
    Automatically detects the delimiter used in a text file containing tabular data.

    Args:
        decoded_string (str): The content of the file as a decoded string.
        max_rows (int): The number of rows to read from the file for analysis.
        skip_rows (int): The number of rows to skip before starting the analysis.

    Returns:
        str: The detected delimiter character.

    Raises:
        ValueError: If the delimiter cannot be detected.
        FileNotFoundError: If the file does not exist.
        ValueError: If max_rows is not a positive integer.
    """
    if max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")

    try:
        with open(io.StringIO(decoded_string).name, "r") as file:
            # Check if the file is empty
            if file.readline() == "":
                raise ValueError("File is empty")

            # Skip the specified number of rows
            for _ in range(skip_rows):
                file.readline()

            # Read the first max_rows rows and join them into a single string
            sample = "\n".join(file.readline() for _ in range(max_rows - 1))
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}") from e

    # Create a Sniffer instance
    sniffer = csv.Sniffer()

    try:
        # Attempt to detect the dialect from the sample text
        dialect = sniffer.sniff(sample)
        return dialect.delimiter
    except csv.Error as e:
        # Raise an exception if the delimiter cannot be detected
        raise ValueError(f"Delimiter detection failed. {str(e)}") from e


def parse_contents(
    contents: str, filename: str, date: float, skip_rows: int = 0, separator: str = "auto"
) -> tuple[html.Div, pl.DataFrame]:
    content_type, content_string = contents.split(",")
    print(content_type)

    decoded = base64.b64decode(content_string)
    suffix = Path(filename).suffix
    try:
        if "csv" in suffix or "txt" in suffix or "tsv" in suffix:
            content = decoded.decode("ansi")
            if separator == "auto":
                separator = detect_delimiter(content, skip_rows=skip_rows)

            df = pl.read_csv(content, skip_rows=skip_rows, separator=separator)
        elif "xls" in suffix:
            # Assume that the user uploaded an excel file
            df = pl.read_excel(io.BytesIO(decoded))
        else:
            return html.Div(["There was an error processing this file."]), pl.DataFrame()
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."]), pl.DataFrame()

    return html.Div(
        [
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(date)),
            dag.AgGrid(
                columnSize="sizeToFit",
                columnDefs=[{"field": col_name} for col_name in df.columns],
                rowData=df.to_dicts(),
            ),
        ]
    ), df


@callback(
    Output("output-data-upload", "children"),
    Output("uploaded-data", "data"),
    Output("x-data", "options"),
    Output("y-data", "options"),
    Output("current-file-label", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("upload-data", "last_modified"),
    State("skip-rows", "value"),
    State("separator", "value"),
)
def update_output(
    content: str | None, name: str, date: float, skip_rows: int, separator: str
) -> tuple[list[html.Div], UploadedData, list[str], list[str], str]:
    if content is not None:
        div, df = parse_contents(content, name, date, skip_rows, separator)
        return [div], {"name": name, "data": df.write_json()}, df.columns, df.columns, f"Current File: {name}"

    return [], {"name": "", "data": ""}, [], [], "Current File: -"


@callback(
    Output("output-graph", "figure", allow_duplicate=True),
    Input("plot-button", "n_clicks"),
    State("plot-template", "value"),
    State("uploaded-data", "data"),
    State("x-data", "value"),
    State("y-data", "value"),
    prevent_initial_call=True,
)
def update_graph(n_clicks: int, template: PlotlyTemplate, data: UploadedData, x: str, y: list[str]) -> go.Figure:
    if not n_clicks or not data:
        return go.Figure()
    df = pl.read_json(io.StringIO(data["data"]))
    lopts = LayoutOpts(theme=template, width=1600, height=1000)
    DataSegment.set_source(data["name"], df, x, y[0], y[1] if len(y) > 1 else None, lopts)
    return DataSegment.source_fig


@callback(
    Output("data-segments", "data", allow_duplicate=True),
    Output("output-graph", "figure", allow_duplicate=True),
    Output("segment-result-grid", "rowData"),
    Input("add-segment-button", "n_clicks"),
    State("output-graph", "selectedData"),
    State("data-segments", "data"),
    prevent_initial_call=True,
)
def update_segments(
    n_clicks: int, selected_data: SelectedData | None, segments: list[DataSegmentDict]
) -> tuple[list[DataSegmentDict], go.Figure, list[dict[str, Any]]]:
    if not n_clicks or not selected_data:
        return [], DataSegment.source_fig, []
    start = selected_data["points"][0]["pointIndex"]
    end = selected_data["points"][-1]["pointIndex"]
    DataSegment(start, end)

    res_dfs = pl.concat(
        [
            pl.read_json(io.StringIO(s["formatted_results"])).select(
                pl.lit(DataSegment.source_name).alias("source_file"),
                pl.lit(s["start_index"]).alias("start_index"),
                pl.lit(s["end_index"]).alias("end_index"),
                pl.col("slope").alias("slope"),
                pl.col("rsquared").alias("rsquared"),
            )
            for s in DataSegment.all_segments
        ]
    )
    return DataSegment.all_segments, DataSegment.source_fig, res_dfs.to_dicts()


@callback(
    Output("output-graph", "figure", allow_duplicate=True),
    Output("data-segments", "data", allow_duplicate=True),
    Output("segment-result-grid", "deleteSelectedRows"),
    Input("clear-segments-button", "n_clicks"),
    State("segment-result-grid", "selectedRows"),
    prevent_initial_call=True,
)
def clear_segments(n_clicks: int, selected_rows: list[ResultRow]) -> tuple[go.Figure, list[DataSegmentDict], bool]:
    current_fits = DataSegment.all_segments.copy()
    # get the start indices from the selected rows and use them to remove the corresponding segments
    for row in selected_rows:
        start = row["start_index"]
        for i, fit in enumerate(current_fits):
            if fit["start_index"] == start:
                current_fits.pop(i)
                break

    DataSegment.make_base_fig()
    for cfit in current_fits:
        DataSegment(cfit["start_index"], cfit["end_index"])

    return DataSegment.source_fig, DataSegment.all_segments, True


@callback(
    Output("segment-result-grid", "exportDataAsCsv"),
    Input("save-segments-button", "n_clicks"),
    prevent_initial_call=True,
)
def save_segments_to_csv(n_clicks: int) -> bool:
    return bool(n_clicks)


if __name__ == "__main__":
    app.run(debug=True)
