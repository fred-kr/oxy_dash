import base64
import csv
import datetime
import io
from pathlib import Path

import dash_ag_grid as dag
import polars as pl
from dash import html


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
        with io.StringIO(decoded_string) as file:
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
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    suffix = Path(filename).suffix
    try:
        if "csv" in suffix or "txt" in suffix or "tsv" in suffix:
            content = decoded.decode("utf-8", errors="replace")
            if separator == "auto":
                separator = detect_delimiter(content, skip_rows=skip_rows)
            # In cases where each column contains whitespace characters (mainly PreSense OxyView) we need to remove these prior to reading the data with polars to correctly infer the schema
            cleaned_content = "\n".join(
                separator.join(field.strip() for field in line.split(separator)) for line in content.splitlines()
            )
            # # Split into separate lines
            # content_lines = content.splitlines()
            # # Split the lines at the separator character, remove leading and trailing whitespace from each part, and join the parts back together
            # stripped_lines: list[str] = []
            # for line in content_lines:
            #     stripped_line = separator.join(part.strip() for part in line.split(separator))
            #     stripped_lines.append(stripped_line)
            # cleaned_content = "\n".join(stripped_lines)
            df = pl.read_csv(io.StringIO(cleaned_content), skip_rows=skip_rows, separator=separator)
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
