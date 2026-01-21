def configure_cli_display() -> None:
    """
    Configure dataframe display defaults for interactive debugging/logging.

    Important: this is intentionally imported lazily from the CLI `run` command
    to avoid slowing down lightweight commands like `templates` and `init`.
    """
    # Keep tracebacks readable: do not dump gigantic locals tables.
    # Also cap dataframe display sizes for any explicit prints/logs.
    try:
        import polars as pl

        pl.Config.set_tbl_rows(10)
        pl.Config.set_tbl_cols(20)
        pl.Config.set_tbl_width_chars(160)
    except Exception:
        pass

    try:
        import pandas as pd

        pd.set_option("display.max_rows", 10)
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 160)
    except Exception:
        pass

