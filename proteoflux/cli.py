import typer
from pathlib import Path
import yaml
from proteoflux.main import run_pipeline
from importlib.resources import files
from proteoflux import templates
from proteoflux.utils.compact_repr import compact_repr
import builtins

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

app = typer.Typer(help="ProteoFlux: Reproducible proteomics workflows")

@app.command()
def init(path: Path = Path("proteoflux_config.yaml")):
    """
    Generate a config scaffold (basic template) at given path.
    """
    default_yaml = files("proteoflux.templates").joinpath("user_template.yaml").read_text()

    path.write_text(default_yaml)
    typer.echo(f"Template written to {path}")

@app.command()
def run(
    config: Path = typer.Option(None, help="Path to YAML config file"),
):
    """
    Run ProteoFlux pipeline. Either use a full config YAML or specify core arguments manually.
    """
    config_data = yaml.safe_load(config.read_text())

    run_pipeline(config=config_data)

if __name__ == "__main__":
    app()

