import typer
from pathlib import Path
import yaml
from proteoflux.main import run_pipeline
from importlib.resources import files
from proteoflux import templates
from proteoflux.utils.compact_repr import compact_repr
import builtins
from typing import Optional, Dict

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

# Template registry: CLI name -> packaged YAML filename
_TEMPLATES: Dict[str, str] = {
    "spectronaut-proteomics": "spectronaut-proteomics.yaml",
    "spectronaut-peptidomics": "spectronaut-peptidomics.yaml",
    "spectronaut-phosphoproteomics": "spectronaut-phosphoproteomics.yaml",
    "fragpipe-peptidomics": "fragpipe-peptidomics.yaml",
}

def _write_template(template_name: str, path: Path) -> None:
    if template_name not in _TEMPLATES:
        available = ", ".join(sorted(_TEMPLATES.keys()))
        raise typer.BadParameter(
            f"Unknown template '{template_name}'. Available: {available}"
        )
    fname = _TEMPLATES[template_name]
    text = files("proteoflux.templates").joinpath(fname).read_text()
    path.write_text(text)
    typer.echo(f"Template '{template_name}' written to {path}")

@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    init: Optional[str] = typer.Option(
        None,
        "--init",
        help="Write a config template and exit. Example: proteoflux --init spectronaut-proteomics",
    ),
    path: Path = typer.Option(
        Path("proteoflux_config.yaml"),
        "--path",
        help="Output path for --init (default: proteoflux_config.yaml)",
    ),
):
    if init is not None:
        _write_template(init, path)
        raise typer.Exit(code=0)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)

@app.command()
def init(
    template: str = typer.Argument(..., help="Template name (see 'proteoflux templates')"),
    path: Path = typer.Option(Path("proteoflux_config.yaml"), "--path", help="Output config path"),
):
    """
    Generate a config scaffold from a named template.
    """
    _write_template(template, path)

@app.command("templates")
def list_templates():
    """
    List available config templates.
    """
    for name in sorted(_TEMPLATES.keys()):
        typer.echo(name)

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

