import typer
from pathlib import Path
from importlib.resources import files
from typing import Optional, Dict

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
):
    if ctx.invoked_subcommand is None:
        typer.echo(
            "\nQuick start:\n"
            "  proteoflux templates\n"
            "  proteoflux init spectronaut-proteomics --path config.yaml\n"
            "  proteoflux run --config config.yaml"
        )
        raise typer.Exit(code=0)

@app.command()
def init(
    template: str = typer.Argument(..., help="Template name (see 'proteoflux templates')"),
    path: Path = typer.Option(Path("proteoflux_config.yaml"), "--path", help="Output config path"),
):
    """
    Generate a config scaffold from a named template.
    """
    if template is None:
        typer.echo("Missing TEMPLATE.\n")
        typer.echo("Available templates:")
        for name in sorted(_TEMPLATES.keys()):
            typer.echo(f"  - {name}")
        typer.echo(
            "\nExample:\n"
            "  proteoflux init spectronaut-proteomics --path config.yaml\n"
            "\nThen run:\n"
            "  proteoflux run --config config.yaml"
        )
        raise typer.Exit(code=0)
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
    Run ProteoFlux pipeline. Use a full config YAML.
    """
    if config is None:
        raise typer.BadParameter(
            "Missing --config. Example: proteoflux run --config config.yaml"
        )

    import yaml
    from proteoflux.utils.cli_setup import configure_cli_display
    from proteoflux.main import run_pipeline

    configure_cli_display()
    config_data = yaml.safe_load(config.read_text())
    run_pipeline(config=config_data)

if __name__ == "__main__":
    app()

