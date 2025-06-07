"""CLI для проекта Credit Risk MLOps."""

import typer

app = typer.Typer(help="Credit Risk MLOps CLI.")


@app.command()
def hello() -> None:
    """Выводит приветствие."""
    typer.echo("Привет, Credit Risk MLOps!")


if __name__ == "__main__":
    app()
