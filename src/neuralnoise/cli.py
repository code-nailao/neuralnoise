import logging
import shutil
from pathlib import Path

# --- LLM Call Logger Name Setup ---
EVENT_LOGGER_NAME = None
FOUND_LOGGER_NAME_VIA = "default"

try:
    from autogen.runtime_logging import EVENT_LOGGER_NAME
    FOUND_LOGGER_NAME_VIA = "autogen.runtime_logging"
except ImportError:
    try:
        from autogen.events.logging import EVENT_LOGGER_NAME
        FOUND_LOGGER_NAME_VIA = "autogen.events.logging"
    except ImportError:
        try:
            from autogen.logging import EVENT_LOGGER_NAME
            FOUND_LOGGER_NAME_VIA = "autogen.logging"
        except ImportError:
            try:
                from autogen_core.logging import EVENT_LOGGER_NAME
                FOUND_LOGGER_NAME_VIA = "autogen_core.logging"
            except ImportError:
                EVENT_LOGGER_NAME = "autogen.events" # Default fallback
                print(f"Warning: Could not import EVENT_LOGGER_NAME from known paths. Defaulting to '{EVENT_LOGGER_NAME}'. LLM call logging might not work as expected.", flush=True)

if EVENT_LOGGER_NAME:
     print(f"INFO: Using EVENT_LOGGER_NAME='{EVENT_LOGGER_NAME}' found via '{FOUND_LOGGER_NAME_VIA}' for LLM call logging.", flush=True)

import json

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
# --- End Logger Name Setup ---

import typer
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from tabulate import tabulate

from neuralnoise.extract import extract_content
from neuralnoise.studio import generate_podcast_episode
from neuralnoise.utils import package_root

app = typer.Typer()

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@app.command()
def generate(
    name: str = typer.Option(..., help="Name of the podcast episode"),
    input: list[str] | None = typer.Argument(
        None,
        help="Paths to input files or URLs. Can specify multiple inputs.",
    ),
    config: Path = typer.Option(
        Path("config/config_openai.json"),
        help="Path to the podcast configuration file",
    ),
    only_script: bool = typer.Option(False, help="Only generate the script and exit"),
):
    """
    Generate a script from one or more input text files using the specified configuration.

    For example:

    nn generate <url|file> [<url|file>...] --name <name> --config config/config_openai.json
    """
    # --- LLM Call Logging Setup for this specific run ---
    log_file_path = LOG_DIR / f"{name}_llm_calls.log"
    
    # Define formatter once
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    # Create the file handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    loggers_to_configure = []
    if EVENT_LOGGER_NAME:
        loggers_to_configure.append(EVENT_LOGGER_NAME)
    else:
        # If import failed, EVENT_LOGGER_NAME is our fallback string
        loggers_to_configure.append("autogen.events") 

    # --- Diagnostic Step: Also add handler to root 'autogen' logger ---
    # This might capture events if they are logged elsewhere in the library
    # and propagate. Avoid adding the handler twice if EVENT_LOGGER_NAME *is* 'autogen'.
    if "autogen" not in loggers_to_configure:
        loggers_to_configure.append("autogen")
    # --- End Diagnostic Step ---

    for logger_name in loggers_to_configure:
        logger_instance = logging.getLogger(logger_name)
        # Set level to DEBUG to capture LLMCallEvent / other details
        # Setting level on the logger itself is important
        logger_instance.setLevel(logging.DEBUG) 

        # Check if this specific logger already has an identical handler
        handler_exists_for_logger = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file_path)
            for h in logger_instance.handlers
        )

        if not handler_exists_for_logger:
            print(f"INFO: Attaching file handler to logger '{logger_name}' for path: {log_file_path}", flush=True)
            logger_instance.addHandler(file_handler)
            # Optional: Control propagation if needed, usually default (True) is fine unless duplicating logs
            # logger_instance.propagate = False 
        else:
             print(f"INFO: File handler for {log_file_path} already exists on logger '{logger_name}'. Skipping add.", flush=True)

    # --- End LLM Call Logging Setup ---

    typer.secho(f"Generating podcast episode {name}", fg=typer.colors.GREEN)

    output_dir = Path("output") / name
    output_dir.mkdir(parents=True, exist_ok=True)

    content_path = output_dir / "content.md"

    if content_path.exists():
        with open(content_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        if input is None:
            typer.secho(
                "No input provided. Please specify input files or URLs.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

        typer.secho(f"Extracting content from inputs {input}", fg=typer.colors.YELLOW)
        content = extract_content(input)

        with open(content_path, "w", encoding="utf-8") as f:
            f.write(content)

    typer.secho(f"Generating podcast episode {name}", fg=typer.colors.GREEN)
    generate_podcast_episode(
        name,
        content,
        config_path=config,
        only_script=only_script,
    )

    typer.secho(
        f"Podcast generation complete. Output saved to {output_dir}",
        fg=typer.colors.GREEN,
    )


def get_audio_length(file_path: Path) -> float:
    """Get the length of an audio file in seconds."""
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000  # Convert milliseconds to seconds
    except CouldntDecodeError:
        typer.echo(f"Error: Couldn't decode audio file {file_path}")
        return -1.0


@app.command("list")
def list_episodes():
    """
    List all generated podcast episodes stored in the 'output' folder,
    including their audio file length in minutes. Episodes with invalid audio files are filtered out.
    """
    output_dir = Path("output")
    if not output_dir.exists():
        typer.echo("No episodes found. The 'output' folder does not exist.")
        return

    episodes = [d for d in output_dir.iterdir() if d.is_dir()]

    if not episodes:
        typer.echo("No episodes found in the 'output' folder.")
        return

    episode_data = []
    for episode in sorted(episodes):
        audio_files = list(episode.glob("*.wav")) + list(episode.glob("*.mp3"))
        if audio_files:
            audio_file = audio_files[0]  # Take the first audio file found
            length_seconds = get_audio_length(audio_file)
            if length_seconds != -1:  # Filter out invalid audio files
                length_minutes = length_seconds / 60  # Convert seconds to minutes
                episode_data.append(
                    [episode.name, audio_file.name, f"{length_minutes:.2f}"]
                )
        else:
            episode_data.append([episode.name, "No audio file", "N/A"])

    if not episode_data:
        typer.echo("No valid episodes found.")
        return

    headers = ["Episode", "Audio File", "Length (minutes)"]
    table = tabulate(episode_data, headers=headers, tablefmt="grid")
    typer.echo("Generated podcast episodes:")
    typer.echo(table)


@app.command()
def init(
    output_path: Path = typer.Option(
        Path("prompts"),
        "--output",
        "-o",
        help="Directory where prompts will be copied to",
        show_default=True,
    ),
):
    """
    Initialize a local copy of the default prompts.
    Creates a directory with the default prompt templates.

    Example:
        nn init
        nn init --output custom/path/prompts
    """
    source_dir = package_root / "prompts"

    if output_path.exists():
        typer.echo(f"Directory {output_path} already exists. Skipping initialization.")
        return

    try:
        shutil.copytree(source_dir, output_path)
        typer.echo(f"Successfully created prompts directory at {output_path}")
        typer.echo("You can now customize these prompts for your podcast generation.")
    except Exception as e:
        typer.echo(f"Error creating prompts directory: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
