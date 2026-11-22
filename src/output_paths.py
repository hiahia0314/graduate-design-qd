"""Single destination for command-line generated artifacts."""
from pathlib import Path

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / 'output'
FACTOR_ROOT = OUTPUT_ROOT.parent / 'factor_library'


def output_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = OUTPUT_ROOT / (Path(*path.parts[1:]) if path.parts and path.parts[0] == 'output' else path)
    path = path.resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError('Output must be inside the project output/ directory')
    return path
