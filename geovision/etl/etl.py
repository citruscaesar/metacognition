from shutil import rmtree
from pathlib import Path

def reset_dir(dir_path: str | Path) -> None:
    dir_path = Path(dir_path)
    if dir_path.is_dir():
        rmtree(dir_path.as_posix())
    dir_path.mkdir()