from shutil import rmtree
from pathlib import Path

def reset_dir(dir_path: str | Path) -> None:
    dir_path = Path(dir_path)
    if dir_path.is_dir():
        rmtree(dir_path.as_posix())
    dir_path.mkdir()

def validate_dir(dir_path: str | Path) -> Path:
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        dir_path.mkdir(parents = True)
    return dir_path

def is_valid_remote(remote_url: str) -> bool:
    assert isinstance(remote_url, str), "URL must be of string type"

    # TODO : Find a way to validate remote URLS, boto3 for s3?

    if "//" in remote_url:
        return True
    return False

def is_valid_path(local_path: str | Path) -> bool:
    local_path = Path(local_path)
    if local_path.exists():
        return True
    return False

def get_local_path_from_remote(remote_url: str) -> Path:
    assert isinstance(remote_url, str), "URL must be of string type"
    return Path.home() / ('/'.join(Path(remote_url.split("//")[-1]).parts[1:]))