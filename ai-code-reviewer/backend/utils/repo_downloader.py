import os
import shutil
import subprocess
import uuid
import time

# Base directory where repos will be cloned
BASE_REPO_DIR = "tmp_repos"


def clone_repo(repo_url: str) -> str:
    """
    Clone a GitHub repository into a temporary folder
    and return the local path.
    """

    if not os.path.exists(BASE_REPO_DIR):
        os.makedirs(BASE_REPO_DIR)

    repo_id = str(uuid.uuid4())
    repo_path = os.path.join(BASE_REPO_DIR, repo_id)

    # Clone repo
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, repo_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # IMPORTANT (Windows-safe):
    # Remove .git folder immediately to avoid file locks
    git_dir = os.path.join(repo_path, ".git")
    if os.path.exists(git_dir):
        shutil.rmtree(git_dir, ignore_errors=True)

    return repo_path


def cleanup_repo(repo_path: str):
    """
    Safely delete the cloned repository.
    On Windows, file locks can occur, so we fail silently.
    """

    # Small delay to allow OS to release handles
    time.sleep(0.5)

    try:
        shutil.rmtree(repo_path, ignore_errors=True)
    except Exception as e:
        print(f"[WARN] Failed to cleanup repo: {e}")
