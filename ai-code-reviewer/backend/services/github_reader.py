import os

ALLOWED_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".html", ".css"}
IGNORE_DIRS = {".git", "node_modules", "venv", "__pycache__"}


def read_repository(repo_path: str, max_files: int = 5):
    """
    Reads at most `max_files` source code files
    """
    collected_files = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            ext = os.path.splitext(file)[1]
            if ext not in ALLOWED_EXTENSIONS:
                continue

            full_path = os.path.join(root, file)

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            collected_files.append({
                "path": full_path,
                "content": content
            })

            if len(collected_files) >= max_files:
                return collected_files

    return collected_files
