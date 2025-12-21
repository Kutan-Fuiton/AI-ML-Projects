from utils.repo_downloader import clone_repo, cleanup_repo
from services.github_reader import read_repository
from agents.bug_finder_agent import detect_bugs
from agents.security_agent import analyze_security
from agents.fix_suggestor_agent import suggest_fixes
from agents.test_writer_agent import write_tests
from agents.summary_agent import summarize_file


def run_analysis(repo_url: str):
    repo_path = clone_repo(repo_url)

    try:
        files = read_repository(repo_path, max_files=5)
        results = []

        for file in files:
            code = file["content"]
            filename = file["path"]

            file_result = {
                "file": filename,
                "bugs": detect_bugs(code, filename),
                "security": analyze_security(code, filename),
                "fixes": suggest_fixes(code, filename),
                "tests": write_tests(code, filename),
                "summary": summarize_file(code, filename)
            }

            results.append(file_result)

        return results

    finally:
        cleanup_repo(repo_path)

