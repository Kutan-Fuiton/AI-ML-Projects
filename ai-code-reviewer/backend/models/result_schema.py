from pydantic import BaseModel
from typing import List


class FileIssue(BaseModel):
    file_path: str
    description: str


class AnalysisResult(BaseModel):
    bugs: List[FileIssue]
    security_issues: List[FileIssue]
    fix_suggestions: List[FileIssue]
    tests: List[FileIssue]
    summary: str
