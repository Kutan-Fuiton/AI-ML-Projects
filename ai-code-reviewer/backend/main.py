from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.orchestrator import run_analysis
from models.result_schema import AnalysisResult
from typing import List

app = FastAPI(title="AI Code Reviewer")


class AnalyzeRequest(BaseModel):
    repo_url: str


@app.get("/")
def health_check():
    return {"status": "running"}


@app.post("/analyze", response_model=List[AnalysisResult])
def analyze_repo(request: AnalyzeRequest):
    try:
        return run_analysis(request.repo_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






# from fastapi import FastAPI
# from services.orchestrator import run_analysis

# app = FastAPI()

# @app.get("/")
# def home():
#     return {"status": "AI Code Reviewer running"}

# @app.post("/analyze")
# def analyze_code(files: dict):
#     """
#     files = {
#       "file1.py": "code here",
#       "file2.py": "code here"
#     }
#     """
#     return run_analysis(files)




# from fastapi import FastAPI
# from pydantic import BaseModel
# from services.orchestrator import run_analysis

# app = FastAPI()

# # Input model for the analyze API
# class AnalyzeRequest(BaseModel):
#     repo_url: str

# @app.get("/health")
# def health_check():
#     return {"status": "ok"}

# @app.post("/analyze")
# async def analyze_repo(payload: AnalyzeRequest):
#     repo_url = payload.repo_url
#     result = await run_analysis(repo_url)
#     return result
