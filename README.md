# SEC Filing Analyst

A Python RAG system and tool-using agent over SEC EDGAR 10-K filings (Apple, Microsoft,
NVIDIA, Tesla, Amazon). It answers plain-English questions and cites the exact filing
passages it used.

> Work in progress. Sections below are filled in as milestones complete.

## Quick start (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install --no-deps -e .
copy .env.example .env   # then fill in GEMINI_API_KEY and EDGAR_USER_AGENT
.venv\Scripts\python -m pytest -q
```
