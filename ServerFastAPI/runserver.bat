@echo off
cls
uvicorn ServerFastAPI.app:app --reload --port 11434

