@echo off
setlocal
set "YTX_REPO=%~dp0.."
set "PATH=%YTX_REPO%\.venv\Scripts;%PATH%"
set "PYTHONUTF8=1"
if not exist "%YTX_REPO%\.venv\Scripts\python.exe" (
    echo transcribe: run bootstrap.ps1 in the repository first. 1>&2
    exit /b 1
)
"%YTX_REPO%\.venv\Scripts\python.exe" "%YTX_REPO%\bin\ytx" %*
exit /b %ERRORLEVEL%
