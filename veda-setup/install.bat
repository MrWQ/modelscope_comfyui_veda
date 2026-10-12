@echo off
rem Veda-on-ComfyUI one-click setup entry (CRLF line endings required, ASCII only)
rem Only locates a Python interpreter then calls setup_veda.py; no business logic here.
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY (
  echo Python not found. Install Python 3.10-3.12: winget install Python.Python.3.12
  exit /b 1
)

"%PY%" "%SCRIPT_DIR%setup_veda.py" %*
endlocal
