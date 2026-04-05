@echo off
title ICU Analytics System Launcher
color 0A

echo.
echo  ========================================
echo    ICU Analytics System - Launcher
echo  ========================================
echo.

:MENU
echo  [1] Setup (first time - loads data + trains model)
echo  [2] Start Backend API
echo  [3] Start Frontend Dashboard
echo  [4] Generate Demo Data (no eICU needed)
echo  [5] Exit
echo.

set /p choice="Choose option: "

if "%choice%"=="1" goto SETUP
if "%choice%"=="2" goto BACKEND
if "%choice%"=="3" goto FRONTEND
if "%choice%"=="4" goto DEMO
if "%choice%"=="5" goto EXIT

:SETUP
echo.
echo Running setup... (this may take several minutes)
python setup.py
pause
goto MENU

:BACKEND
echo.
echo Starting Backend API on http://127.0.0.1:8000
start "ICU Backend" cmd /k "cd /d %~dp0 && python backend/api.py"
echo Backend started in new window.
pause
goto MENU

:FRONTEND
echo.
echo Starting Streamlit Dashboard...
start "ICU Dashboard" cmd /k "cd /d %~dp0 && streamlit run frontend/dashboard.py"
echo Dashboard started in new window. Opening browser...
timeout /t 3
start http://localhost:8501
pause
goto MENU

:DEMO
echo.
echo Generating synthetic demo data...
python generate_demo_data.py
echo.
echo Done! Update config/config.py to point EICU_RAW_PATH to 'demo_data'
pause
goto MENU

:EXIT
exit
