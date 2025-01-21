@echo off
REM Navigate to the current script directory
@cd %~dp0

REM Check if an argument was provided
@if "%1"=="" (
    @echo No simulation name provided. Please specify a simulation name.
    exit /b
)

REM Activate the virtual environment
@call env_vivarium\Scripts\activate

REM Stop any running instances
start /B cmd /C "python .\scripts\stop_all.py"
echo.

REM Add a delay to ensure the server starts first
@timeout /t 5 /nobreak >nul

REM Start the server in the background
start /B cmd /C "python .\scripts\run_server.py scene=%1"
echo.

REM Add a delay to ensure the server starts first
@timeout /t 5 /nobreak >nul

REM Start the interface in the background
start /B cmd /C "panel serve .\scripts\run_interface.py"
echo.

@REM TODO: Add a way to enforce the stop of the server and interface when the script is stopped