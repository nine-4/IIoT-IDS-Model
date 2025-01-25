@echo off
REM Turn off command echoing for a cleaner look

:: Title for the CMD window
title Python Script Executor - Thesis

:: Initial message
echo ============================================
echo Python Script Executor for Thesis Project
echo ============================================

:: Spacer
echo.
echo ----------- Activating Python Environment -----------
echo.

REM Activate the Python environment
call .\sklearn-env\Scripts\activate
if errorlevel 1 (
    echo ERROR: Failed to activate the Python environment.
    echo Make sure the environment path is correct.
    echo Exiting...
    pause >nul
    exit
)
echo Activated Successfully

:: Spacer
echo.
echo ----------- Running Python Scripts -----------
echo.

REM Run the Python scripts
echo Running remove.py...
python remove.py
if errorlevel 1 (
    echo ERROR: remove.py encountered an issue.
    echo Stopping script execution...
    pause >nul
    exit
)

:: Spacer
echo.
echo ----------- Stratified Split -----------
echo.

echo Running stratifiedsplit.py...
python stratifiedsplit.py
if errorlevel 1 (
    echo ERROR: stratifiedsplit.py encountered an issue.
    echo Stopping script execution...
    pause >nul
    exit
)

:: Spacer
echo.
echo ----------- Random Undersampling -----------
echo.

echo Running rnd-undersample.py...
python rnd-undersample.py
if errorlevel 1 (
    echo ERROR: rnd-undersample.py encountered an issue.
    echo Stopping script execution...
    pause >nul
    exit
)

:: Spacer
echo.
echo ----------- Random oversampling (not yet implemented) -----------
echo.

:: Spacer
echo.
echo ----------- Process Complete -----------
echo.

REM Prevent the Command Prompt from closing
echo All scripts executed successfully!
echo Press any key to exit...
pause >nul
exit
