@echo off
SET ENV_NAME=ML

REM Check if conda is installed
CALL conda env list | FINDSTR /R /C:"^%ENV_NAME% " >nul
IF %ERRORLEVEL% NEQ 0 (
    ECHO Environment %ENV_NAME% not found. Installing...
    CALL conda env create -f ML.yml
) ELSE (
    ECHO Environment %ENV_NAME% found.
)
