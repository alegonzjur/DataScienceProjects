@echo off
echo =========================================================
echo  PistonCup Setup Script
echo =========================================================

:: Name of the conda environment to create or update
set CONDA_ENV_NAME=ML

:: 1. Checking if Conda is installed
where conda >NUL 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo [STEP 1/4] Conda not found.
    echo Please, install Miniconda or Anaconda to continue.
    echo Visit: https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
) else (
    echo [STEP 1/3] Conda detected.
)

echo.
echo [STEP 2/4] Checking if the Conda environment '%CONDA_ENV_NAME%' exists...

conda env remove --name %CONDA_ENV_NAME% -y || echo.
echo.
echo [STEP 3/4] Creating/Updating conda environment '%CONDA_ENV_NAME%'...
:: --force ensures that the environment is updated if it already exists.
conda env create -f ML.yml
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Encountered error creating/updating the conda environment.
    echo Make sure that the file ML.yml is correct and there aren't network problems.
    echo.
    pause
    exit /b 1
) else (
    echo Environment '%CONDA_ENV_NAME%' created/updated successfully.
)

echo.
echo [STEP 3/3] Activating environment and creating the app exe file...
call conda activate %CONDA_ENV_NAME%
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Error activating conda environment '%CONDA_ENV_NAME%'.
    echo.
    pause
    exit /b 1
) else (
    echo Environment activated.
)

:: Generating the exe.
echo.
echo Creating the exe file 'PistonCup.exe'. This process may take a few minutes...
:: Removing the compilation folders for a clean compilation
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

:: PyInstaller command to create .exe
:: --noconsole: The exe will not open a console window.
:: --onefile: Packs everything into a single file
:: --name "PistonCup": Defines the name of the executable
:: --add-data "source;destination": Includes the necessary folders
::    templates y static son para Flask. model es para tus modelos de ML.
pyinstaller --noconsole --onefile --name "PistonCup" ^
    --add-data "templates;templates" ^
    --add-data "static;static" ^
    --add-data "model;model" ^
    app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Error encountered at creating PyInstaller executable.
    echo See the previous exit for more details.
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo Exe file 'PistonCup.exe' created successfully at folder 'dist'.
    echo You can copy this file to any location to run the application.
    echo.
    echo =========================================================
    echo  INSTALLATION AND EXECUTABLE CREATION COMPLETED.
    echo  Now, you can execute 'dist\PistonCup.exe'.
    echo =========================================================
)

pause
exit /b 0