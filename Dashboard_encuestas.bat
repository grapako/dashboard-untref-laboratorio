@echo off
setlocal EnableDelayedExpansion
REM pushd %~dp0
REM =========================================
REM Ejecutar finanzas_view.py en el env JIP_env
REM con detección automática de conda
REM =========================================

REM Nombre del environment
set "ENV_NAME=JIP_env"

REM Ruta del script (en la misma carpeta que el .bat)
set "SCRIPT_PATH=%~dp0Dashboard_encuestas.py"

:: ------------------ DETECCIÓN DE CONDA ------------------
:: 1) Intentar usar "where conda"
for /f "delims=" %%C in ('where conda 2^>nul') do (
    set "CONDA_EXE=%%C"
    set "CONDA_DIR=%%~dpC"
    set "ACTIVATE_BAT=!CONDA_DIR!activate.bat"
    goto :FOUND_CONDA
)

:: 2) Buscar instalación típica de Anaconda3
if exist "%USERPROFILE%\Anaconda3\Scripts\conda.exe" (
    set "CONDA_DIR=%USERPROFILE%\Anaconda3\Scripts\"
    set "ACTIVATE_BAT=%CONDA_DIR%activate.bat"
    goto :FOUND_CONDA
)

if exist "C:\ProgramData\Anaconda3\Scripts\conda.exe" (
    set "CONDA_DIR=C:\ProgramData\Anaconda3\Scripts\"
    set "ACTIVATE_BAT=%CONDA_DIR%activate.bat"
    goto :FOUND_CONDA
)

:: 3) Buscar instalación típica de Miniconda3
if exist "%USERPROFILE%\Miniconda3\Scripts\conda.exe" (
    set "CONDA_DIR=%USERPROFILE%\Miniconda3\Scripts\"
    set "ACTIVATE_BAT=%CONDA_DIR%activate.bat"
    goto :FOUND_CONDA
)

if exist "C:\ProgramData\Miniconda3\Scripts\conda.exe" (
    set "CONDA_DIR=C:\ProgramData\Miniconda3\Scripts\"
    set "ACTIVATE_BAT=%CONDA_DIR%activate.bat"
    goto :FOUND_CONDA
)

echo ERROR: No se encontró ninguna instalación de conda (Anaconda o Miniconda).
pause
exit /b 1

:FOUND_CONDA
REM =========================================
REM Activar el environment y ejecutar el script
REM =========================================
CALL "%ACTIVATE_BAT%" %ENV_NAME%
REM python "%SCRIPT_PATH%"
streamlit run "%SCRIPT_PATH%"

REM Mantener ventana abierta para ver salida/errores
pause