@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install it from python.org, tick "Add to PATH", then run this again.
  pause
  exit /b 1
)
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
python scripts\make_extract.py
if errorlevel 1 goto failed
python scripts\run_analysis.py
if errorlevel 1 goto failed
echo.
echo Finished. The file to upload is data\votes.csv.gz
pause
exit /b 0
:failed
echo.
echo Something went wrong. Copy the text above and send it over.
pause
exit /b 1
