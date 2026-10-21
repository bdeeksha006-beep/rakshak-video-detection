@echo off
echo Starting Video Fake Detection Platform...
echo.
echo Installing dependencies if needed...
pip install -r requirements.txt
echo.
echo Starting Flask server on port 5001...
echo Access the platform at: http://localhost:5001
echo.
python app.py
pause
