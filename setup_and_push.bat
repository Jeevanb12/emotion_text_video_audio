@echo off
echo ===================================================
echo  Initializing Git and Preparing Push to GitHub
echo ===================================================

git init
git add .
git commit -m "Initial commit: Multimodal Emotion Detector"
git branch -M main

echo.
echo ===================================================
echo  Please enter your GitHub Repository URL
echo  Example: https://github.com/your-username/your-repo-name.git
echo ===================================================
set /p REPO_URL="Repository URL: "

git remote add origin %REPO_URL%
git push -u origin main

echo.
echo Setup Complete! Press any key to exit.
pause
