@echo off
title Push to GitHub - Odessacool1
cls
echo ========================================================
echo   Pushing Repository to GitHub: Odessacool1
echo ========================================================
echo.
git branch -M main
echo Pushing code to main branch...
git push -u origin main
echo.
echo ========================================================
echo   Done! Check your repository:
echo   https://github.com/Odessacool1/multi-agent-memory-system
echo ========================================================
pause
