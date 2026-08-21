Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Pushing Repository to GitHub: Odessacool1" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

git branch -M main
git push -u origin main

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Check your repository: https://github.com/Odessacool1/multi-agent-memory-system" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan
Read-Host -Prompt "Press Enter to exit"
