# Ruby 3.2로 전환하는 PowerShell 스크립트
Write-Host "Switching to Ruby 3.2..." -ForegroundColor Green

# Ruby 3.2 경로를 PATH 앞쪽에 추가
$env:PATH = "C:\Ruby32-x64\bin;" + $env:PATH

Write-Host "Current Ruby version:" -ForegroundColor Yellow
ruby --version

Write-Host ""
Write-Host "You can now run Jekyll commands:" -ForegroundColor Cyan
Write-Host "  bundle install" -ForegroundColor White
Write-Host "  bundle exec jekyll serve" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to continue in this session..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")