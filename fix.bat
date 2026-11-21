@echo off
cd /d "%~dp0"
git add -A
git -c core.autocrlf=false commit -m "chore: drop a stray scratch script" > "%~dp0cc.txt" 2>&1
echo COMMIT_EXIT=%ERRORLEVEL% >> "%~dp0cc.txt"
git -c http.sslBackend=openssl push origin main >> "%~dp0cc.txt" 2>&1
echo PUSH_EXIT=%ERRORLEVEL% >> "%~dp0cc.txt"
