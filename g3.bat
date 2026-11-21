@echo off
cd /d "%~dp0"
git add -A
git -c core.autocrlf=false commit -m "fix(ci): grant packages:write so the image can be pushed" -m "The workflow never declared token permissions, so GITHUB_TOKEN kept its read-only default and the push was denied with 'installation not allowed to Create organization package'." > "%~dp0c.txt" 2>&1
echo COMMIT_EXIT=%ERRORLEVEL% >> "%~dp0c.txt"
git -c http.sslBackend=openssl push origin main >> "%~dp0c.txt" 2>&1
echo PUSH_EXIT=%ERRORLEVEL% >> "%~dp0c.txt"
"C:\Program Files\GitHub CLI\gh.exe" workflow run docker.yml --repo BiancaVeritas/metatube-sdk-go --ref main >> "%~dp0c.txt" 2>&1
echo DISPATCH_EXIT=%ERRORLEVEL% >> "%~dp0c.txt"
