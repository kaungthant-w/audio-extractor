@echo off
set "PROJECT_DIR=%CD%"
if not exist "tmp" mkdir tmp
if not exist "uploads" mkdir uploads

echo Starting PHP Server...
echo Project Directory: %PROJECT_DIR%
echo Temp Directory: %PROJECT_DIR%\tmp
echo Upload Max Size: 50M

php -d upload_tmp_dir="%PROJECT_DIR%\tmp" -d upload_max_filesize=50M -d post_max_size=50M -S 0.0.0.0:8000
