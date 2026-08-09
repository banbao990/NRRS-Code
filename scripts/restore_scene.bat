@echo off
setlocal

for %%I in ("%~dp0..\common\assets\scenes-nrrs\noenv\apartment2") do set "SCENE_DIR=%%~fI"

set "OUTPUT=%SCENE_DIR%\scene.bin"
set "PREFIX=%SCENE_DIR%\scene.bin.part."

if not exist "%PREFIX%00" (
    echo [ERROR] Split files not found:
    echo %PREFIX%00
    pause
    exit /b 1
)

if exist "%OUTPUT%" (
    echo Existing scene.bin will be overwritten.
    del /f /q "%OUTPUT%"
)

echo Restoring:
echo %OUTPUT%
echo.

powershell -NoProfile -Command "$outputFile='%OUTPUT%'; $prefix='%PREFIX%'; $files=Get-ChildItem -File ($prefix+'*') | Sort-Object Name; $ofs=[System.IO.File]::Create($outputFile); try { foreach ($file in $files) { Write-Host ('Reading: '+$file.Name); $ifs=[System.IO.File]::OpenRead($file.FullName); try { $ifs.CopyTo($ofs) } finally { $ifs.Dispose() } } } finally { $ofs.Dispose() }"

if errorlevel 1 (
    echo.
    echo [ERROR] Restore failed.
    pause
    exit /b 1
)

echo.
echo Restore completed:
echo %OUTPUT%
pause