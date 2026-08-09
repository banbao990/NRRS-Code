@echo off
setlocal

for %%I in ("%~dp0..\common\assets\scenes-nrrs\noenv\apartment2") do set "SCENE_DIR=%%~fI"

set "INPUT=%SCENE_DIR%\scene.bin"
set "PREFIX=%SCENE_DIR%\scene.bin.part."
set "CHUNK_SIZE=83886080"

if not exist "%INPUT%" (
    echo [ERROR] File not found:
    echo %INPUT%
    pause
    exit /b 1
)

echo Splitting:
echo %INPUT%
echo.
echo Chunk size: 80 MB
echo.

powershell -NoProfile -Command "$inputFile='%INPUT%'; $prefix='%PREFIX%'; $chunkSize=%CHUNK_SIZE%; $buffer=New-Object byte[] $chunkSize; $fs=[System.IO.File]::OpenRead($inputFile); $index=0; try { while (($read=$fs.Read($buffer,0,$buffer.Length)) -gt 0) { $outFile='{0}{1:D2}' -f $prefix,$index; $ofs=[System.IO.File]::Create($outFile); try { $ofs.Write($buffer,0,$read) } finally { $ofs.Dispose() }; Write-Host ('Created: '+$outFile); $index++ } } finally { $fs.Dispose() }"

if errorlevel 1 (
    echo.
    echo [ERROR] Split failed.
    pause
    exit /b 1
)

echo.
echo Split completed.
pause