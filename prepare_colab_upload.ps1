# prepare_colab_upload.ps1
# Run this script from the project root directory to create DOLOS_upload.zip
# Then upload that zip to your Google Drive root folder.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File prepare_colab_upload.ps1

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Preparing DOLOS_upload.zip for Colab" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check that source directories exist
$audioDir = Join-Path $projectRoot "audio_output"
$videoDir = Join-Path $projectRoot "DOLOSDATA\raw_videos"
$outputZip = Join-Path $projectRoot "DOLOS_upload.zip"

if (-not (Test-Path $audioDir)) {
    Write-Host "ERROR: audio_output/ not found at $audioDir" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $videoDir)) {
    Write-Host "ERROR: DOLOSDATA/raw_videos/ not found at $videoDir" -ForegroundColor Red
    exit 1
}

# Count files
$audioCount = (Get-ChildItem -Path $audioDir -Filter "*.wav" -File).Count
$videoCount = (Get-ChildItem -Path $videoDir -Filter "*.mp4" -File).Count

Write-Host "Found:" -ForegroundColor Yellow
Write-Host "  Audio files (audio_output/):  $audioCount .wav files"
Write-Host "  Raw videos (raw_videos/):     $videoCount .mp4 files"
Write-Host ""

# Check available disk space
$audioSize = (Get-ChildItem -Path $audioDir -Recurse -File | Measure-Object -Property Length -Sum).Sum
$videoSize = (Get-ChildItem -Path $videoDir -Recurse -File | Measure-Object -Property Length -Sum).Sum
$totalSizeMB = [math]::Round(($audioSize + $videoSize) / 1MB, 2)
Write-Host "Total data size: $totalSizeMB MB" -ForegroundColor Yellow
Write-Host ""

# Create temp staging directory
$tempDir = Join-Path $env:TEMP "DOLOS_staging_$(Get-Random)"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

Write-Host "Staging files..." -ForegroundColor Gray

# Copy audio files
$stagedAudio = Join-Path $tempDir "audio_files"
Copy-Item -Path $audioDir -Destination $stagedAudio -Recurse -Force

# Copy raw videos
$stagedVideos = Join-Path $tempDir "raw_videos"
Copy-Item -Path $videoDir -Destination $stagedVideos -Recurse -Force

Write-Host "Creating zip archive (this may take several minutes)..." -ForegroundColor Yellow

# Remove old zip if exists
if (Test-Path $outputZip) { Remove-Item $outputZip -Force }

# Create zip
Compress-Archive -Path "$tempDir\*" -DestinationPath $outputZip -CompressionLevel Fastest

# Cleanup
Remove-Item -Path $tempDir -Recurse -Force

$zipSizeMB = [math]::Round((Get-Item $outputZip).Length / 1MB, 2)
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  DONE!" -ForegroundColor Green
Write-Host "  Created: DOLOS_upload.zip ($zipSizeMB MB)" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host "  1. Open Google Drive in your browser"
Write-Host "  2. Upload DOLOS_upload.zip to My Drive (root folder)"
Write-Host "     URL: https://drive.google.com/drive/my-drive"
Write-Host "  3. Open the colab_train.ipynb notebook in Colab"
Write-Host "  4. Set runtime to T4 GPU"
Write-Host "  5. Run all cells"
Write-Host ""
