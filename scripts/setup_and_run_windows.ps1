# Set up and run the GeoData pipeline on Windows.
#   powershell -ExecutionPolicy Bypass -File scripts\setup_and_run_windows.ps1 -Root E:\coding
# Needs: git and Miniconda/Miniforge (conda) on PATH.
param(
    [string]$Root = "E:\coding",
    [string]$Branch = "claude/busy-ride-xglr2p",
    [switch]$SkipExtract,
    [string]$Bbox = ""          # e.g. "320000 420000 380000 520000" for a test area in section 3
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Root | Out-Null
$Repo = Join-Path $Root "GeoData"
if (-not (Test-Path $Repo)) {
    git clone https://github.com/richardkappa/GeoData.git $Repo
}
Set-Location $Repo
git fetch origin $Branch
git checkout $Branch
git pull origin $Branch

# conda environment (created once, updated afterwards)
$envs = conda env list
if ($envs -match "^GeoData\s") { conda env update -n GeoData -f environment.yml --prune }
else { conda env create -f environment.yml }

$env:GEODATA_DIR = Join-Path $Root "GeoData\data"
New-Item -ItemType Directory -Force -Path $env:GEODATA_DIR | Out-Null

if (-not $SkipExtract) { conda run -n GeoData --no-capture-output python 01_extract.py }
conda run -n GeoData --no-capture-output python 02_collate.py
if ($Bbox) { conda run -n GeoData --no-capture-output python 03_features.py --bbox $Bbox.Split(" ") }
else { conda run -n GeoData --no-capture-output python 03_features.py }
Write-Host "Done. Outputs in $env:GEODATA_DIR"
