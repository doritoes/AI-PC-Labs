# fix_intel_gpu.ps1
# This script fixes the WinError 126 for Intel XPU by consolidating DLLs

$VENV_NAME = "nputest_advanced_env"
$BASE_PATH = "$env:USERPROFILE\Captcha-AI\$VENV_NAME"

Write-Host "--- Starting Intel XPU DLL Repair ---" -ForegroundColor Cyan

if (-not (Test-Path $BASE_PATH)) {
    Write-Error "Virtual environment not found at $BASE_PATH. Please check your folder name."
    exit
}

# 1. Define the Source and Destination folders
$sourceDir = "$BASE_PATH\Library\bin"
$destDir   = "$BASE_PATH\Lib\site-packages\torch\lib"

# 2. List of critical Arrow Lake / XPU DLLs
$dllsToCopy = @(
    "sycl8.dll",
    "mkl_rt.2.dll",
    "libiomp5md.dll",
    "pi_win_proxy_loader.dll"
)

# 3. Perform the copy
foreach ($dll in $dllsToCopy) {
    $srcFile = Join-Path $sourceDir $dll
    if (Test-Path $srcFile) {
        Write-Host "Found $dll, copying to torch/lib..." -ForegroundColor Green
        Copy-Item -Path $srcFile -Destination $destDir -Force
    } else {
        Write-Host "Warning: $dll not found in $sourceDir" -ForegroundColor Yellow
    }
}

Write-Host "--- Repair Complete ---" -ForegroundColor Cyan
Write-Host "Now run: python advanced/verify_xpu.py" -ForegroundColor White
