# fix_intel_gpu.ps1 (v5 - CLI Advanced Version)
$VENV_BASE = "$env:USERPROFILE\Captcha-AI\nputest_advanced_env"
$TORCH_LIB = "$VENV_BASE\Lib\site-packages\torch\lib"
$INTEL_BIN = "$VENV_BASE\Library\bin"

# Official Microsoft NuGet Source for libuv
$NUGET_URL = "https://www.nuget.org/api/v2/package/Libuv/1.10.0"
$TEMP_NUPKG = "$env:TEMP\libuv_dist.nupkg"
$TEMP_DIR   = "$env:TEMP\libuv_extracted"

Write-Host "--- [XPU Advanced Lab: CLI Dependency Injection] ---" -ForegroundColor Cyan

# 1. Automated Retrieval of libuv via NuGet API
if (-not (Test-Path "$TORCH_LIB\libuv.dll")) {
    Write-Host "[!] libuv.dll missing. Pulling via NuGet CLI Bridge..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $NUGET_URL -OutFile $TEMP_NUPKG -ErrorAction Stop
    
    # NuGet packages are just ZIP files. Extract the x64 binary.
    Expand-Archive -Path $TEMP_NUPKG -DestinationPath $TEMP_DIR -Force
    $extractedDll = Get-ChildItem -Path $TEMP_DIR -Filter "libuv.dll" -Recurse | 
                    Where-Object { $_.FullName -match "x64|amd64" } | Select-Object -First 1
    
    if ($extractedDll) {
        Copy-Item $extractedDll.FullName -Destination $TORCH_LIB -Force
        Write-Host "[+] libuv.dll (x64) successfully injected into torch core." -ForegroundColor Green
    }
    # Cleanup
    Remove-Item $TEMP_NUPKG; Remove-Item $TEMP_DIR -Recurse -ErrorAction SilentlyContinue
}

# 2. Intel OneAPI Runtime Consolidation (SYCL 2025)
if (Test-Path $INTEL_BIN) {
    Write-Host "[*] Consolidating Intel 2025.0 Runtime libraries..." -ForegroundColor Gray
    $dependencies = @("sycl8.dll", "mkl_rt.2.dll", "libiomp5md.dll")
    
    foreach ($dll in $dependencies) {
        if (Test-Path "$INTEL_BIN\$dll") {
            Copy-Item "$INTEL_BIN\$dll" -Destination $TORCH_LIB -Force
            Write-Host "[+] Linked: $dll" -ForegroundColor Green
        }
    }

    # 3. Arrow Lake Hardware Alias (Essential Bugfix)
    # Replicates sycl8 as sycl7 to satisfy IPEX 2.5 internal metadata calls.
    if (Test-Path "$TORCH_LIB\sycl8.dll") {
        Copy-Item "$TORCH_LIB\sycl8.dll" -Destination "$TORCH_LIB\sycl7.dll" -Force
        Write-Host "[+] Created hardware-abstraction alias: sycl7.dll -> sycl8.dll" -ForegroundColor Cyan
    }
}

Write-Host "--- [Process Complete] ---" -ForegroundColor Cyan
