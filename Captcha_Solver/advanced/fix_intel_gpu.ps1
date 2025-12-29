# Advanced Lab: Intel XPU Runtime Dependency Automator
$VENV_BASE = "$env:USERPROFILE\Captcha-AI\nputest_advanced_env"
$TORCH_LIB = "$VENV_BASE\Lib\site-packages\torch\lib"
$INTEL_BIN = "$VENV_BASE\Library\bin"
$TEMP_ZIP  = "$env:TEMP\libuv_dist.zip"
$UV_URL    = "https://dist.libuv.org/dist/v1.48.0/libuv-v1.48.0-x64.zip"

Write-Host "--- [NPU/XPU Advanced Lab: Dependency Bridge] ---" -ForegroundColor Cyan

# 1. Automated Retrieval of libuv.dll
if (-not (Test-Path "$TORCH_LIB\libuv.dll")) {
    Write-Host "[!] libuv.dll missing. Fetching from official mirror..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $UV_URL -OutFile $TEMP_ZIP
    
    # Extract only the DLL using CLI (no GUI interaction)
    Expand-Archive -Path $TEMP_ZIP -DestinationPath "$env:TEMP\libuv_temp" -Force
    $extractedDll = Get-ChildItem -Path "$env:TEMP\libuv_temp" -Filter "libuv.dll" -Recurse | Select-Object -First 1
    
    if ($extractedDll) {
        Copy-Item $extractedDll.FullName -Destination $TORCH_LIB -Force
        Write-Host "[+] libuv.dll successfully injected into torch core." -ForegroundColor Green
    }
    Remove-Item $TEMP_ZIP; Remove-Item "$env:TEMP\libuv_temp" -Recurse -ErrorAction SilentlyContinue
}

# 2. Intel OneAPI Runtime Consolidation
if (Test-Path $INTEL_BIN) {
    Write-Host "[*] Consolidating Intel 2025.0 Runtime libraries..." -ForegroundColor Gray
    $dependencies = @("sycl8.dll", "mkl_rt.2.dll", "libiomp5md.dll")
    
    foreach ($dll in $dependencies) {
        if (Test-Path "$INTEL_BIN\$dll") {
            Copy-Item "$INTEL_BIN\$dll" -Destination $TORCH_LIB -Force
            Write-Host "[+] Linked: $dll" -ForegroundColor Green
        }
    }

    # 3. Arrow Lake Hardware Alias (Bugfix for IPEX 2.5)
    # IPEX 2.5 expects sycl7.dll, but Arrow Lake (v2025) provides sycl8.dll.
    if (Test-Path "$INTEL_BIN\sycl8.dll") {
        Copy-Item "$INTEL_BIN\sycl8.dll" -Destination "$TORCH_LIB\sycl7.dll" -Force
        Write-Host "[+] Created sycl7 hardware-abstraction alias." -ForegroundColor Cyan
    }
}

Write-Host "--- [Process Complete] ---" -ForegroundColor Cyan
