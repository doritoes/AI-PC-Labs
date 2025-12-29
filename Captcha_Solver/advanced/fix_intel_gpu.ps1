# fix_intel_gpu.ps1 (v7 - The "Lab-Ready" Version)
$VENV_BASE = "$env:USERPROFILE\Captcha-AI\nputest_advanced_env"
$TORCH_LIB = "$VENV_BASE\Lib\site-packages\torch\lib"

# Official NuGet Sources for missing XPU dependencies
$PTI_URL   = "https://www.nuget.org/api/v2/package/intel-pti/0.10.0"
$UV_URL    = "https://www.nuget.org/api/v2/package/Libuv/1.10.0"

Write-Host "--- [XPU Advanced Lab: Hardware Bridge] ---" -ForegroundColor Cyan

function Install-DllFromNuGet ($url, $dllName) {
    if (-not (Test-Path "$TORCH_LIB\$dllName")) {
        Write-Host "[!] Fetching $dllName..." -ForegroundColor Yellow
        $tempPkg = "$env:TEMP\temp_pkg.nupkg"
        $tempDir = "$env:TEMP\temp_extract"
        
        Invoke-WebRequest -Uri $url -OutFile $tempPkg -ErrorAction Stop
        Expand-Archive -Path $tempPkg -DestinationPath $tempDir -Force
        
        $match = Get-ChildItem -Path $tempDir -Filter $dllName -Recurse | 
                 Where-Object { $_.FullName -match "x64|amd64" } | Select-Object -First 1
        
        if ($match) {
            Copy-Item $match.FullName -Destination $TORCH_LIB -Force
            Write-Host "[+] Injected: $dllName" -ForegroundColor Green
        }
        Remove-Item $tempPkg; Remove-Item $tempDir -Recurse -ErrorAction SilentlyContinue
    }
}

# 1. Inject the missing PTI and UV libraries
Install-DllFromNuGet $PTI_URL "pti_view-0-10.dll"
Install-DllFromNuGet $UV_URL  "libuv.dll"

# 2. Consolidate SYCL and MKL (Search within venv)
$intelBin = "$VENV_BASE\Library\bin"
if (Test-Path $intelBin) {
    $dlls = @("sycl8.dll", "mkl_rt.2.dll", "libiomp5md.dll")
    foreach ($dll in $dlls) {
        if (Test-Path "$intelBin\$dll") {
            Copy-Item "$intelBin\$dll" -Destination $TORCH_LIB -Force
            Write-Host "[+] Linked: $dll" -ForegroundColor Green
        }
    }
    # 3. Arrow Lake Alias Trick
    Copy-Item "$intelBin\sycl8.dll" -Destination "$TORCH_LIB\sycl7.dll" -Force
    Write-Host "[+] Created sycl7.dll alias for hardware compatibility." -ForegroundColor Cyan
}

Write-Host "--- [Process Complete] ---" -ForegroundColor Cyan
