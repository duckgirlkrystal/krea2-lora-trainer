# =============================================================================
#  Krystal's LoRA Trainer - Windows environment bootstrap
# =============================================================================
#  Called by the .bat launchers. Not meant to be run by hand.
#
#  -Mode cloud : install the small client (requests, paramiko, pillow) only
#  -Mode local : the above, plus hardware gates and a full ai-toolkit install
#
#  Written for Windows PowerShell 5.1, which is what ships with Windows and
#  is what the .bat files invoke. Avoid PowerShell 7 syntax here.
#
#  Everything is installed inside this folder. Nothing is installed
#  system-wide except Python itself, and only if it is missing.
# =============================================================================

param(
    [ValidateSet("cloud", "local")]
    [string]$Mode = "cloud"
)

$ErrorActionPreference = "Stop"
# Invoke-WebRequest is roughly an order of magnitude slower with the progress
# bar enabled, and PS 5.1 will not negotiate TLS 1.2 by default on older builds.
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$TrainerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root       = Split-Path -Parent $TrainerDir
$VenvDir    = Join-Path $TrainerDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$ReadyFlag  = Join-Path $VenvDir ".ready"
$AiToolkit  = Join-Path $Root "ai-toolkit"
$AiPython   = Join-Path $AiToolkit ".venv\Scripts\python.exe"
$AiReady    = Join-Path $AiToolkit ".deps_ready"
$WorkDir    = Join-Path $Root "work"
$HardwareJson = Join-Path $WorkDir "hardware.json"

$PythonInstallerUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
$AiToolkitZipUrl    = "https://github.com/ostris/ai-toolkit/archive/refs/heads/main.zip"

# Requirements for the local trainer, roughly. See READ ME FIRST.txt.
$MinVramGb = 15
$MinDiskGb = 60
$GoodRamGb = 60

function Say($m)     { Write-Host "  $m" }
function StepMsg($m) { Write-Host "  $m" }
function OkMsg($m)   { Write-Host "  [OK] $m" }
function WarnMsg($m) { Write-Host "  [!]  $m" -ForegroundColor Yellow }
function BadMsg($m)  { Write-Host "  [X]  $m" -ForegroundColor Red }
function HeaderMsg($m) {
    Write-Host ""
    Write-Host ("  " + ("-" * 62))
    Write-Host "   $m"
    Write-Host ("  " + ("-" * 62))
}

function Fail($message) {
    Write-Host ""
    Write-Host ("=" * 66) -ForegroundColor Red
    BadMsg $message
    Write-Host ("=" * 66) -ForegroundColor Red
    Write-Host ""
    exit 1
}

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------

function Find-Python {
    # The py launcher is the most reliable way to find a usable interpreter,
    # because `python` on Windows is often the Microsoft Store stub that does
    # nothing but open the Store.
    foreach ($version in @("3.12", "3.11", "3.10")) {
        try {
            $out = & py "-$version" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) { return $out.Trim() }
        } catch { }
    }
    foreach ($candidate in @("python", "python3")) {
        try {
            $out = & $candidate -c "import sys; print(sys.executable if sys.version_info[:2] >= (3,10) else '')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out -and $out.Trim() -ne "") { return $out.Trim() }
        } catch { }
    }
    foreach ($base in @("$env:LOCALAPPDATA\Programs\Python", "C:\Python311", "C:\Python312")) {
        if (Test-Path $base) {
            $found = Get-ChildItem -Path $base -Recurse -Filter "python.exe" -ErrorAction SilentlyContinue |
                     Select-Object -First 1
            if ($found) { return $found.FullName }
        }
    }
    return $null
}

function Install-Python {
    HeaderMsg "Installing Python (one time, about 2 minutes)"
    Say "Your PC does not have Python, which this needs in order to run."
    Say "Downloading the official installer from python.org..."

    $installer = Join-Path $env:TEMP "python-3.11.9-amd64.exe"
    try {
        Invoke-WebRequest -Uri $PythonInstallerUrl -OutFile $installer -UseBasicParsing
    } catch {
        Fail ("Could not download Python. Check your internet connection.`n" +
              "  You can also install it yourself from python.org (version 3.11),`n" +
              "  ticking 'Add python.exe to PATH', then run this again.")
    }

    StepMsg "Installing (a User Account Control prompt may appear - click Yes)..."
    # Per-user install needs no admin rights on most machines.
    $args = @("/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_pip=1",
              "Include_launcher=1", "Include_test=0")
    $proc = Start-Process -FilePath $installer -ArgumentList $args -Wait -PassThru
    Remove-Item $installer -ErrorAction SilentlyContinue

    if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
        Fail ("The Python installer failed (code $($proc.ExitCode)).`n" +
              "  Please install Python 3.11 from python.org yourself, tick`n" +
              "  'Add python.exe to PATH' during setup, then run this again.")
    }

    # PATH changes do not reach an already-running process.
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")

    $python = Find-Python
    if (-not $python) {
        Fail ("Python installed but could not be found afterwards.`n" +
              "  Please close this window, open a new one, and try again.")
    }
    OkMsg "Python installed."
    return $python
}

# -----------------------------------------------------------------------------
# Client environment
# -----------------------------------------------------------------------------

function Ensure-ClientVenv {
    if (Test-Path $ReadyFlag) {
        OkMsg "Tools already installed."
        return
    }

    $python = Find-Python
    if (-not $python) { $python = Install-Python }

    HeaderMsg "Installing the tools this needs (one time)"

    if (-not (Test-Path $VenvPython)) {
        StepMsg "Creating a private Python environment..."
        & $python -m venv $VenvDir
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPython)) {
            Fail ("Could not create the Python environment.`n" +
                  "  If your antivirus is blocking it, allow this folder and retry.")
        }
    }

    StepMsg "Downloading a few small packages..."
    & $VenvPython -m pip install --upgrade pip --quiet --disable-pip-version-check
    # pillow-heif is listed last and allowed to fail: it only adds iPhone .HEIC
    # support, and there is no reason to block the whole install over it.
    & $VenvPython -m pip install --quiet --disable-pip-version-check `
        "requests>=2.31" "paramiko>=3.4" "pillow>=10.0"
    if ($LASTEXITCODE -ne 0) {
        Fail ("Could not install the required packages.`n" +
              "  This is usually a firewall or antivirus blocking pip.`n" +
              "  Check your internet connection and try again.")
    }
    & $VenvPython -m pip install --quiet --disable-pip-version-check "pillow-heif>=0.16" 2>$null
    if ($LASTEXITCODE -ne 0) {
        WarnMsg "iPhone .HEIC photo support could not be installed."
        WarnMsg "Everything else works; just use JPG or PNG photos."
    }

    New-Item -ItemType File -Path $ReadyFlag -Force | Out-Null
    OkMsg "Tools installed."
}

# -----------------------------------------------------------------------------
# Hardware gates (local mode only)
# -----------------------------------------------------------------------------

function Get-GpuInfo {
    try {
        $raw = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
    } catch {
        return $null
    }
    if ($LASTEXITCODE -ne 0 -or -not $raw) { return $null }

    $best = $null
    foreach ($line in @($raw)) {
        if (-not $line) { continue }
        $parts = $line -split ","
        if ($parts.Count -lt 2) { continue }
        $name = $parts[0].Trim()
        $mib = 0
        if (-not [int]::TryParse($parts[1].Trim(), [ref]$mib)) { continue }
        $gb = [math]::Round($mib / 1024.0, 1)
        if (-not $best -or $gb -gt $best.VramGb) {
            $best = [pscustomobject]@{ Name = $name; VramGb = $gb }
        }
    }
    return $best
}

function Check-Hardware {
    HeaderMsg "Checking whether your PC can do this"

    $gpu = Get-GpuInfo
    if (-not $gpu) {
        Fail ("No NVIDIA graphics card was found on this PC.`n`n" +
              "  Training needs an NVIDIA card with at least ${MinVramGb}GB of memory.`n" +
              "  AMD and Intel cards, and laptops with only built-in graphics,`n" +
              "  cannot run this.`n`n" +
              "  GOOD NEWS: you can still do it in the cloud for about `$3.`n" +
              "  Close this window and double-click:`n" +
              "      1 - TRAIN IN THE CLOUD.bat")
    }

    Say "Graphics card:  $($gpu.Name)"
    Say "Card memory:    $($gpu.VramGb) GB"

    if ($gpu.VramGb -lt $MinVramGb) {
        Fail ("Your graphics card has $($gpu.VramGb)GB of memory, but at least`n" +
              "  ${MinVramGb}GB is needed. The AI model is simply too big for this card.`n`n" +
              "  GOOD NEWS: you can still do it in the cloud for about `$3.`n" +
              "  Close this window and double-click:`n" +
              "      1 - TRAIN IN THE CLOUD.bat")
    }
    OkMsg "Graphics card is good enough."

    $drive = (Get-Item $Root).PSDrive.Name
    $free = (Get-PSDrive $drive).Free / 1GB
    Say ("Free disk space: {0:N0} GB on drive {1}:" -f $free, $drive)
    if ($free -lt $MinDiskGb) {
        Fail (("Only {0:N0}GB of disk space is free on drive {1}:, but about" -f $free, $drive) + "`n" +
              "  ${MinDiskGb}GB is needed. The AI model alone is about 35GB.`n`n" +
              "  Free up some space and try again, or use the cloud instead:`n" +
              "      1 - TRAIN IN THE CLOUD.bat")
    }
    OkMsg "Enough disk space."

    $ramGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 0)
    Say "System memory:  $ramGb GB"
    if ($ramGb -lt $GoodRamGb) {
        Write-Host ""
        WarnMsg "You have ${ramGb}GB of system RAM. 64GB is strongly recommended."
        WarnMsg "With less, training can crash while it compresses the model."
        Write-Host ""
        Say "  You can avoid that crash by increasing your Windows page file:"
        Say "    1. Press the Windows key and type: advanced system settings"
        Say "    2. Open it, then under 'Performance' click 'Settings...'"
        Say "    3. Go to the 'Advanced' tab, then click 'Change...'"
        Say "    4. Untick 'Automatically manage paging file size'"
        Say "    5. Select 'Custom size' and enter 40000 for both boxes"
        Say "    6. Click Set, then OK, then restart your PC"
        Write-Host ""
        Say "  If it crashes without doing this, that is why."
        Write-Host ""
    } else {
        OkMsg "Plenty of system memory."
    }

    # There is no reliable way to read this setting, so it is always shown.
    # It matters: with the fallback on, a config that does not fit does not
    # error, it silently runs about fifty times slower.
    Write-Host ""
    WarnMsg "ONE SETTING TO CHANGE - this makes a big difference:"
    Say "    1. Right-click your desktop and open the NVIDIA Control Panel"
    Say "    2. Click 'Manage 3D settings' on the left"
    Say "    3. Find 'CUDA - Sysmem Fallback Policy' in the list"
    Say "    4. Set it to 'Prefer No Sysmem Fallback'"
    Say "    5. Click Apply"
    Write-Host ""
    Say "  Without this, if something does not fit, Windows hides the problem"
    Say "  and training runs about 50x slower instead of telling you."
    Write-Host ""

    New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null
    $hw = @{
        gpu_name = $gpu.Name
        vram_gb  = $gpu.VramGb
        ram_gb   = $ramGb
        disk_free_gb = [math]::Round($free, 1)
    }
    ($hw | ConvertTo-Json) | Set-Content -Path $HardwareJson -Encoding ASCII
}

# -----------------------------------------------------------------------------
# ai-toolkit (local mode only)
# -----------------------------------------------------------------------------

function Ensure-AiToolkit {
    if (Test-Path $AiReady) {
        OkMsg "Training software already installed."
        return
    }

    HeaderMsg "Installing the training software (one time, 15-30 minutes)"
    Say "This downloads several gigabytes. Please leave it running."
    Write-Host ""

    if (-not (Test-Path (Join-Path $AiToolkit "run.py"))) {
        StepMsg "Downloading ai-toolkit..."
        $zip = Join-Path $env:TEMP "ai-toolkit-main.zip"
        $extract = Join-Path $env:TEMP "ai-toolkit-extract"
        Remove-Item $extract -Recurse -Force -ErrorAction SilentlyContinue
        try {
            # A zip rather than `git clone`, so the user does not need git.
            Invoke-WebRequest -Uri $AiToolkitZipUrl -OutFile $zip -UseBasicParsing
            Expand-Archive -Path $zip -DestinationPath $extract -Force
        } catch {
            Fail ("Could not download the training software.`n" +
                  "  Check your internet connection and try again.`n" +
                  "  Technical detail: $($_.Exception.Message)")
        }
        $inner = Get-ChildItem -Path $extract -Directory | Select-Object -First 1
        if (-not $inner) { Fail "The downloaded training software looks corrupted." }
        Remove-Item $AiToolkit -Recurse -Force -ErrorAction SilentlyContinue
        Move-Item -Path $inner.FullName -Destination $AiToolkit
        Remove-Item $zip -Force -ErrorAction SilentlyContinue
        Remove-Item $extract -Recurse -Force -ErrorAction SilentlyContinue
        OkMsg "Downloaded."
    } else {
        OkMsg "Already downloaded."
    }

    $python = Find-Python
    if (-not $python) { $python = Install-Python }

    if (-not (Test-Path $AiPython)) {
        StepMsg "Creating its Python environment..."
        & $python -m venv (Join-Path $AiToolkit ".venv")
        if (-not (Test-Path $AiPython)) {
            Fail "Could not create the training software's Python environment."
        }
    }

    & $AiPython -m pip install --upgrade pip --quiet --disable-pip-version-check

    StepMsg "Installing PyTorch with GPU support (this is the big one)..."
    $torchOk = $false
    foreach ($index in @("https://download.pytorch.org/whl/cu130",
                         "https://download.pytorch.org/whl/cu128")) {
        & $AiPython -m pip install --quiet --disable-pip-version-check `
            torch torchvision torchaudio --index-url $index
        if ($LASTEXITCODE -eq 0) {
            & $AiPython -c "import torch; assert torch.cuda.is_available()" 2>$null
            if ($LASTEXITCODE -eq 0) { $torchOk = $true; break }
        }
        WarnMsg "That PyTorch build did not work, trying another..."
    }
    if (-not $torchOk) {
        Fail ("PyTorch could not be installed with GPU support.`n`n" +
              "  Your graphics driver may be too old. Update it from`n" +
              "  nvidia.com/drivers and try again.`n`n" +
              "  Or just use the cloud instead - it always works:`n" +
              "      1 - TRAIN IN THE CLOUD.bat")
    }
    OkMsg "PyTorch installed with GPU support."

    StepMsg "Installing everything else (10-20 minutes)..."
    Push-Location $AiToolkit
    try {
        & $AiPython -m pip install --quiet --disable-pip-version-check -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            Fail ("Some of the training software could not be installed.`n" +
                  "  Check your internet connection and try again. If it keeps`n" +
                  "  failing, use the cloud version instead.")
        }
    } finally {
        Pop-Location
    }

    # Neither of these is in ai-toolkit's requirements, and neither is fatal.
    # triton-windows unlocks faster code paths; bitsandbytes provides the
    # adamw8bit optimizer the configs ask for.
    StepMsg "Installing two optional Windows extras..."
    & $AiPython -m pip install --quiet --disable-pip-version-check "triton-windows" 2>$null
    if ($LASTEXITCODE -ne 0) { WarnMsg "Optional speed-up (triton) unavailable. Not a problem." }
    & $AiPython -m pip install --quiet --disable-pip-version-check --upgrade "bitsandbytes" 2>$null
    if ($LASTEXITCODE -ne 0) { WarnMsg "Optional memory-saver (bitsandbytes) unavailable." }

    # requirements.txt resolution can quietly replace the CUDA build of torch
    # with a CPU-only one, which fails much later and very confusingly.
    & $AiPython -c "import torch; assert torch.cuda.is_available()" 2>$null
    if ($LASTEXITCODE -ne 0) {
        StepMsg "Repairing PyTorch (something replaced the GPU version)..."
        & $AiPython -m pip install --quiet --disable-pip-version-check --force-reinstall `
            torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu128"
        & $AiPython -c "import torch; assert torch.cuda.is_available()" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Fail ("PyTorch lost GPU support and could not be repaired.`n" +
                  "  Please use the cloud version instead:`n" +
                  "      1 - TRAIN IN THE CLOUD.bat")
        }
    }

    New-Item -ItemType File -Path $AiReady -Force | Out-Null
    OkMsg "Training software installed."
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

try {
    Ensure-ClientVenv
    if ($Mode -eq "local") {
        Check-Hardware
        Ensure-AiToolkit
        Write-Host ""
        OkMsg "Your PC is ready."
    }
    Write-Host ""
    exit 0
} catch {
    Write-Host ""
    Write-Host ("=" * 66) -ForegroundColor Red
    BadMsg "Setup failed unexpectedly."
    Write-Host "  $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ("=" * 66) -ForegroundColor Red
    Write-Host ""
    exit 1
}
