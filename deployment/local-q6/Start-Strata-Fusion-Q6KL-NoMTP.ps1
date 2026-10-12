param(
    [switch]$CheckOnly,
    [switch]$NoBrowser,
    [switch]$RamOnly,
    [ValidateRange(2.0, 90.0)]
    [double]$MappedGiB = 64.0,
    [ValidateRange(1.0, 16.0)]
    [double]$HeadroomGiB = 1.0
)
$ErrorActionPreference = 'Stop'
$strataRoot = $PSScriptRoot
$python = 'G:\Strata\Strata\.venv\Scripts\python.exe'
$configPath = Join-Path $strataRoot 'strata-fusion-q6kl-no-mtp.json'
$url = 'http://127.0.0.1:8880'
try {
    $cfg = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $required = @($python, $cfg.exe, (Join-Path $strataRoot 'serve\server.py'), $cfg.tokenizer,
        $cfg.vision.exe, $cfg.vision.mmproj, $cfg.vision.model)
    foreach ($flag in @('--pack', '--native', '--ple-gguf', '--expert-profile')) {
        $idx = [Array]::IndexOf($cfg.args, $flag)
        if ($idx -lt 0) { throw "Missing config option: $flag" }
        $required += $cfg.args[$idx + 1]
    }
    if (@($cfg.args | Where-Object { $_ -like '--mtp*' -or $_ -eq '--batch-mtp' }).Count -gt 0) {
        throw 'This launcher requires MTP to remain disabled. Remove MTP options from its config.'
    }
    $shard1 = $cfg.args[[Array]::IndexOf($cfg.args, '--native') + 1]
    foreach ($i in 1..6) {
        $required += ($shard1 -replace '00001-of-00006', ('{0:D5}-of-00006' -f $i))
    }
    $packPath = $cfg.args[[Array]::IndexOf($cfg.args, '--pack') + 1]
    foreach ($name in @('native_experts.txt', 'dense.bin', 'index.txt', 'tokenizer\vocab.json')) {
        $required += Join-Path $packPath $name
    }
    foreach ($path in $required) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Missing file or directory: $path" }
    }
    $manifest = Get-Content -LiteralPath (Join-Path $packPath 'conversions.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($shard in $manifest.source_shards) {
        $path = Join-Path (Split-Path -Parent $shard1) $shard.name
        if ((Get-Item -LiteralPath $path).Length -ne $shard.size) {
            throw "Model shard size differs from the prepared Q6 pack: $path"
        }
    }
    if ($CheckOnly) {
        Write-Host '[Strata Fusion Q6KL NoMTP] Config, six model shards, Q6_K_XL engine, FP8 ngram, pack and vision: OK. MTP is disabled.'
        exit 0
    }
    $listener = Get-NetTCPConnection -State Listen -LocalPort 8880 -ErrorAction SilentlyContinue
    if ($listener) {
        try { $health = Invoke-RestMethod "$url/health" -TimeoutSec 3 } catch { $health = $null }
        if ($health -and $health.model -eq $cfg.model_name) {
            Write-Host "[Strata Fusion Q6KL NoMTP] Already running: $url"
            if (-not $NoBrowser) { Start-Process $url }
            exit 0
        }
        throw 'Port 8880 is in use. Close the other model server before starting Q6KL.'
    }
    if (Get-Process -Name 'strata*' -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -notlike 'strata-vision*' }) {
        throw 'Another Strata model is running. Close its server window before starting Q6KL.'
    }
    $totalGiB = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
    if ($totalGiB -lt 90) { throw 'This Q6KL configuration requires the installed 96 GB RAM.' }
    Remove-Item Env:STRATA_API_KEY -ErrorAction SilentlyContinue
    $env:STRATA_ALLOWED_HOSTS = '*'
    $env:STRATA_STAGE_PIN = '0'
    $env:STRATA_RESIDENT_PIN = $(if ($RamOnly) { '0' } else { '1' })
    $env:STRATA_PARTIAL_PIN = $(if ($RamOnly) { '0' } else { '1' })
    $env:STRATA_PARTIAL_PIN_GIB = $MappedGiB.ToString('0.###', [Globalization.CultureInfo]::InvariantCulture)
    $env:STRATA_RESIDENT_HEADROOM_GIB = $HeadroomGiB.ToString('0.###', [Globalization.CultureInfo]::InvariantCulture)
    Write-Host '[Strata Fusion Q6KL NoMTP] UD-Q6_K_XL + original FP8 ngram; MTP disabled; upstream 0.1.43.1 with local memory planning; SSD session cache disabled.'
    Write-Host '[Strata Fusion Q6KL NoMTP] 262144 context; int8 RAM KV (32768 GPU cells); RAM conversation snapshot cache disabled; one request at a time; PCIe fraction auto; prefill auto; GPU vision.'
    Write-Host "[Strata Fusion Q6KL NoMTP] Expert RAM budget: up to 81 GiB; available-memory clamp; headroom $HeadroomGiB GiB."
    Write-Host "[Strata Fusion Q6KL NoMTP] Mapped RAM: up to $MappedGiB GiB; the CUDA driver may accept less."
    Write-Host '[Strata Fusion Q6KL NoMTP] Q6_K_XL kernels are precompiled. No compilation is performed at startup.'
    Write-Host "[Strata Fusion Q6KL NoMTP] Browser/API: $url | $url/v1 | no API key required."
    Write-Host '[Strata Fusion Q6KL NoMTP] Loading this larger model takes time. Ctrl+C stops the server.'
    Set-Location -LiteralPath $strataRoot
    $serverArgs = @('serve\server.py', '--engine', 'strata', '--config', $configPath,
        '--host', '127.0.0.1', '--port', '8880')
    if (-not $NoBrowser) { $serverArgs += '--open' }
    & $python @serverArgs
    exit $LASTEXITCODE
} catch {
    Write-Host "[Strata Fusion Q6KL NoMTP] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "[Strata Fusion Q6KL NoMTP] Log: $strataRoot\strata-fusion-q6kl-no-mtp.log"
    exit 1
}
