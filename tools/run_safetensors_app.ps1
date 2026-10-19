param([ValidateRange(1024,65535)][int]$Port = 8891)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$hostDir = Join-Path $repo 'apps\host'
$python = Join-Path $repo '.venv-native\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Install requirements-native.txt in .venv-native first.' }
if (-not (Test-Path -LiteralPath (Join-Path $hostDir 'dist\index.html'))) {
    throw 'Build the app first: cd apps\host; npm ci; npm run build'
}
$node = (Get-Command node.exe -ErrorAction Stop).Source
$env:STRATA_NATIVE_REPO = $repo
$env:STRATA_NATIVE_PYTHON = $python
$env:STRATA_HOME = Join-Path $repo 'data\app-v020beta'
$env:STRATA_PORT = [string]$Port
$env:STRATA_BIND = '127.0.0.1'
$env:STRATA_REAL_ENGINE = '1'
$env:STRATA_UI_MODE = 'browser'
# A source launch must use this checkout, even from an installed app's shell.
Remove-Item Env:STRATA_CONTROL_DIR -ErrorAction SilentlyContinue
Remove-Item Env:STRATA_DIST -ErrorAction SilentlyContinue
if (-not $env:STRATA_HARNESS_DIR) {
    $candidates = @((Join-Path $repo 'build-deps\harness'),
                    (Join-Path $repo '..\Strata\Strata-v0.2.0-beta\harness'))
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'lib\bin.js')) {
            $env:STRATA_HARNESS_DIR = (Resolve-Path -LiteralPath $candidate).Path
            break
        }
    }
}
Write-Host "Strata safetensors app: http://127.0.0.1:$Port"
Write-Host 'Choose FP4-Local Native in Models. Stop the old model server before loading here.'
Write-Host 'The app has its own settings folder. Ctrl+C stops this foreground host.'
Push-Location -LiteralPath $hostDir
try {
    & $node server/index.mjs --dist dist
    if ($LASTEXITCODE -ne 0) { throw "App host exited with code $LASTEXITCODE" }
} finally { Pop-Location }
