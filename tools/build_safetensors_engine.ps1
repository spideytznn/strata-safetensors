param(
    [int]$Jobs = 2,
    [string[]]$Targets = @('strata', 'nvfp4_cutlass_test', 'native_bf16_attention_test', 'conversation_cache_test', 'conversation_memory_test', 'conversation_snapshot_test', 'native_prefill_policy_test', 'native_prefill_policy_unbuilt_test'),
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$OutputName = 'strata-classic-bf16-kv-cache',
    [string]$ToolDir = '',
    [string]$GgmlDir = '',
    [string]$CutlassDir = '',
    [string]$CudaDir = $env:CUDA_PATH,
    [string]$VcVars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$build = Join-Path $repo 'build-native-engine'
$outputExe = Join-Path $build "$OutputName.exe"
foreach ($process in Get-CimInstance Win32_Process -Filter "Name='$OutputName.exe'") {
    if ($process.ExecutablePath -and [string]::Equals($process.ExecutablePath, $outputExe, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Engine is running at $outputExe. Use another -OutputName to stage this build."
    }
}
function Find-BuildTool([string]$Name) {
    if ($ToolDir) {
        $candidate = Join-Path $ToolDir $Name
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    $candidate = Join-Path $repo ".venv-native\Scripts\$Name"
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    throw "Missing $Name. Install cmake and ninja in .venv-native or pass -ToolDir."
}
$cmake = Find-BuildTool 'cmake.exe'
$ninja = Find-BuildTool 'ninja.exe'
if (-not $CudaDir) { $CudaDir = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0' }
if (-not (Test-Path -LiteralPath (Join-Path $CudaDir 'bin\nvcc.exe'))) { throw "CUDA compiler missing at $CudaDir" }
if (-not (Test-Path -LiteralPath $VcVars)) { throw "MSVC environment script missing at $VcVars" }

# Fetch pinned CUTLASS into this repo's ignored build area.
$revision = '56ca28adb32ffadd097566545b8f7d83b652e998'
if (-not $CutlassDir) { $CutlassDir = Join-Path $repo 'build-deps\cutlass' }
if (-not (Test-Path -LiteralPath $CutlassDir)) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $CutlassDir) -Force | Out-Null
    & git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/cutlass.git $CutlassDir
    if ($LASTEXITCODE -ne 0) { throw 'CUTLASS clone failed' }
    & git -C $CutlassDir sparse-checkout init --cone
    if ($LASTEXITCODE -ne 0) { throw 'CUTLASS sparse checkout init failed' }
    & git -C $CutlassDir sparse-checkout set include tools/util
    if ($LASTEXITCODE -ne 0) { throw 'CUTLASS sparse checkout failed' }
    & git -C $CutlassDir checkout --detach $revision
    if ($LASTEXITCODE -ne 0) { throw 'CUTLASS checkout failed' }
}
$head = & git -C $CutlassDir rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $head -ne $revision) { throw "CUTLASS must be at $revision" }
$dirty = & git -C $CutlassDir status --porcelain
if ($LASTEXITCODE -ne 0 -or $dirty) { throw 'CUTLASS has local changes; it was not modified.' }

New-Item -ItemType Directory -Path $build -Force | Out-Null
$envScript = Join-Path $build 'compiler-env.cmd'
@("@call `"$VcVars`" >nul", '@if errorlevel 1 exit /b 1', '@set') | Set-Content -LiteralPath $envScript -Encoding ASCII
$environment = & $env:ComSpec /d /c $envScript
if ($LASTEXITCODE -ne 0) { throw 'vcvars64 failed' }
foreach ($line in $environment) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$env:PATH = "$CudaDir\bin;$env:PATH"
$dependencyArgs = @()
if ($GgmlDir) { $dependencyArgs += "-DSTRATA_GGML_DIR=$GgmlDir" }
$ErrorActionPreference = 'Continue' # PowerShell 5 treats native stderr warnings as ErrorRecords.
& $cmake -S $repo -B $build -G Ninja '-DCMAKE_BUILD_TYPE=Release' "-DCMAKE_MAKE_PROGRAM=$ninja" '-DSTRATA_ENABLE_CUDA=ON' '-DCMAKE_CUDA_ARCHITECTURES=120' "-DCMAKE_CUDA_COMPILER=$CudaDir\bin\nvcc.exe" '-DSTRATA_BUILD_TESTS=ON' '-DSTRATA_BUILD_CONVERSATION_TESTS=ON' '-DSTRATA_NVFP4_PREFILL=ON' '-DSTRATA_NVFP4_CUTLASS=ON' '-DSTRATA_BF16_KV_TC=ON' "-DSTRATA_CUTLASS_DIR=$CutlassDir" '-DCMAKE_CXX_FLAGS=/utf-8 /EHsc' "-DSTRATA_EXECUTABLE_NAME=$OutputName" @dependencyArgs
if ($LASTEXITCODE -ne 0) { throw 'configure failed' }
& $cmake --build $build --target $Targets --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'build failed' }
