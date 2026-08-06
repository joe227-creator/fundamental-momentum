param(
    [string]$AsOf,
    [string]$CurrentPositions = "positions.csv",
    [string]$OutputDir = "artifacts/live",
    [switch]$RefreshData
)

# Champion live signal runner.
# Runs the champion strategy screen with TimesFM volume veto.
# Scheduled Mon-Sat at 21:00 SGT (= 08:00/09:00 ET, before the US open). The
# calendar guard resolves the last *completed* NYSE session in ET, so a 21:00
# SGT trigger evaluates the previous day's finished US session, never the
# not-yet-traded current ET date, regardless of SGT/ET date drift. A
# state file (artifacts/live/last_rebalance_processed.txt) records each
# processed month so a missed first-trading-day run is caught up on any later
# day of that month instead of being silently skipped.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

function Resolve-Python {
    $candidates = @(
        (Join-Path $repoRoot ".venv\Scripts\python.exe"),
        $env:ALGO_TRADING_PYTHON,
        "C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe"
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $command -and -not [string]::IsNullOrWhiteSpace($command.Source)) {
        return $command.Source
    }

    throw "Python interpreter not found. Checked the repo venv, ALGO_TRADING_PYTHON, C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe, and PATH."
}

# Resolve paths
$logPath = Join-Path $repoRoot "artifacts\live\champion_run.log"
$logDir = Split-Path -Parent $logPath
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}
$stateFile = Join-Path $logDir "last_rebalance_processed.txt"
$lockPath = Join-Path $logDir ".run.lock"

# UTF-8 (no BOM) encoder. Pinning the encoding explicitly makes the log
# identical under Windows PowerShell 5.1 (powershell.exe, used by Task
# Scheduler) and PowerShell 7 (pwsh). Without this, Tee-Object defaulted to
# UTF-16 LE under PS 5.1 and UTF-8 under PS 7, which produced a garbled
# mixed-encoding file when the same log was appended across hosts. PS 5.1's
# Tee-Object has no -Encoding parameter, so we write via .NET AppendAllText
# with a no-BOM UTF8Encoding and echo to stdout via Write-Output.
$script:Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Write-RunLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$stamp] $Message"
    # Write-Host (not Write-Output): the echo is a console-only user message. The
    # durable record is the AppendAllText file write. Write-Output would pollute
    # the output stream of any capturing caller (e.g. Invoke-PythonLogged's
    # return value), which corrupted $calendarExitCode into [log-line, 0].
    Write-Host $line
    [System.IO.File]::AppendAllText($logPath, "$line`r`n", $script:Utf8NoBom)
}

# Run a python command, logging every stdout/stderr line to the run log.
# Scoped $ErrorActionPreference="Continue": the script runs under "Stop" (so
# its own cmdlet errors terminate), but Stop ALSO promotes ANY native-command
# stderr to a terminating NativeCommandError -- which silently aborted runs on
# the first python warning/yfinance notice (survivorship-bias UserWarning,
# "$HEIA: possibly delisted"). Continue makes that stderr non-terminating so the
# 2>&1 merge yields a clean interleaved log. The function-scope setting shadows
# the script's only for this call. Returns the python exit code.
function Invoke-PythonLogged {
    param(
        [Parameter(Mandatory)][string]$Interpreter,
        [Parameter(Mandatory)][string[]]$Arguments
    )
    $ErrorActionPreference = "Continue"
    & $Interpreter @Arguments 2>&1 | ForEach-Object {
        # Under 2>&1, native stderr becomes ErrorRecord objects; stringifying one
        # can yield its type name (e.g. "System.Management.Automation.RemoteException")
        # rather than the message. Extract the real message for ErrorRecords.
        $msg = if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { $_ }
        Write-RunLog $msg
    }
    return $LASTEXITCODE
}

# As Invoke-PythonLogged but also returns the captured stdout/stderr text so the
# caller can parse structured output (e.g. the guard's SESSION= line).
function Invoke-PythonCapture {
    param(
        [Parameter(Mandatory)][string]$Interpreter,
        [Parameter(Mandatory)][string[]]$Arguments
    )
    $ErrorActionPreference = "Continue"
    $lines = New-Object System.Collections.Generic.List[string]
    & $Interpreter @Arguments 2>&1 | ForEach-Object {
        $msg = if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { $_ }
        Write-RunLog $msg
        $lines.Add("$msg")
    }
    return @{ ExitCode = $LASTEXITCODE; Output = ($lines -join "`n") }
}

# Single-instance lock: prevents a manual run and a scheduled run (or two
# scheduled triggers) from racing on the GPU forecast + screen. A stale lock
# whose PID is no longer alive is reclaimed.
$lockAcquired = $false
if (Test-Path $lockPath) {
    $existingPid = (Get-Content $lockPath -Raw -ErrorAction SilentlyContinue).Trim()
    $alive = $false
    if ($existingPid -match '^\d+$') {
        if (Get-Process -Id ([int]$existingPid) -ErrorAction SilentlyContinue) { $alive = $true }
    }
    if ($alive) {
        Write-RunLog "Another run is in progress (PID $existingPid). Exiting."
        exit 0
    }
    Write-RunLog "Stale lock found (PID $existingPid not running). Reclaiming."
    Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
}
"$PID" | Out-File -FilePath $lockPath -Encoding ascii -NoNewline
$lockAcquired = $true

if ($AsOf) {
    Write-RunLog "Champion live signal check started for $AsOf."
} else {
    Write-RunLog "Champion live signal check started (auto-detect last completed NYSE session)."
}

try {
    try {
        $python = Resolve-Python
    } catch {
        Write-RunLog "Failed to resolve Python interpreter: $_"
        exit 1
    }

    # Calendar guard: exit 0 only on the first NYSE trading day of the month.
    # Prints SESSION=YYYY-MM-DD (the candidate session) for the launcher to use.
    $calendarScript = Join-Path $PSScriptRoot "is_first_nyse_rebalance_session.py"
    if (-not (Test-Path $calendarScript)) {
        Write-RunLog "Calendar guard not found: $calendarScript — proceeding without guard."
        $sessionDate = $AsOf
        $calendarExitCode = 0
    } else {
        $guardArgs = @($calendarScript)
        if ($AsOf) { $guardArgs += $AsOf }
        $guardResult = Invoke-PythonCapture -Interpreter $python -Arguments $guardArgs
        $calendarExitCode = $guardResult.ExitCode
        $sessionMatch = [regex]::Match($guardResult.Output, "SESSION=(\d{4}-\d{2}-\d{2})")
        if ($sessionMatch.Success) {
            $sessionDate = $sessionMatch.Groups[1].Value
        } else {
            $sessionDate = $AsOf
        }
    }

    if ($calendarExitCode -eq 2) {
        Write-RunLog "Not the first NYSE trading day of the month (session=$sessionDate). No action needed."
        exit 0
    }
    if ($calendarExitCode -ne 0) {
        Write-RunLog "Calendar check failed (exit $calendarExitCode). Aborting."
        exit 1
    }

    # Catch-up: if this month's rebalance was already screened successfully, skip.
    # This lets any later day of the month pick up a missed/failed first-day run.
    $processedMonth = ""
    if (Test-Path $stateFile) { $processedMonth = (Get-Content $stateFile -Raw -ErrorAction SilentlyContinue).Trim() }
    $sessionMonth = if ($sessionDate) { $sessionDate.Substring(0, 7) } else { "" }
    if ($sessionMonth -and $processedMonth -eq $sessionMonth) {
        Write-RunLog "Month $sessionMonth already processed (state file). No action needed."
        exit 0
    }
    Write-RunLog "First NYSE trading day of the month: $sessionDate. Proceeding with screen."

    # Run the champion live screen
    $arguments = @("run_champion.py", "live", "--as-of", $sessionDate)

    if ($RefreshData) {
        $arguments += "--refresh-data"
    }

    $resolvedPositions = Join-Path $repoRoot $CurrentPositions
    if (Test-Path $resolvedPositions) {
        $arguments += @("--current-positions", $resolvedPositions)
    }

    $env:PYTHONPATH = Join-Path $repoRoot "src"
    $env:TRANSFORMERS_OFFLINE = "1"
    $env:OMP_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS = "1"

    Push-Location $repoRoot
    try {
        # Refresh TimesFM P20 forecasts for any new S&P 500 symbols/dates (auto-
        # updates with the Wikipedia holdings refresh). generate_spy_forecasts.py
        # is idempotent: it loads the (auto-refreshed) holdings, finds (symbol,
        # date) pairs missing from the forecast cache, and forecasts only those on
        # GPU. Exits early (no model load) if the cache already covers the
        # universe. Failure is non-fatal: the live screen still runs (veto off +
        # warning for new symbols).
        Write-RunLog "Refreshing TimesFM forecasts (incremental; skips if cache covers universe)..."
        $fcCode = Invoke-PythonLogged -Interpreter $python -Arguments @("generate_spy_forecasts.py", "--quantiles")
        if ($fcCode -ne 0) {
            Write-RunLog "Forecast refresh failed (exit $fcCode). Continuing to live screen (veto may be off for new symbols)."
        } else {
            Write-RunLog "Forecast refresh completed."
        }

        Write-RunLog "Running: python $arguments"
        $liveCode = Invoke-PythonLogged -Interpreter $python -Arguments $arguments
        if ($liveCode -ne 0) {
            Write-RunLog "Champion live screen failed (exit $liveCode)."
            exit 1
        }
        # Record the processed month ONLY after a successful screen so a failed
        # run is retried on the next scheduled trigger (catch-up).
        if ($sessionMonth) {
            $sessionMonth | Out-File -FilePath $stateFile -Encoding ascii -NoNewline
            Write-RunLog "Recorded processed month: $sessionMonth"
        }
        Write-RunLog "Champion live screen completed successfully."
    } finally {
        Pop-Location
    }
} catch {
    Write-RunLog "Unhandled error: $_"
    exit 1
} finally {
    if ($lockAcquired) { Remove-Item $lockPath -Force -ErrorAction SilentlyContinue }
    Write-RunLog "Run ended."
}
