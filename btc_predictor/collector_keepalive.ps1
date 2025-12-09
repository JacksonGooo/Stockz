# Collector Keep-Alive Script
# Prevents sleep and auto-restarts the collector if it crashes
#
# Run as: powershell -ExecutionPolicy Bypass -File collector_keepalive.ps1
# Or create a scheduled task to run at startup

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CollectorPath = Join-Path $ScriptDir "end_of_day_collector.js"
$BackfillPath = Join-Path $ScriptDir "backfill_gaps.js"
$LogFile = Join-Path $ScriptDir "collector.log"

function Write-Log {
    param($Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage
    Add-Content -Path $LogFile -Value $logMessage
}

function Prevent-Sleep {
    # Prevent system from sleeping
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    $code = @"
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
"@
    $ste = Add-Type -MemberDefinition $code -Name "System" -Namespace "Win32" -PassThru
    # 0x80000000 = ES_CONTINUOUS, 0x00000001 = ES_SYSTEM_REQUIRED
    $ste::SetThreadExecutionState(0x80000001) | Out-Null
    Write-Log "Sleep prevention enabled"
}

function Allow-Sleep {
    $code = @"
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
"@
    $ste = Add-Type -MemberDefinition $code -Name "System2" -Namespace "Win32" -PassThru -ErrorAction SilentlyContinue
    if ($ste) {
        $ste::SetThreadExecutionState(0x80000000) | Out-Null
    }
    Write-Log "Sleep prevention disabled"
}

function Run-Backfill {
    Write-Log "Running backfill to recover any missing data..."
    try {
        $process = Start-Process -FilePath "node" -ArgumentList $BackfillPath -Wait -PassThru -NoNewWindow
        Write-Log "Backfill completed with exit code: $($process.ExitCode)"
    } catch {
        Write-Log "Backfill failed: $_"
    }
}

function Start-Collector {
    Write-Log "Starting data collector..."
    $process = Start-Process -FilePath "node" -ArgumentList $CollectorPath -PassThru -NoNewWindow
    return $process
}

Write-Log "=============================================="
Write-Log "S.U.P.I.D. Collector Keep-Alive Service"
Write-Log "=============================================="
Write-Log "Collector: $CollectorPath"
Write-Log "Log file: $LogFile"
Write-Log "=============================================="

# Prevent sleep
Prevent-Sleep

# Run backfill on startup to recover any gaps
Run-Backfill

# Start collector
$collectorProcess = Start-Collector

# Monitor loop
$restartCount = 0
$maxRestarts = 100  # Max restarts before giving up

try {
    while ($restartCount -lt $maxRestarts) {
        # Wait for process to exit
        $collectorProcess.WaitForExit()
        $exitCode = $collectorProcess.ExitCode

        Write-Log "Collector exited with code: $exitCode"

        if ($exitCode -eq 0) {
            # Clean exit (Ctrl+C or manual stop)
            Write-Log "Collector stopped cleanly"
            break
        }

        # Crash or error - restart after backfill
        $restartCount++
        Write-Log "Collector crashed! Restart attempt $restartCount of $maxRestarts"

        # Wait a bit before restart
        Start-Sleep -Seconds 5

        # Run backfill to recover any data from the crash gap
        Run-Backfill

        # Restart collector
        $collectorProcess = Start-Collector
    }
} finally {
    Allow-Sleep
    Write-Log "Keep-alive service stopped"
}
