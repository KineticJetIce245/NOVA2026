# Start the auditory demo server and hand the user the URL it printed.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\auditory_ui\serve.ps1
#
# It runs the repository's own entry point -- scripts.auditory_ui.demo -- with the
# built frontend, the fitted decoder and a real KU Leuven trial, opens the page in
# the default browser, and prints the loopback URL. The demo binds an ephemeral
# port, so the URL is read back from the demo's own output rather than assumed.
#
# Nothing here reimplements the demo: this is a flag-remembering wrapper, plus the
# one thing the demo needs around it today:
#
#   * the frontend build (`apps/attune-ui/dist`) is git-ignored and is what the
#     demo serves; `-Rebuild` refreshes it from `src/` before starting, so the page
#     shows the current sources instead of whatever was built last.
#
# The demo's own stay-open window does the rest: `-ServeSeconds` (12 h by default)
# is forwarded as `--serve-seconds`, and the demo keeps answering on the URL it
# printed for that whole window. `-KeepAlive` restores the older behaviour of
# restarting the demo and printing a new URL; see the note at the bottom.

[CmdletBinding()]
param(
    # The converted trial to replay. trial_004 is label class 1 (attend B), so the
    # dashboard shows a real A/B decision instead of only the majority answer.
    [string]$Trial = 'datasets/AAD-KULeuven/converted/S1/trial_004.npz',

    # Fitted decoder.
    [string]$Model = 'models/auditory_kuleuven.npz',

    # Replay only this many seconds; 0 replays the whole trial (trial_004: 389 s).
    [double]$Seconds = 0,

    # How long the demo keeps serving after the replay ends: forwarded as
    # `--serve-seconds`, and the demo really serves it (commit d7f5375). 12 h is the
    # "leave a page up" default; 0 turns the stay-open window off.
    [double]$ServeSeconds = 43200,

    # 0 means "the calibrated default" (0.05), per the demo's own --margin help.
    [double]$Margin = 0,

    # Rebuild apps/attune-ui/dist from src/ first, so the page is current.
    [switch]$Rebuild,

    # Do not ask the OS to open a browser tab (used by the verification smoke run).
    [switch]$NoBrowser,

    # Restart the demo when it exits and print the new URL, instead of letting the
    # one demo serve out its own -ServeSeconds window. Only useful with a bounded
    # -ServeSeconds; see the note at the bottom of this file.
    [switch]$KeepAlive,

    # Run once with short, scratch output names and exit; used to verify this file.
    [switch]$Smoke,

    # How long to wait for the demo to print its URL on each start.
    [int]$UrlTimeoutSeconds = 90
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# This script sits in <repo>/scripts/auditory_ui/, so the repository root is two
# directories up. Everything below is resolved against that, not the caller's cwd.
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "no virtualenv interpreter at $python (run from a checkout with .venv)"
}

$trialPath = Join-Path $repo $Trial
if (-not (Test-Path -LiteralPath $trialPath)) {
    throw "trial not found: $trialPath"
}
$modelPath = Join-Path $repo $Model
if (-not (Test-Path -LiteralPath $modelPath)) {
    throw "decoder not found: $modelPath"
}

$outDir = Join-Path $repo 'output\auditory_ui'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

if ($Rebuild) {
    $ui = Join-Path $repo 'apps\attune-ui'
    Write-Host "rebuilding the frontend: npm run build (in $ui)"
    $npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue)
    if (-not $npm) { throw "npm not found; cannot rebuild apps/attune-ui/dist" }
    # npm writes warnings to stderr; under $ErrorActionPreference='Stop' those
    # lines would abort the launcher, so they are demoted to plain output here.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    Push-Location -LiteralPath $ui
    try {
        & $npm.Source 'run' 'build' 2>&1 | ForEach-Object { Write-Host "  build| $_" }
        $buildCode = $LASTEXITCODE
    } finally {
        Pop-Location
        $ErrorActionPreference = $previous
    }
    if ($buildCode -ne 0) { throw "the frontend build failed (npm run build -> $buildCode)" }
    $distIndex = Join-Path $ui 'dist\index.html'
    if (-not (Test-Path -LiteralPath $distIndex)) { throw "the build produced no $distIndex" }
    Write-Host ("  build| dist/index.html written " + (Get-Item -LiteralPath $distIndex).LastWriteTime)
}

function New-DemoArgs([string]$out, [string]$streamOut) {
    # `--media-out` is always explicit, and is never the demo's own default
    # (output/auditory_ui/demo_stereo.wav). That name belongs to whichever run
    # claimed it first: the page reads `duration` from the file, so a second run
    # rendering a different trial over it silently re-tunes the session already
    # being served. One media file per run, named after the run record, so -Smoke
    # gets serve_smoke.wav and a normal start gets serve_<stamp>.wav.
    $mediaOut = "output/auditory_ui/$([System.IO.Path]::GetFileNameWithoutExtension($out)).wav"
    $demoArgs = @(
        '-B', '-m', 'scripts.auditory_ui.demo',
        '--trial', $Trial,
        '--model', $Model,
        '--out', $out,
        '--stream-out', $streamOut,
        '--media-out', $mediaOut
    )
    if ($Seconds -gt 0) { $demoArgs += @('--seconds', [string]$Seconds) }
    if ($Margin -gt 0) { $demoArgs += @('--margin', [string]$Margin) }
    if ($ServeSeconds -gt 0 -and -not $Smoke) { $demoArgs += @('--serve-seconds', [string]$ServeSeconds) }
    if (-not $NoBrowser) { $demoArgs += '--open-browser' }
    return $demoArgs
}

function Start-Demo([string]$out, [string]$streamOut) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $logBase = [System.IO.Path]::GetFileNameWithoutExtension($out)
    $stdoutLog = Join-Path $outDir "$logBase.stdout.log"
    $stderrLog = Join-Path $outDir "$logBase.stderr.log"
    $demoArgs = New-DemoArgs $out $streamOut

    Write-Host ""
    Write-Host "command  : .venv\Scripts\python.exe $($demoArgs -join ' ')"
    Write-Host "stdout   : $stdoutLog"

    $proc = Start-Process -FilePath $python -ArgumentList $demoArgs -WorkingDirectory $repo `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
    return [pscustomobject]@{
        Proc      = $proc
        StdoutLog = $stdoutLog
        StderrLog = $stderrLog
        StartedAt = $stamp
    }
}

function Get-DemoUrl([string]$text) {
    # The demo prints "serving the demo at http://127.0.0.1:<port>/" before the
    # replay starts, and "open: http://127.0.0.1:<port>/" in its final report.
    foreach ($line in ($text -split "`r?`n")) {
        $m = [regex]::Match($line, 'http://127\.0\.0\.1:(\d+)/')
        if ($m.Success) { return $m.Value }
    }
    return $null
}

function Wait-ForUrl($run) {
    $waited = 0
    while ($waited -lt $UrlTimeoutSeconds) {
        $text = $null
        if (Test-Path -LiteralPath $run.StdoutLog) {
            $text = Get-Content -LiteralPath $run.StdoutLog -Raw -ErrorAction SilentlyContinue
        }
        $url = if ($text) { Get-DemoUrl $text } else { $null }
        if ($url) { return $url }
        if ($run.Proc.HasExited) { return $null }
        Start-Sleep -Milliseconds 500
        $waited += 0.5
    }
    return $null
}

function Show-Failure($run) {
    Write-Host "the demo did not print a URL within ${UrlTimeoutSeconds}s."
    if ($run.Proc.HasExited) { Write-Host "it exited with code $($run.Proc.ExitCode)." }
    if (Test-Path -LiteralPath $run.StderrLog) {
        $tail = Get-Content -LiteralPath $run.StderrLog -Tail 25
        if ($tail) { Write-Host "--- stderr tail ---"; $tail | ForEach-Object { Write-Host $_ } }
    }
    if (Test-Path -LiteralPath $run.StdoutLog) {
        Write-Host "--- stdout tail ---"
        Get-Content -LiteralPath $run.StdoutLog -Tail 25 | ForEach-Object { Write-Host $_ }
    }
}

# ---------------------------------------------------------------- start once ----
if ($Smoke) {
    $run = Start-Demo 'output/auditory_ui/serve_smoke.json' 'output/auditory_ui/serve_smoke_packets.jsonl'
} else {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $run = Start-Demo "output/auditory_ui/serve_$stamp.json" "output/auditory_ui/serve_${stamp}_packets.jsonl"
}

$url = Wait-ForUrl $run
if (-not $url) { Show-Failure $run; exit 1 }

Write-Host "================================================================"
Write-Host "  demo server is up:  $url"
Write-Host "================================================================"
Write-Host "pid       : $($run.Proc.Id)"
Write-Host "replay    : $(if ($Seconds -gt 0) { "$Seconds s" } else { 'the whole trial' })"
Write-Host "stay-open : $(if ($ServeSeconds -gt 0) { "$ServeSeconds s after the replay" } else { 'none: the demo exits with the replay' })"
Write-Host "stop      : Stop-Process -Id $($run.Proc.Id)"
Write-Host "transcript: Get-Content '$($run.StdoutLog)' -Wait"
Write-Host ""

if ($Smoke) {
    Write-Host "Started once (-Smoke): the server exists this process tree only."
    exit 0
}

# ------------------------------------------------- wait for the demo to end ----
# The default path, and all the "keep a page up" this launcher needs: the demo
# itself serves `--serve-seconds` on the port it printed, so there is nothing to
# restart. This waits for that window (or for the demo to fail) and reports how it
# ended. Ctrl-C here stops *this* launcher; the demo runs in its own hidden
# console, so end it with the `stop:` command printed above.
if (-not $KeepAlive) {
    Write-Host "The demo serves this URL on its own for its stay-open window; waiting for it"
    Write-Host "to exit. No restart is needed and no second URL will be printed."
    Write-Host ""
    try { $run.Proc.WaitForExit() } catch { }
    $exitCode = 'unknown'
    try { $exitCode = $run.Proc.ExitCode } catch { }
    Write-Host ("[{0}] the demo exited (code {1})." -f (Get-Date -Format 'HH:mm:ss'), $exitCode)
    if ($exitCode -is [int]) { exit $exitCode }
    exit 0
}

# ------------------------------------------------- -KeepAlive: start again ----
# Opt-in. A restart prints a NEW URL and starts a fresh replay from t=0, while the
# URL printed above stops being served: anyone watching that tab gets a dead page.
# That is only wanted when the stay-open window is deliberately short.
Write-Host "Keeping a page up by restarting the demo (-KeepAlive): each restart prints a"
Write-Host "new URL and replays from the beginning, and the previous URL stops being"
Write-Host "served. Ctrl-C stops this launcher loop; the hidden demo keeps running, so"
Write-Host "end it with Stop-Process -Id <pid>."
Write-Host ""

while ($true) {
    try {
        $run.Proc.WaitForExit()
    } catch {
        # the process handle can vanish; fall through to the liveness check
    }
    $exitCode = 'unknown'
    try { $exitCode = $run.Proc.ExitCode } catch { }
    Write-Host ""
    Write-Host ("[{0}] the demo exited (code {1}); the page is no longer served." -f (Get-Date -Format 'HH:mm:ss'), $exitCode)
    Write-Host ("[{0}] transcript: {1}" -f (Get-Date -Format 'HH:mm:ss'), $run.StdoutLog)

    Start-Sleep -Seconds 2
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $run = Start-Demo "output/auditory_ui/serve_$stamp.json" "output/auditory_ui/serve_${stamp}_packets.jsonl"
    $url = Wait-ForUrl $run
    if (-not $url) { Show-Failure $run; exit 1 }
    Write-Host ("[{0}] restarted: {1}" -f (Get-Date -Format 'HH:mm:ss'), $url)
}

# Why -KeepAlive is opt-in, not the default (for the demo's maintainer, not for the
# operator): the restart loop existed because `drive()` ended with an unconditional
# `stop.set()`, while the stay-open loop that `--serve-seconds` and `--browser` rely
# on is `while not stop.is_set()` -- so the loop saw an already-set event, broke on
# its first check, and the transport shut down in the second the replay ended.
# "replay finished; still serving ..." was printed and then nothing was served.
#
# That is fixed (commit d7f5375, "auditory_ui: keep serving after the replay ends,
# instead of dying with it"): `drive()` now ends its own phase on a separate
# `finished` event, and only the Ctrl-C/SIGTERM path (and the end of the stay-open
# window) sets `stop`. Locked by scripts/auditory/tests/test_demo_serve.py -- two
# tests that fail in opposite directions, so neither "the replay kills the server"
# nor "nothing ever stops" passes -- and by scripts/auditory/tests/mutate_demo_serve.py,
# which restores the defect and shows which test goes red (2/2 caught).
#
# With the fix in, restarting is no longer the only way to leave a page up; it is a
# way to replace a working page with a new URL. Hence a switch. Delete it once
# nothing needs a fresh replay after a bounded window.
