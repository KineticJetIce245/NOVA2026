<#
.SYNOPSIS
    The two-command live demo, for Windows PowerShell.

.DESCRIPTION
    Wrapper around `documents/live_demo_two_commands.md`. It resolves its own
    location, finds this repository's Python **on this platform**, and runs one
    of the commands:

        .\live_demo.ps1 source      command 1: is the amplifier publishing what we need?
        .\live_demo.ps1 run         command 2: the session in front of the browser
        .\live_demo.ps1 calibrate   the audio-to-EEG loopback offset
        .\live_demo.ps1 check       list every LSL outlet on this network

    Anything after the step is passed straight through, so the manual command in
    the document works here too:

        .\live_demo.ps1 run --sfreq 500 --source-units uV --open-browser
        .\live_demo.ps1 calibrate --live --method cable
        .\live_demo.ps1 source --mode bridge

    THE INTERPRETER IS DETECTED, NEVER ASSUMED. A virtual environment keeps its
    interpreter at `.venv\Scripts\python.exe` on Windows and at `.venv/bin/python`
    on macOS and Linux; this script tries both, in that order, and then `python3`
    and `python` on PATH. A `.venv` cannot be carried between platforms - the
    interpreter and every compiled wheel are platform-specific - so if none of
    them exists, this refuses and says how to build one, rather than failing
    later with an import error.

    Shell setup, in THIS shell's own syntax: PowerShell sets the environment with
    `$env:NAME = 'value'`. The POSIX `NAME=value command` prefix is `sh`/`zsh`
    syntax and does not exist here, and `.\` separators do not work there; the
    `.sh` beside this file uses the other form. The two are deliberately not
    copies of each other.

    Refusals are loud and non-zero, never silent:
      2  the Python environment is missing or too old, or the amplifier is not
         publishing what the run asserts
      1  the command itself failed

    Python 3.12 or newer is required (pyproject.toml), on every platform.
<#
[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Step = "help",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest = @()
)

$ErrorActionPreference = "Stop"

# 1. Where this repository is, whatever directory the operator is standing in.
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = (Resolve-Path (Join-Path $Here "..\..")).Path
Set-Location $Repo

function Fail([string]$Message, [int]$Code = 2) {
    Write-Host ""
    Write-Host "REFUSED: $Message" -ForegroundColor Red
    exit $Code
}

# 2. The Python this repository was built with, detected rather than assumed.
#    An explicit ATTUNE_PYTHON wins, so an operator with a different environment
#    can point at it without editing this file.
$Candidates = @()
if ($env:ATTUNE_PYTHON) { $Candidates += $env:ATTUNE_PYTHON }
$Candidates += (Join-Path $Repo ".venv\Scripts\python.exe")   # Windows venv
$Candidates += (Join-Path $Repo ".venv/bin/python")           # macOS / Linux venv
$Candidates += (Join-Path $Repo ".venv\bin\python")           # the same, from Windows
$Candidates += "python3"
$Candidates += "python"

$Python = $null
foreach ($Candidate in $Candidates) {
    if ($Candidate -eq "python" -or $Candidate -eq "python3") {
        $found = Get-Command $Candidate -ErrorAction SilentlyContinue
        if ($found) { $Python = $found.Source; break }
        continue
    }
    if (Test-Path -LiteralPath $Candidate) { $Python = $Candidate; break }
}
if (-not $Python) {
    Fail @"
no Python interpreter found. Tried, in order:
  `$env:ATTUNE_PYTHON, .venv\Scripts\python.exe, .venv/bin/python, python3, python

A .venv cannot be copied between Windows and macOS: the interpreter and every
compiled wheel are platform-specific. Build the environment on THIS machine,
with Python 3.12 or newer:

  Windows        py -3.12 -m venv .venv ; .venv\Scripts\python.exe -m pip install -e .
  macOS/Linux    python3.12 -m venv .venv ; .venv/bin/python -m pip install -e .

(or set `$env:ATTUNE_PYTHON to an interpreter you already have)
"@
}

# 3. The interpreter must be able to import this package, and must be new enough.
#    macOS ships an older system Python, so an explicit version check says which
#    wall was hit instead of letting a syntax error three imports deep explain it.
& $Python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"
if ($LASTEXITCODE -ne 0) {
    $version = (& $Python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
    Fail "$Python is Python $version; this repository requires 3.12 or newer. Install one with 'brew install python@3.12', python.org, or 'uv python install 3.12', then recreate .venv."
}
& $Python -c "import nova2026" 2>$null
if ($LASTEXITCODE -ne 0) {
    Fail "$Python cannot import nova2026. Install the package into that environment:  & '$Python' -m pip install -e ."
}

# 4. Hand the environment to the child process the way THIS shell does it, and
#    put the repository on the import path so `-m scripts....` resolves from any
#    working directory. (`;` separates PYTHONPATH entries on Windows, `:` on
#    macOS and Linux - another thing the .sh cannot share.)
$env:ATTUNE_PYTHON = $Python
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$Repo;$env:PYTHONPATH" } else { $Repo }

function Invoke-Module([string]$Module, [string[]]$Arguments) {
    Write-Host "python  : $Python"
    Write-Host "module  : $Module $($Arguments -join ' ')"
    Write-Host ""
    & $Python -B -m $Module @Arguments
    $Code = $LASTEXITCODE
    if ($Code -eq 2) {
        Write-Host ""
        Write-Host ("That exit code means the amplifier was not found or did not " +
                    "publish the contract this run asserted. Tick 'Application " +
                    "options -> Network Operation -> Enable LSL EEG streaming' in " +
                    "the eego control software, confirm this host is on the " +
                    "amplifier's network, then run:  .\live_demo.ps1 check") -ForegroundColor Yellow
    }
    exit $Code
}

switch ($Step.ToLower()) {
    "source"    { Invoke-Module "scripts.getlive.live_source" $Rest }
    "1"         { Invoke-Module "scripts.getlive.live_source" $Rest }
    "run"       { Invoke-Module "scripts.auditory_ui.live" $Rest }
    "2"         { Invoke-Module "scripts.auditory_ui.live" $Rest }
    "calibrate" { Invoke-Module "scripts.getlive.calibrate_loopback" $Rest }
    "check"     { Invoke-Module "scripts.getlive.probe" (@("--all") + $Rest) }
    default {
        Get-Help $MyInvocation.MyCommand.Path -Detailed
        Write-Host "Steps: source | run | calibrate | check   (see documents/live_demo_two_commands.md)"
        exit 0
    }
}
