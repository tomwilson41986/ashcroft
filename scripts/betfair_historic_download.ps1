<#
racing² - download Betfair Historic Data (BASIC plan) to a folder on this PC, one file at a time, with no clicking
through month by month on the website.

Run it on a UK computer (Betfair refuses logins from some countries, the US among them). It asks for the Betfair
username, password and application key each time and never saves them.

  1. On historicdata.betfair.com, signed in with the trading account, "buy" the free BASIC plan for the sport and the
     widest date range offered. The API lists only what the account holds.
  2. In PowerShell:
       Set-ExecutionPolicy -Scope Process Bypass
       .\betfair_historic_download.ps1 -Sport Soccer -From 2016-01 -To 2026-10 -OutDir "G:\My Drive\betfair_historic"
     (a Google Drive for desktop folder works as OutDir; so does any local folder.)
  3. At the end it packs each year into one file, <sport>_<year>.tar, in OutDir. Share those (Drive links) and they go
     in sources/betfair_imports.json; the pipeline imports each once and builds the closing-price tables.

Safe to stop and run again: a file already on disk is skipped. Newest month first. Read-only: nothing here bets.
#>
param(
    [string]$Sport = "Soccer",
    [string]$From = "2016-01",
    [string]$To = (Get-Date).AddMonths(-1).ToString("yyyy-MM"),
    [string]$OutDir = "$HOME\betfair_historic",
    [string[]]$MarketTypes = @("MATCH_ODDS", "OVER_UNDER_25", "ASIAN_HANDICAP"),
    [string[]]$Countries = @(),          # empty: every country
    [string]$Plan = "Basic Plan",
    [switch]$NoTar
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is many times slower with the progress bar on
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Api = "https://historicdata.betfair.com/api"

$cred = Get-Credential -Message "Betfair login (username and password)"
$keySecure = Read-Host "Betfair application key" -AsSecureString
$appKey = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($keySecure))

function Get-Token {
    $body = @{ username = $cred.UserName; password = $cred.GetNetworkCredential().Password }
    $r = Invoke-RestMethod -Method Post -Uri "https://identitysso.betfair.com/api/login" -Body $body `
        -Headers @{ "X-Application" = $appKey; "Accept" = "application/json" }
    if ($r.status -ne "SUCCESS") { throw "Betfair login refused: $($r.error)" }
    return $r.token
}

$script:token = Get-Token
Write-Host "Logged in."

function Invoke-Historic([string]$Method, $Body) {
    for ($try = 0; $try -lt 2; $try++) {
        try {
            return Invoke-RestMethod -Method Post -Uri "$Api/$Method" -ContentType "application/json" `
                -Headers @{ ssoid = $script:token } -Body ($Body | ConvertTo-Json -Depth 5) -TimeoutSec 300
        } catch {
            if ($try -eq 0) { Write-Host "  $Method failed ($($_.Exception.Message)); logging in again"; $script:token = Get-Token }
            else { throw }
        }
    }
}

# what the account holds
$mine = Invoke-Historic "GetMyData" @{}
Write-Host "The account holds:"
$mine | Format-Table -AutoSize | Out-String | Write-Host
if (-not ($mine | Where-Object { $_.sport -eq $Sport -and $_.plan -eq $Plan })) {
    throw "The account does not hold '$Sport' on '$Plan'. Get it (free) on historicdata.betfair.com first."
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$start = [datetime]::ParseExact($From, "yyyy-MM", $null)
$month = [datetime]::ParseExact($To, "yyyy-MM", $null)
$total = 0; $got = 0; $failed = 0
$years = @{}

while ($month -ge $start) {
    $last = $month.AddMonths(1).AddDays(-1)
    $body = @{
        sport = $Sport; plan = $Plan
        fromDay = 1; fromMonth = $month.Month; fromYear = $month.Year
        toDay = $last.Day; toMonth = $last.Month; toYear = $last.Year
        eventId = $null; eventName = $null
        marketTypesCollection = $MarketTypes; countriesCollection = $Countries; fileTypeCollection = @("M")
    }
    $files = @(Invoke-Historic "DownloadListOfFiles" $body)
    $new = 0
    foreach ($p in $files) {
        # keep the path from the plan on (BASIC/2024/Mar/2/...): the pipeline names a file by it
        $i = $p.IndexOf("BASIC/")
        $rel = if ($i -ge 0) { $p.Substring($i) } else { $p.TrimStart("/") }
        $dest = Join-Path $OutDir ($rel -replace "/", "\")
        if (Test-Path $dest) { continue }
        New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
        $url = "$Api/DownloadFile?filePath=" + [Uri]::EscapeDataString($p)
        try {
            Invoke-WebRequest -Uri $url -Headers @{ ssoid = $script:token } -OutFile "$dest.part" -TimeoutSec 300
            Move-Item -Force "$dest.part" $dest
            $new++
        } catch {
            Remove-Item -ErrorAction SilentlyContinue "$dest.part"
            $script:token = Get-Token
            try {
                Invoke-WebRequest -Uri $url -Headers @{ ssoid = $script:token } -OutFile "$dest.part" -TimeoutSec 300
                Move-Item -Force "$dest.part" $dest
                $new++
            } catch { $failed++; Remove-Item -ErrorAction SilentlyContinue "$dest.part" }
        }
    }
    $total += $files.Count; $got += $new
    if ($files.Count) { $years[$month.Year] = $true }
    Write-Host ("{0}: {1} files listed, {2} downloaded (running total {3} new, {4} failed)" -f `
        $month.ToString("yyyy-MM"), $files.Count, $new, $got, $failed)
    $month = $month.AddMonths(-1)
}

Write-Host "Done: $total files listed, $got downloaded, $failed failed (run again to retry the failures)."

if (-not $NoTar) {
    foreach ($y in ($years.Keys | Sort-Object)) {
        $tarName = "{0}_{1}.tar" -f $Sport.ToLower().Replace(" ", "_"), $y
        if (Test-Path (Join-Path $OutDir "BASIC\$y")) {
            tar.exe -cf (Join-Path $OutDir $tarName) -C $OutDir "BASIC/$y"
            Write-Host "Packed $tarName"
        }
    }
}
