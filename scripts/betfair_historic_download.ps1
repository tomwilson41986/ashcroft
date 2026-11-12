<#
racing² - download Betfair Historic Data (BASIC plan) to a folder on this PC, one file at a time.

Run it on a UK computer. It asks for the Betfair username, password and application key each time and never saves
them. Safe to stop and run again: a file already on disk is skipped. Newest month first. Read-only: nothing here bets.
A dropped connection is waited out (about 50 minutes for the login, six tries for each list and file); a login Betfair
refuses stops it at once. At the end it packs each year into <sport>_<year>.tar in OutDir.

  Set-ExecutionPolicy -Scope Process Bypass
  .\betfair_historic_download.ps1 -Sport Soccer -From 2016-01 -To 2026-10 -OutDir "G:\My Drive\betfair_historic"

Share the yearly .tar files (Drive links); they go in sources/betfair_imports.json and the pipeline imports each once.
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

Write-Host "Betfair Historic Data download: $Sport, $From to $To, into $OutDir"
$user = Read-Host "Betfair username"
$pass = Read-Host "Betfair password" -AsSecureString
$cred = New-Object System.Management.Automation.PSCredential($user, $pass)
$keySecure = Read-Host "Betfair application key" -AsSecureString
$appKey = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($keySecure))

function Wait-Retry([int]$Try, [string]$What, $Err) {
    $wait = [Math]::Min(300, 15 * $Try)
    Write-Host ("  {0} failed ({1}); trying again in {2} s" -f $What, $Err.Exception.Message, $wait)
    Start-Sleep -Seconds $wait
}

function Test-AuthError($Err) {
    $resp = $Err.Exception.Response
    # 400 too: after a few hours the service answers an expired session with 400 Bad Request (the owner's run, 10 Nov)
    return ($null -ne $resp) -and ([int]$resp.StatusCode -in 400, 401, 403)
}

function Get-Token {
    for ($try = 1; $try -le 20; $try++) {
        try {
            $body = @{ username = $cred.UserName; password = $cred.GetNetworkCredential().Password }
            $r = Invoke-RestMethod -Method Post -Uri "https://identitysso.betfair.com/api/login" -Body $body `
                -Headers @{ "X-Application" = $appKey; "Accept" = "application/json" } -TimeoutSec 60
        } catch { Wait-Retry $try "login" $_; continue }
        if ($r.status -ne "SUCCESS") { throw "Betfair login refused: $($r.error)" }
        return $r.token
    }
    throw "Could not reach Betfair to log in. Check the internet connection, then run the script again: it carries on where it stopped."
}

function Invoke-Historic([string]$Method, $Body) {
    for ($try = 1; $try -le 6; $try++) {
        try {
            return Invoke-RestMethod -Method Post -Uri "$Api/$Method" -ContentType "application/json" `
                -Headers @{ ssoid = $script:token } -Body ($Body | ConvertTo-Json -Depth 5) -TimeoutSec 300
        } catch {
            if (Test-AuthError $_) { $script:token = Get-Token } else { Wait-Retry $try $Method $_ }
        }
    }
    throw "$Method failed 6 times"
}

function Get-ServerMessage($Err) {
    $m = $Err.ErrorDetails.Message
    if (-not $m) { $m = $Err.Exception.Message }
    if ($m.Length -gt 300) { $m = $m.Substring(0, 300) }
    return $m
}

function Save-File([string]$Url, [string]$Dest, [string]$Path) {
    # a 400 (or 401/403) is answered with a fresh login and one more try; if Betfair refuses the file again it is
    # skipped at once with the server's own message (no long waits); a dropped connection is waited out
    $relogged = $false
    for ($try = 1; $try -le 6; $try++) {
        try {
            Invoke-WebRequest -Uri $Url -Headers @{ ssoid = $script:token } -OutFile "$Dest.part" -TimeoutSec 300
            Move-Item -Force "$Dest.part" $Dest
            return $true
        } catch {
            Remove-Item -ErrorAction SilentlyContinue "$Dest.part"
            if (Test-AuthError $_) {
                if ($relogged) {
                    Write-Host ("  refused: {0}`n    server says: {1}" -f $Path, (Get-ServerMessage $_))
                    return $false
                }
                $script:token = Get-Token; $relogged = $true
            } else { Wait-Retry $try "download" $_ }
        }
    }
    return $false
}

$script:token = Get-Token
Write-Host "Logged in."

# what the account holds (a GET: the service refuses a POST here with 405)
$mine = $null; $ok = $false
for ($try = 1; $try -le 6 -and -not $ok; $try++) {
    try {
        $mine = Invoke-RestMethod -Method Get -Uri "$Api/GetMyData" -Headers @{ ssoid = $script:token } -TimeoutSec 300
        $ok = $true
    } catch {
        if (Test-AuthError $_) { $script:token = Get-Token } else { Wait-Retry $try "GetMyData" $_ }
    }
}
if (-not $ok) { throw "Could not read what the account holds; run the script again." }
Write-Host "The account holds:"
$mine | Group-Object sport, plan | ForEach-Object { Write-Host ("  {0}: {1} months" -f $_.Name, $_.Count) }
$held = @($mine | Where-Object { $_.sport -eq $Sport -and $_.plan -eq $Plan })
if (-not $held) {
    throw "The account does not hold '$Sport' on '$Plan'. Get it (free) on historicdata.betfair.com first."
}
$heldMonths = @{}
foreach ($h in $held) { if ($h.forDate) { $heldMonths[([datetime]$h.forDate).ToString("yyyy-MM")] = $true } }
Write-Host ("Months held for {0}: {1}" -f $Sport, $heldMonths.Count)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$start = [datetime]::ParseExact($From, "yyyy-MM", $null)
$month = [datetime]::ParseExact($To, "yyyy-MM", $null)
$total = 0; $got = 0; $failed = 0
$years = @{}
$unlisted = @()
$script:inRow = 0

while ($month -ge $start) {
    $last = $month.AddMonths(1).AddDays(-1)
    $key = $month.ToString("yyyy-MM")
    if ($heldMonths.Count -and -not $heldMonths.ContainsKey($key)) {
        Write-Host "${key}: not held by the account (buy it on the website to include it); skipped"
        $month = $month.AddMonths(-1)
        continue
    }
    $body = @{
        sport = $Sport; plan = $Plan
        fromDay = 1; fromMonth = $month.Month; fromYear = $month.Year
        toDay = $last.Day; toMonth = $last.Month; toYear = $last.Year
        eventId = $null; eventName = $null
        marketTypesCollection = $MarketTypes; countriesCollection = $Countries; fileTypeCollection = @("M")
    }
    try {
        # piped so the list is unrolled: Windows PowerShell 5.1 returns a JSON array as one object
        $files = @((Invoke-Historic "DownloadListOfFiles" $body) | ForEach-Object { $_ } | ForEach-Object { [string]$_ })
    } catch {
        Write-Host "${key}: the file list could not be read ($($_.Exception.Message)); skipped for now"
        $unlisted += $key
        $month = $month.AddMonths(-1)
        continue
    }
    Write-Host "${key}: $($files.Count) files listed, downloading..."
    $new = 0; $seen = 0
    foreach ($p in $files) {
        $seen++
        if ($seen % 500 -eq 0) { Write-Host "  ${key}: $seen of $($files.Count) done" }
        # keep the path from the plan on (BASIC/2024/Mar/2/...): the pipeline names a file by it
        $i = $p.IndexOf("BASIC/")
        $rel = if ($i -ge 0) { $p.Substring($i) } else { $p.TrimStart("/") }
        $dest = Join-Path $OutDir ($rel -replace "/", "\")
        if (Test-Path $dest) { continue }
        New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
        $url = "$Api/DownloadFile?filePath=" + [Uri]::EscapeDataString($p)
        if (Save-File $url $dest $p) { $new++; $script:inRow = 0 } else {
            $failed++; $script:inRow++
            if ($script:inRow -ge 20) {
                throw ("20 files in a row refused ({0} failed so far). Betfair may be limiting downloads: wait an hour or " +
                       "two and run the same command again (files on disk are skipped). Send a screenshot of the " +
                       "'server says' lines above." -f $failed)
            }
        }
    }
    $total += $files.Count; $got += $new
    if ($files.Count) { $years[$month.Year] = $true }
    Write-Host ("{0}: {1} files listed, {2} downloaded (running total {3} new, {4} failed)" -f `
        $key, $files.Count, $new, $got, $failed)
    $month = $month.AddMonths(-1)
}

Write-Host "Done: $total files listed, $got downloaded, $failed failed."
if ($unlisted.Count) { Write-Host ("Months to fill on a rerun (list not read): {0}" -f ($unlisted -join ", ")) }
if ($failed -or $unlisted.Count) { Write-Host "Run the same command again to fill them: files already on disk are skipped." }

if (-not $NoTar) {
    foreach ($y in ($years.Keys | Sort-Object)) {
        $tarName = "{0}_{1}.tar" -f $Sport.ToLower().Replace(" ", "_"), $y
        if (Test-Path (Join-Path $OutDir "BASIC\$y")) {
            tar.exe -cf (Join-Path $OutDir $tarName) -C $OutDir "BASIC/$y"
            Write-Host "Packed $tarName"
        }
    }
}
