Get-ChildItem 'G:\PROJECTS\MyIAModelChat\DOCS\GENERIC_BOOKS\*.md' | Where-Object { $_.Name -notmatch '^temp_' } | ForEach-Object {
    $kb = [math]::Round($_.Length / 1KB, 2)
    $status = if ($kb -ge 150) { 'DONE' } elseif ($kb -ge 100) { 'NEAR' } else { 'NEEDS' }
    Write-Output "$($_.Name): $kb KB - $status"
}
