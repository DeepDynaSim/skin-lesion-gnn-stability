param([Parameter(Mandatory=$true)][string]$InputDocx,[Parameter(Mandatory=$true)][string]$OutputPdf)
$ErrorActionPreference='Stop'
$wordApp=$null
$wordDoc=$null
try {
    $wordApp=New-Object -ComObject Word.Application
    $wordApp.Visible=$false
    $wordApp.DisplayAlerts=0
    $wordDoc=$wordApp.Documents.Open($InputDocx,$false,$true)
    $wordDoc.Fields.Update() | Out-Null
    $wordDoc.Repaginate()
    Write-Output ('Native Word equations: '+$wordDoc.OMaths.Count)
    Write-Output ('Word pages: '+$wordDoc.ComputeStatistics(2))
    $wordDoc.ExportAsFixedFormat($OutputPdf,17)
} finally {
    if ($null -ne $wordDoc) { $wordDoc.Close(0); [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordDoc) }
    if ($null -ne $wordApp) { $wordApp.Quit(); [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordApp) }
}
