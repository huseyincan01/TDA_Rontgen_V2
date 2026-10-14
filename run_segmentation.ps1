# 5-Bölge Akciğer Segmentasyon ve Hasta Kesim Çalıştırıcı
param (
    [int]$Limit = 0,
    [string]$OutputDir = "data/processed/zones",
    [switch]$SkipExisting
)

$PythonExe = "C:\Users\Huseyincan\Desktop\segmentasyon deneme\venv\Scripts\python.exe"

$ArgsList = @("pipelines/03_make_zones.py", "--output-dir", $OutputDir)
if ($Limit -gt 0) {
    $ArgsList += @("--limit", $Limit)
}
if ($SkipExisting) {
    $ArgsList += @("--skip-existing")
}

Write-Host "GPU Segmentasyon Boru Hattı Başlatılıyor..." -ForegroundColor Cyan
& $PythonExe $ArgsList
