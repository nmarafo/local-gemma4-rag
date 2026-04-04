param (
    [string]$msg = "Sync changes from Antigravity"
)

Write-Host "🔄 Synchronizing repository..." -ForegroundColor Yellow

# Check for changes
$status = git status --porcelain
if (-not $status) {
    Write-Host "✨ No changes to sync." -ForegroundColor Green
    exit
}

# 1. Add all changes
git add --all

# 2. Commit
git commit -m $msg

# 3. Push to GitHub
Write-Host "🚀 Pushing to GitHub (origin)..." -ForegroundColor Cyan
git push origin main

# 4. Push to Hugging Face
Write-Host "🚀 Pushing to Hugging Face (huggingface)..." -ForegroundColor Cyan
git push huggingface main

Write-Host "✅ Sync complete!" -ForegroundColor Green
