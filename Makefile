run:
	uvicorn main:app --reload

test:
	powershell -NoProfile -Command "$$OutputEncoding = [Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; $$res = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/generate-podcast -ContentType 'application/json' -Body '{\"topic\": \"Black Holes\", \"tone\": \"Casual\", \"language\": \"English\"}'; Write-Host \"`n--- GENERATED SCRIPT ---`n\"; [System.Console]::WriteLine($$res.script)"
