
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{run_name}} — In Progress</title>
<style>
body { font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.6; padding: 2rem; }
a { color: #818cf8; text-decoration: none; }
.in-progress { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 2rem; margin-top: 1rem; text-align: center; }
.spinner { display: inline-block; width: 24px; height: 24px; border: 3px solid #334155; border-top-color: #6366f1; border-radius: 50%%; animation: spin 1s linear infinite; margin-bottom: 1rem; }
@keyframes spin { to { transform: rotate(360deg); } }
h2 { color: #93c5fd; margin-bottom: 0.5rem; }
p { color: #64748b; }
</style>
</head>
<body>
  <a href="/">&larr; Back</a>
  <div class="in-progress">
    <div class="spinner"></div>
    <h2>Run in progress</h2>
    <p>{{run_name}} is still running. Results will appear once complete.</p>
    <p style="margin-top:0.5rem;"><a href="/run/{{run_name}}">Refresh</a></p>
  </div>
</body>
</html>
