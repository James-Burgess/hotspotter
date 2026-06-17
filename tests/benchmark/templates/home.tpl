
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Parity Benchmark Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.6; }
.container { max-width: 1100px; margin: 0 auto; padding: 2rem; }
h1 { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem; }
.subtitle { color: #94a3b8; margin-bottom: 2rem; font-size: 0.95rem; }
.run-list { display: flex; flex-direction: column; gap: 0.75rem; }
.run-card { background: #1e293b; border-radius: 10px; padding: 1.25rem 1.5rem; border: 1px solid #334155; transition: border-color 0.15s; cursor: pointer; text-decoration: none; color: inherit; display: block; }
.run-card:hover { border-color: #6366f1; }
.run-card h3 { font-size: 1.05rem; margin-bottom: 0.5rem; }
.run-meta { display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.85rem; color: #94a3b8; }
.run-meta span { display: inline-flex; align-items: center; gap: 0.35rem; }
.badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.badge.pass { background: #065f46; color: #6ee7b7; }
.badge.fail { background: #7f1d1d; color: #fca5a5; }
.badge.info { background: #1e3a5f; color: #93c5fd; }
.badge.warn { background: #78350f; color: #fcd34d; }
.badge.success { background: #064e3b; color: #6ee7b7; }
.badge.partial { background: #78350f; color: #fcd34d; }
.mono { font-family: 'SF Mono','Fira Code','Consolas',monospace; font-size: 0.8rem; }
.empty { text-align: center; padding: 3rem; color: #64748b; }
.empty h2 { font-size: 1.5rem; margin-bottom: 0.5rem; }
</style>
</head>
<body>
<div class="container">
  <h1>&#x1F50D; Benchmark Dashboard</h1>
  <p class="subtitle">Test run results for wbia-core</p>
  % if not runs:
  <div class="empty">
    <h2>No results found</h2>
    <p>Run <code>python3 tests/benchmark/run_benchmark.py</code> to generate results, or check the <code>test-results/</code> directory.</p>
  </div>
  % end
  <div class="run-list">
  % for r in runs:
    <a href="/run/{{r['id']}}" class="run-card">
      <h3>{{r['name']}}</h3>
      <div class="run-meta">
        % if r.get('status') and r['status'] != 'unknown':
          <span class="badge {{r['status']}}">{{r['status']}}</span>
        % end
        % if r['targets']:
          <span>targets: {{' + '.join(r['targets'])}}</span>
        % end
        % if r['n_queries']:
          <span>{{r['n_queries']}} queries</span>
        % end
        % if r.get('n_annotations'):
          <span>{{r['n_annotations']}} annotations</span>
        % end
        % if r.get('duration'):
          <span>{{'%.0f' % r['duration']}}s</span>
        % end
        % if r['n_errors']:
          <span class="badge warn">{{r['n_errors']}} error(s)</span>
        % end
        % if r.get('species'):
          <span>{{r['species']}}</span>
        % end
        % if r.get('seed'):
          <span>seed={{r['seed']}}</span>
        % end
        % if r['agreement'].get('top1_identical') is not None:
          {{!pf(r['agreement']['top1_identical'])}}
        % end
        % if r.get('date'):
          <span class="mono">{{r['date']}}</span>
        % end
        % if r.get('git', {}).get('commit'):
          <span class="mono">{{r['git']['commit'][:12]}}</span>
        % end
      </div>
    </a>
  % end
  </div>
</div>
</body>
</html>
