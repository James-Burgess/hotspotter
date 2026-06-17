<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{data['run_id']}} — Benchmark</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.6; }
.container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
h1 { font-size: 1.4rem; font-weight: 700; }
h2 { font-size: 1.05rem; font-weight: 600; margin: 1.75rem 0 0.5rem; padding-bottom: 0.3rem; border-bottom: 1px solid #334155; }
h3 { font-size: 0.9rem; font-weight: 600; color: #e2e8f0; margin: 0.6rem 0 0.3rem; }
a { color: #818cf8; text-decoration: none; }
a:hover { text-decoration: underline; }
.back { display: inline-block; margin-bottom: 1rem; font-size: 0.85rem; color: #94a3b8; }
.back:hover { color: #e2e8f0; }
.card { background: #1e293b; border-radius: 10px; padding: 1.25rem 1.5rem; border: 1px solid #334155; margin-bottom: 0.75rem; }
.badge { display: inline-block; padding: 0.1rem 0.4rem; border-radius: 3px; font-size: 0.65rem; font-weight: 600; }
.badge.match { background: #065f46; color: #6ee7b7; }
.mono { font-family: 'SF Mono','Fira Code','Consolas',monospace; font-size: 0.72rem; }

/* Results table */
.results-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
.results-table th { text-align: left; padding: 0.3rem 0.3rem; color: #94a3b8; font-weight: 600; font-size: 0.65rem; text-transform: uppercase; border-bottom: 1px solid #334155; white-space: nowrap; }
.results-table td { padding: 0.25rem 0.3rem; border-bottom: 1px solid #1e293b; vertical-align: middle; }
.results-table tr:hover td { background: #24344d; }
.results-table tr:hover td[style] { background: #24344d; }
.results-table .rn { font-weight: 700; color: #818cf8; }
.results-table .rs { font-size: 0.75rem; }
.qchip { width: 28px; height: 28px; object-fit: cover; border-radius: 3px; border: 1px solid #334155; vertical-align: middle; }

/* Agreement */
.agree-stat { display: inline-flex; align-items: center; gap: 0.25rem; margin-right: 1.5rem; margin-bottom: 0.35rem; font-size: 0.82rem; }
.agree-val { font-weight: 700; }

/* Logs */
.log-link { cursor: pointer; font-size: 0.8rem; }
.modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.75); z-index: 1000; align-items: center; justify-content: center; }
.modal-overlay.open { display: flex; }
.modal-dialog { background: #1e293b; border: 1px solid #334155; border-radius: 12px; max-width: 90vw; max-height: 85vh; width: 1000px; display: flex; flex-direction: column; }
.modal-header { display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.5rem; border-bottom: 1px solid #334155; }
.modal-header h3 { margin: 0; font-size: 0.95rem; }
.modal-close { background: none; border: none; color: #94a3b8; font-size: 1.5rem; cursor: pointer; padding: 0; line-height: 1; }
.modal-close:hover { color: #e2e8f0; }
.modal-body { padding: 1rem 1.5rem; overflow: auto; flex: 1; }
.modal-body pre { margin: 0; font-size: 0.75rem; white-space: pre-wrap; word-break: break-all; color: #e2e8f0; }
.modal-spinner { display: inline-block; width: 20px; height: 20px; border: 2px solid #334155; border-top-color: #6366f1; border-radius: 50%%; animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

.overlay-img { max-width: 100%; border-radius: 8px; border: 1px solid #334155; margin-top: 0.5rem; }
</style>
</head>
<body>
<div class="container">
  <a href="/" class="back">&larr; Back to runs</a>
  <h1>{{data['run_id']}}</h1>

  % cfg = data.get('config', {})
  <div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:center;margin-top:0.25rem;font-size:0.8rem;color:#94a3b8;">
    % if cfg.get('n_annots'):
    <span>{{cfg['n_annots']}} annots</span>
    % end
    <span>{{len(data.get('per_query', []))}} queries</span>
    % if cfg.get('seed'):
    <span>seed={{cfg['seed']}}</span>
    % end
    % if cfg.get('species'):
    <span>{{cfg['species']}}</span>
    % end
    % for name in data.get('targets', []):
    <span style="display:inline-flex;align-items:center;gap:0.2rem;">
      <span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:{{tc(name)}};"></span>
      {{name}}
    </span>
    % end
  </div>

  % ag = data.get('agreement', {})
  % aspear = data.get('aggregate_spearman', {})
  % ov = data.get('top3_overall_overlap', {})
  <h2>Agreement</h2>
  <div class="card">
    % for key, stats in aspear.items():
    <span class="agree-stat">
      <span style="color:#94a3b8;">&rho;({{key}})</span>
      <span class="agree-val" style="color:{{'#6ee7b7' if stats['mean_rho'] > 0.95 else '#fcd34d'}};">{{'%.4f' % stats['mean_rho']}}</span>
    </span>
    % end
    % for key, val in ov.items():
    <span class="agree-stat">
      <span style="color:#94a3b8;">top-3 {{key}}</span>
      <span class="agree-val" style="color:{{'#6ee7b7' if val > 0.8 else '#fcd34d'}};">{{'%.3f' % val}}</span>
    </span>
    % end
    % if ag.get('top1_identical') is not None:
    <span class="agree-stat">
      <span style="color:#94a3b8;">top-1 identical</span>
      <span class="agree-val" style="color:{{'#6ee7b7' if ag['top1_identical'] else '#fca5a5'}};">{{'yes' if ag['top1_identical'] else 'no'}}</span>
    </span>
    % end
    % if ag.get('all_rankings_match') is not None:
    <span class="agree-stat">
      <span style="color:#94a3b8;">rankings match</span>
      <span class="agree-val" style="color:{{'#6ee7b7' if ag['all_rankings_match'] else '#fcd34d'}};">{{'yes' if ag['all_rankings_match'] else 'no'}}</span>
    </span>
    % end
    <span class="agree-stat">
      <span style="color:#94a3b8;">max score &delta;</span>
      <span class="agree-val">{{'%.4f' % ag.get('max_score_delta', 0)}}</span>
    </span>
  </div>

  % acc_q = data.get('accuracy', {}).get('per_query', [])
  % targets = data.get('targets', [])
  % if targets:
  <h2>Results</h2>
  <div class="card" style="padding:0.5rem 0;">
    <table class="results-table">
      <thead>
        <tr>
          <th>Q</th>
          <th>Target</th>
          <th style="width:32px;"></th>
          <th>#1</th>
          <th>score</th>
          <th style="width:32px;"></th>
          <th>#2</th>
          <th>score</th>
          <th style="width:32px;"></th>
          <th>#3</th>
          <th>score</th>
          <th style="width:32px;"></th>
          <th>#4</th>
          <th>score</th>
          <th style="width:32px;"></th>
          <th>#5</th>
          <th>score</th>
          <th>&mu;</th>
          <th>&sigma;</th>
          <th>match</th>
        </tr>
      </thead>
      <tbody>
        % for q in data.get('per_query', []):
        % qi = q['query_index']
        % aq = next((a for a in acc_q if a.get('query_index') == qi), {})
        % tk = data.get('top_k_aids', {}).get(str(qi), {})
        % for name in targets:
        <tr>
          <td class="rn">{{qi}}</td>
          <td style="color:{{tc(name)}};font-weight:600;">{{name}}</td>
          % scores = tk.get(name, [])
          % for rank_idx in range(5):
          % if rank_idx < len(scores):
          <td><img class="qchip" src="/coco_image/{{scores[rank_idx]['aid']}}" alt="" onerror="this.style.display='none'"></td>
          <td class="mono">{{scores[rank_idx]['aid'].replace('coco-annot-', '')}}</td>
          <td class="rs">{{scores[rank_idx]['score']}}</td>
          % else:
          <td></td>
          <td></td>
          <td></td>
          % end
          % end
          % qs = q.get('score_stats', {}).get(name, {})
          <td class="rs" style="color:#64748b;">{{round(qs.get('mean', 0), 1) if qs else '-'}}</td>
          <td class="rs" style="color:#64748b;">{{round(qs.get('std', 0), 1) if qs else '-'}}</td>
          % tinfo = aq.get('targets', {}).get(name, {})
          % rank = tinfo.get('rank')
          % if rank == 1:
          <td style="color:#6ee7b7;">#1</td>
          % elif rank is not None:
          <td style="color:#fcd34d;">#{{rank}}</td>
          % else:
          <td style="color:#fca5a5;">miss</td>
          % end
        </tr>
        % end
        <tr style="background:#0f172a;font-size:0.7rem;color:#64748b;">
          <td></td>
          <td></td>
          <td colspan="17">
            % for pair in q.get('spearman_pairs', []):
            &rho;({{pair['a']}}, {{pair['b']}}) = {{pair['rho'] if pair.get('rho') else 'N/A'}} &nbsp;
            % end
            max &delta; = {{q.get('max_score_delta', 0)}}
            % if run_id:
            <br><img src="/overlay/{{run_id}}/{{qi}}" alt="Overlay q{{qi}}" style="max-width:100%;border-radius:6px;border:1px solid #334155;margin-top:0.3rem;" onerror="this.style.display='none'">
            % end
          </td>
        </tr>
        % end
      </tbody>
    </table>
  </div>
  % end

  % if data.get('errors'):
  <h2>Errors</h2>
  <div class="card">
    % for e in data['errors']:
    <div style="font-size:0.8rem;color:#fca5a5;padding:0.2rem 0;">[{{e.get('target', '?')}}] query {{e.get('query_index', '?')}}: {{e.get('message', e)}}</div>
    % end
  </div>
  % end

  % if log_files:
  <h2>Debug Logs</h2>
  <div class="card">
    % for lf in log_files:
    <div style="display:flex;align-items:center;gap:0.5rem;padding:0.25rem 0;">
      <a href="#" class="log-link mono" data-log="/run/{{run_id}}/logs/{{lf['name']}}" data-name="{{lf['name']}}" style="color:#818cf8;">{{lf['name']}}</a>
      <span style="color:#64748b;font-size:0.75rem;">({{lf['size_str']}})</span>
    </div>
    % end
  </div>
  % end

</div>

<div class="modal-overlay" id="log-modal">
  <div class="modal-dialog">
    <div class="modal-header">
      <h3 id="log-modal-title">Log</h3>
      <button class="modal-close" onclick="closeLogModal()">&times;</button>
    </div>
    <div class="modal-body" id="log-modal-body">
      <div class="modal-spinner"></div>
    </div>
  </div>
</div>

<script>
const modal = document.getElementById('log-modal');
const modalTitle = document.getElementById('log-modal-title');
const modalBody = document.getElementById('log-modal-body');
function closeLogModal() { modal.classList.remove('open'); }
modal.addEventListener('click', (e) => { if (e.target === modal) closeLogModal(); });
document.querySelectorAll('.log-link').forEach(link => {
  link.addEventListener('click', async (e) => {
    e.preventDefault();
    modalTitle.textContent = link.dataset.name;
    modalBody.innerHTML = '<div class="modal-spinner"></div>';
    modal.classList.add('open');
    try {
      const resp = await fetch(link.dataset.log);
      modalBody.innerHTML = '<pre>' + escapeHtml(await resp.text()) + '</pre>';
    } catch (err) {
      modalBody.innerHTML = '<pre style="color:#fca5a5;">Failed to load: ' + escapeHtml(err.message) + '</pre>';
    }
  });
});
function escapeHtml(str) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}
</script>
</body>
</html>
