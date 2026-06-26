"""Render a self-contained, GitHub-Pages-ready dashboard from leaderboard.json.

The hero is a 2-D "cultural alignment map": every model is a point whose
y-position is human-likeness (within-language HM similarity) and x-position is
cross-cultural diversity (how far its cross-language similarity sits BELOW the
human-flattening line). The human baselines are drawn as crosshairs, so the
top-right quadrant is the ideal zone: human-like AND culturally diverse.

Data is embedded inline so the page works as a static file (no fetch/CORS),
which is exactly what GitHub Pages serves.
"""
from __future__ import annotations

import json

from . import DOCS

PROVIDER_COLORS = {
    "openai": "#10a37f", "google": "#4285f4", "anthropic": "#d97757",
    "alibaba": "#7b2ff7", "deepseek": "#2563eb", "lambda": "#6b7280", "—": "#9ca3af",
}

_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multilingual Story-Morals Leaderboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root{--bg:#0f1117;--card:#171a23;--ink:#e7e9ee;--mut:#9aa3b2;--line:#262b38;--good:#3ddc97;--bad:#ff6b6b}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  .wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
  h1{font-size:30px;margin:0 0 4px} .sub{color:var(--ink);font-size:17px;margin:0 0 12px}
  .desc{color:var(--mut);font-size:14px;max-width:780px;margin:0 0 10px}
  .cite{display:block;margin-top:8px;font-size:12.5px;color:#7e879a}
  .meta{color:var(--mut);font-size:13px;margin-bottom:24px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:24px}
  h2{font-size:18px;margin:0 0 4px} .hint{color:var(--mut);font-size:13px;margin:0 0 14px}
  table{width:100%;border-collapse:collapse;font-size:14px}
  th,td{text-align:right;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
  th:nth-child(2),td:nth-child(2){text-align:left} th{color:var(--mut);font-weight:600;cursor:pointer;user-select:none}
  th:hover{color:var(--ink)} tr:hover td{background:#1c2030}
  .rank{color:var(--mut)} .name{font-weight:600}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;vertical-align:middle}
  .badge{font-size:11px;padding:2px 8px;border-radius:999px;font-weight:600}
  .ideal{background:rgba(61,220,151,.16);color:#3ddc97}
  .near{background:rgba(255,224,102,.16);color:#ffe066}
  .off{background:rgba(255,159,67,.16);color:#ff9f43}
  .far-off{background:rgba(255,107,107,.16);color:#ff6b6b}
  .legend{color:var(--mut);font-size:12.5px;margin-top:10px}
  .langlegend{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
  .lchip{font-size:12px;color:var(--mut);background:#0c0e14;border:1px solid var(--line);
    padding:3px 9px;border-radius:7px} .lchip b{color:var(--ink);font-weight:600}
  code{background:#0c0e14;padding:1px 6px;border-radius:5px;color:#c9d2e3}
  a{color:#8aa6ff}
</style></head><body><div class="wrap">
  <h1>Lessons Without Borders — Multilingual Story-Morals Leaderboard</h1>
  <p class="sub">Do language models capture the cultural diversity of human story morals — or just restate one lesson in many languages?</p>
  <p class="desc">Story morals are the short lessons readers infer from narratives, and the lesson drawn from the same story varies across cultures.
    This leaderboard adapts the evaluation of <b>Wu &amp; Piper (2026)</b> to a public, periodically-updated board: using a dataset of
    human-written morals across 14 language–culture pairs, it measures whether each model's morals (1) match human interpretations
    <i>within</i> a language and (2) preserve the cross-cultural variation humans show, rather than collapsing to a single moral expressed
    in 14 languages. A model succeeds by being human-like <i>and</i> culturally diverse.
    <span class="cite">Sophie Wu &amp; Andrew Piper, <i>“Lessons Without Borders: Evaluating Cultural Alignment of LLMs Using Multilingual
    Story Moral Generation.”</i> <a href="https://arxiv.org/pdf/2604.08797" target="_blank" rel="noopener">arXiv:2604.08797</a>.</span></p>
  <p class="meta" id="meta"></p>

  <div class="card">
    <h2>Distance to the human point</h2>
    <p class="hint">The goal is to <b>match</b> humans, not beat them — sit at the crosshair (the human baseline).
      Axes are in <b>human standard deviations from the baseline</b>: ▲/▼ = more/less similar to human morals
      within a language; ◀ = flattening (less cross-cultural variation than humans). Dots are colored by
      model family; the green arcs mark the <b>ideal</b> (≤0.5 SD) and <b>near</b> (≤1 SD) zones around the
      human point. Closest to the crosshair wins.</p>
    <div id="map" style="height:560px"></div>
  </div>

  <div class="card">
    <h2>Alignment by language</h2>
    <p class="hint">How far each model's morals sit <b>above</b> (green) or
      <b>below</b> (red) typical human–human agreement <i>within</i> that language. <b>*</b> marks a
      statistically significant difference (p&lt;.05, clustered by story). Reveals where a model is human-like
      and where it slips.</p>
    <div id="heat"></div>
    <div id="langlegend" class="langlegend"></div>
  </div>

  <div class="card">
    <h2>Full standings</h2>
    <p class="hint">Click a column to sort. <b>Distance</b> = human standard deviations from the human point
      (lower = better); ranked by it.</p>
    <table id="tbl"><thead><tr>
      <th data-k="rank">#</th><th data-k="display">Model</th><th data-k="provider">Provider</th>
      <th data-k="alignment_mean">Alignment</th><th data-k="alignment_gap">Δ human</th>
      <th data-k="diversity_mean">Cross-lang sim</th><th data-k="diversity_gap">Flattening</th>
      <th data-k="distance">Distance (SD)</th><th data-k="verdict">Zone</th>
    </tr></thead><tbody></tbody></table>
    <p class="legend">
      <span class="badge ideal">ideal</span> ≤0.5 SD &nbsp;
      <span class="badge near">near</span> 0.5–1 SD &nbsp;
      <span class="badge off">off</span> 1–1.5 SD &nbsp;
      <span class="badge far-off">far-off</span> &gt;1.5 SD &nbsp; · hover a row's zone for how it misses
    </p>
  </div>
  <p class="meta">Method: cosine similarity over multilingual sentence embeddings (Wu &amp; Piper, Figs 3–4).
    Score = Euclidean distance to the human point, each axis standardized by the human standard deviation.
    Embedders: <span id="emb"></span>.</p>
</div>
<script>
const DATA = __DATA__;
const COLORS = __COLORS__;
const rows = DATA.rows, B = DATA.baselines;
document.getElementById('meta').textContent =
  `${DATA.n_models} models · human within-language ${B.within} ±${B.within_sd} · ` +
  `human cross-language ${B.cross} ±${B.cross_sd} (±1 SD = inner ring) · updated ${DATA.generated_at}`;
document.getElementById('emb').textContent = DATA.embedders.join(', ');
const col = p => COLORS[p] || '#9ca3af';
// 4-tier zone — must stay in sync with _zone() in leaderboard.py.
const zoneOf = d => d <= 0.5 ? 'ideal' : d <= 1 ? 'near' : d <= 1.5 ? 'off' : 'far-off';

// ---- Map: distance to the human point, in SD units. Origin = human.
// x = -x_se so rightward = toward human diversity (left = flattening); y = y_se.
// Axes autorange INDEPENDENTLY so both the (wide) flattening axis and the
// (narrow) alignment axis fill the frame — the top-aligned model sits near the
// top. Markers keep their model-family (provider) color; the 0.5/1 SD rings are
// drawn as circle shapes that render as clipped ellipse ARCS here — still the
// correct distance contours, so a dot inside an arc is genuinely within that SD.
const px = rows.map(r => -r.x_se), py = rows.map(r => r.y_se);
const cd = rows.map(r => [r.display, r.distance, r.alignment_mean, r.diversity_mean, zoneOf(r.distance)]);
const ring = (rad, color, fill) => ({type:'circle', xref:'x', yref:'y',
  x0:-rad, y0:-rad, x1:rad, y1:rad, line:{color, width:1.2, dash:'dot'}, fillcolor:fill, layer:'below'});
// Explicit ranges from the data + the human point (origin), so the frame hugs
// the cloud and the rings clip rather than driving the range. Small y-pad keeps
// the top model near the top. Works for any result set.
const PADX = 0.2, PADY = 0.06;
const xlo = Math.min(...px, 0) - PADX, xhi = Math.max(...px, 0) + PADX;
const ylo = Math.min(...py, 0) - PADY, yhi = Math.max(...py, 0) + PADY;
Plotly.newPlot('map', [
  {x:px, y:py, customdata:cd, text:rows.map(r=>r.display),
   mode:'markers+text', textposition:'top center', textfont:{color:'#cdd3df',size:10},
   marker:{size:13, color:rows.map(r=>col(r.provider)), line:{color:'#0f1117',width:1.5}},
   hovertemplate:'%{customdata[0]}<br>distance %{customdata[1]} SD (%{customdata[4]})'
     +'<br>align %{customdata[2]:.3f} · cross-sim %{customdata[3]:.3f}<extra></extra>'},
  {x:[0], y:[0], mode:'markers+text', text:['human'], textposition:'bottom center',
   textfont:{color:'#3ddc97',size:12}, marker:{symbol:'star', size:15, color:'#3ddc97'},
   hovertemplate:'human baseline (target)<extra></extra>'}
],{
  showlegend:false, paper_bgcolor:'#171a23', plot_bgcolor:'#171a23', font:{color:'#9aa3b2'},
  margin:{l:60,r:30,t:20,b:55},
  xaxis:{title:'◀ flattening  ·  cross-cultural diversity (SD from human)  ·  human level ▶',
         range:[xlo, xhi], zeroline:false, gridcolor:'#262b38'},
  yaxis:{title:'within-language similarity (SD from human)',
         range:[ylo, yhi], zeroline:false, gridcolor:'#262b38'},
  shapes:[
    ring(1, '#5b6473', 'rgba(0,0,0,0)'),
    ring(0.5, '#3ddc97', 'rgba(61,220,151,.08)'),
    {type:'line', xref:'x', x0:0, x1:0, yref:'paper', y0:0, y1:1, line:{color:'#9aa3b2', dash:'dash', width:1}},
    {type:'line', xref:'paper', x0:0, x1:1, yref:'y', y0:0, y1:0, line:{color:'#9aa3b2', dash:'dash', width:1}}
  ]
},{responsive:true,displayModeBar:false});

// ---- Heatmap: per-language alignment advantage (Fig 9 style)
const langs = (DATA.language_order && DATA.language_order.length)
  ? DATA.language_order
  : [...new Set(rows.flatMap(r=>Object.keys(r.by_language||{})))].sort();
const hmRows = [...rows].sort((a,b)=>a.rank-b.rank);   // rank 1 first
const z = hmRows.map(r => langs.map(L => {
  const c = (r.by_language||{})[L]; return c && c.advantage!=null ? c.advantage : null;}));
const cellTxt = hmRows.map(r => langs.map(L => {
  const c = (r.by_language||{})[L];
  return (c && c.p!=null && c.p<0.05) ? '∗' : '';}));
const amax = Math.max(0.001, ...z.flat().filter(v=>v!=null).map(Math.abs));
Plotly.newPlot('heat', [{
  type:'heatmap', x:langs, y:hmRows.map(r=>r.display), z:z,
  text:cellTxt, texttemplate:'%{text}', textfont:{size:15,color:'#0f1117'},
  zmid:0, zmin:-amax, zmax:amax,
  colorscale:[[0,'#c0392b'],[0.5,'#222633'],[1,'#3ddc97']],
  xgap:2, ygap:2,
  colorbar:{title:{text:'Δ vs human',side:'right'},thickness:12,len:.75,tickfont:{color:'#9aa3b2'}},
  hovertemplate:'%{y} · %{x}<br>advantage %{z:.3f}<extra></extra>'
}],{
  paper_bgcolor:'#171a23', plot_bgcolor:'#171a23', font:{color:'#9aa3b2'},
  margin:{l:150,r:20,t:10,b:36}, height:Math.max(160, hmRows.length*34+90),
  yaxis:{automargin:true, autorange:'reversed'}, xaxis:{side:'top'}
},{responsive:true,displayModeBar:false});

// ---- Language legend
const ll = document.getElementById('langlegend');
ll.innerHTML = langs.map(L=>{
  const m = (DATA.languages||{})[L];
  const label = m ? m.name : L;
  return `<span class="lchip"><b>${L}</b> · ${label}</span>`;
}).join('');

// ---- Table
const tb = document.querySelector('#tbl tbody');
const fmt = (v)=> (v===null||v===undefined||Number.isNaN(v))?'—':v;
function render(data){
  tb.innerHTML='';
  for(const r of data){
    const tr=document.createElement('tr');
    tr.innerHTML=`<td class="rank">${r.rank}</td>
      <td class="name"><span class="dot" style="background:${col(r.provider)}"></span>${r.display}</td>
      <td>${r.provider}</td>
      <td>${fmt(r.alignment_mean)}</td>
      <td>${r.alignment_gap>=0?'+':''}${fmt(r.alignment_gap)}</td>
      <td>${fmt(r.diversity_mean)}</td>
      <td>${r.diversity_gap>=0?'+':''}${fmt(r.diversity_gap)}</td>
      <td>${fmt(r.distance)}</td>
      <td><span class="badge ${zoneOf(r.distance)}" title="miss: ${r.miss}">${zoneOf(r.distance)}</span></td>`;
    tb.appendChild(tr);
  }
}
render(rows);
let asc={};
document.querySelectorAll('#tbl th').forEach(th=>th.onclick=()=>{
  const k=th.dataset.k; asc[k]=!asc[k];
  const s=[...rows].sort((a,b)=>{const x=a[k],y=b[k];
    return (typeof x==='number'?x-y:String(x).localeCompare(String(y)))*(asc[k]?1:-1);});
  render(s);
});
</script></body></html>
"""


def build(payload: dict) -> str:
    DOCS.mkdir(exist_ok=True)
    html = (_TEMPLATE
            .replace("__DATA__", json.dumps(payload))
            .replace("__COLORS__", json.dumps(PROVIDER_COLORS)))
    out = DOCS / "index.html"
    out.write_text(html)
    return str(out)
