/**
 * Systematic Review UI — app.js
 * Communicates with FastAPI backend at /api/*
 */

const API = '/api';
let currentRecordId = null;
let pipelineLogInterval = null;

// ─────────────────────────────────────────────
// NAVIGATION
// ─────────────────────────────────────────────

document.querySelectorAll('.nav-item').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    const panel = link.dataset.panel;
    switchPanel(panel);
    link.closest('ul').querySelectorAll('.nav-item').forEach(l => l.classList.remove('active'));
    link.classList.add('active');
  });
});

function switchPanel(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  const target = document.getElementById(`panel-${name}`);
  if (target) target.classList.add('active');
  // Load data for panel
  if (name === 'dashboard') loadDashboard();
  if (name === 'verify') loadVerifyQueue();
  if (name === 'fulltext') loadFulltextNeeded();
  if (name === 'second-pass') loadSecondPass(1);
  if (name === 'included') loadIncluded();
  if (name === 'excluded') loadExcluded();
  if (name === 'config') loadConfig();
  if (name === 'export') loadExportPreview();
}

// ─────────────────────────────────────────────
// API HELPERS
// ─────────────────────────────────────────────

async function apiFetch(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function apiPost(path, body) {
  return apiFetch(path, { method: 'POST', body: JSON.stringify(body) });
}

async function apiPatch(path, body) {
  return apiFetch(path, { method: 'PATCH', body: JSON.stringify(body) });
}

// ─────────────────────────────────────────────
// DASHBOARD
// ─────────────────────────────────────────────

async function loadDashboard() {
  try {
    const status = await apiFetch('/pipeline/status');
    const p = status.prisma_counts;

    setText('stat-identified', p.identified ?? '—');
    setText('stat-after-dedup', p.after_dedup ?? '—');
    setText('stat-title-included', p.title_abstract_included ?? '—');
    setText('stat-fulltext-retrieved', p.fulltext_retrieved ?? '—');
    setText('stat-fulltext-needed', p.full_text_needed ?? '—');
    setText('stat-r1-excluded', p.fulltext_r1_excluded ?? '—');
    setText('stat-r1-passed', p.fulltext_r1_passed ?? '—');
    setText('stat-r2-excluded', p.fulltext_r2_excluded ?? '—');
    setText('stat-final-included', p.final_included ?? '—');
    setText('stat-needs-human', p.needs_human_verification ?? '—');

    setBadge('badge-uncertain', p.needs_human_verification);
    setBadge('badge-fulltext', p.full_text_needed);

    renderPrismaFlow(p);
    await loadRecentRecords();
  } catch (e) {
    console.error('Dashboard load error:', e);
  }
}

function renderPrismaFlow(p) {
  const el = document.getElementById('prisma-flow');

  // Helper: show '—' for null/undefined, with optional color
  const n = (v, color) => {
    const val = (v == null) ? '—' : v;
    return color ? `<span style="color:${color};">${val}</span>` : val;
  };

  // Derived check totals (for the verification note)
  const ftSum = (p.fulltext_retrieved ?? 0) + (p.full_text_needed ?? 0);
  const r1Sum = (p.fulltext_r1_excluded ?? 0) + (p.fulltext_r1_passed ?? 0) + (p.fulltext_r1_uncertain ?? 0) + (p.full_text_needed ?? 0);
  const r2Sum = (p.fulltext_r2_excluded ?? 0) + (p.fulltext_r2_included ?? 0) + (p.fulltext_r2_uncertain ?? 0);
  const ta = p.title_abstract_included ?? 0;

  const checkIcon = (sum, expected) =>
    (expected > 0 && sum === expected)
      ? `<span style="color:var(--green); font-size:11px;"> ✓ sums to ${sum}</span>`
      : (expected > 0 ? `<span style="color:var(--red); font-size:11px;"> ✗ ${sum} ≠ ${expected}</span>` : '');

  const r2HasRun = (p.fulltext_r2_excluded ?? 0) + (p.fulltext_r2_included ?? 0) + (p.fulltext_r2_uncertain ?? 0) > 0;

  el.innerHTML = `
  <div style="font-family:var(--mono); font-size:12px; line-height:1.8; padding:8px 0;">

    <!-- Row: Identified -->
    <div class="prisma-row">
      <div class="prisma-box">
        <div class="pbox-count">${n(p.identified)}</div>
        <div class="pbox-label">Records Identified</div>
      </div>
    </div>
    <div class="prisma-arrow">↓</div>

    <!-- Row: After dedup -->
    <div class="prisma-row">
      <div class="prisma-box">
        <div class="pbox-count">${n(p.after_dedup)}</div>
        <div class="pbox-label">After Deduplication</div>
      </div>
      <div class="prisma-side" style="color:var(--red);">← ${n(p.duplicates_removed)} duplicates removed</div>
    </div>
    <div class="prisma-arrow">↓</div>

    <!-- Row: Title/abstract included -->
    <div class="prisma-row">
      <div class="prisma-box">
        <div class="pbox-count">${n(p.title_abstract_included)}</div>
        <div class="pbox-label">Title / Abstract Included</div>
      </div>
      <div class="prisma-side" style="color:var(--red);">← ${n(p.title_abstract_excluded)} excluded at title/abstract</div>
    </div>
    <div class="prisma-arrow">↓</div>

    <!-- Row: Full text retrieved vs needed -->
    <div class="prisma-row">
      <div class="prisma-box">
        <div class="pbox-count" style="color:var(--green);">${n(p.fulltext_retrieved)}</div>
        <div class="pbox-label">Full Text Retrieved</div>
      </div>
      <div class="prisma-side" style="color:var(--orange);">← ${n(p.full_text_needed)} full texts not retrieved (manual upload needed)</div>
    </div>
    <div style="font-size:11px; color:var(--text-muted); text-align:center; margin:-4px 0 4px;">
      retrieved + not retrieved = ${n(ftSum)}${checkIcon(ftSum, ta)}
    </div>
    <div class="prisma-arrow">↓</div>

    <!-- Row: Round-1 full-text screening -->
    <div class="prisma-row">
      <div class="prisma-box" style="border-color:var(--accent);">
        <div class="pbox-count" style="color:var(--accent);">${n(p.fulltext_r1_passed)}</div>
        <div class="pbox-label">Round-1 Passed (for Round-2)</div>
      </div>
      <div class="prisma-side" style="color:var(--red);">← ${n(p.fulltext_r1_excluded)} excluded at round-1 full-text
        ${p.fulltext_r1_uncertain > 0 ? `<br>← ${n(p.fulltext_r1_uncertain, 'var(--orange)')} pending human review (round-1)` : ''}
      </div>
    </div>
    <div style="font-size:11px; color:var(--text-muted); text-align:center; margin:-4px 0 4px;">
      r1-excluded + r1-passed + r1-uncertain + not-retrieved = ${n(r1Sum)}${checkIcon(r1Sum, ta)}
    </div>

    ${r2HasRun ? `
    <div class="prisma-arrow">↓</div>

    <!-- Row: Round-2 full-text screening -->
    <div class="prisma-row">
      <div class="prisma-box" style="border-color:var(--green);">
        <div class="pbox-count" style="color:var(--green);">${n(p.final_included)}</div>
        <div class="pbox-label">Final Included</div>
      </div>
      <div class="prisma-side">
        <span style="color:var(--red);">← ${n(p.fulltext_r2_excluded)} excluded at round-2 full-text</span>
        ${p.fulltext_r2_uncertain > 0 ? `<br><span style="color:var(--orange);">← ${n(p.fulltext_r2_uncertain)} pending human review (round-2)</span>` : ''}
      </div>
    </div>
    <div style="font-size:11px; color:var(--text-muted); text-align:center; margin:-4px 0 4px;">
      r2-included + r2-excluded + r2-uncertain = ${n(r2Sum)}${checkIcon(r2Sum, p.fulltext_r1_passed ?? 0)}
    </div>
    ` : `
    <div style="font-size:11px; color:var(--text-muted); text-align:center; margin:8px 0;">
      ↓ Round-2 full-text screening not yet run
    </div>
    <div class="prisma-row">
      <div class="prisma-box" style="border-color:var(--green);">
        <div class="pbox-count" style="color:var(--green);">${n(p.final_included)}</div>
        <div class="pbox-label">Final Included (Round-1)</div>
      </div>
    </div>
    `}

    ${p.needs_human_verification > 0 ? `
    <div class="mt-3" style="color:var(--orange); font-size:12px;">
      ⚠ ${p.needs_human_verification} records pending human verification
      (title: ${p.needs_human_title_screening ?? 0},
       round-1 FT: ${p.needs_human_fulltext_screening ?? 0},
       round-2 FT: ${p.needs_human_round2_screening ?? 0},
       extraction: ${p.needs_human_extraction ?? 0})
    </div>` : ''}

  </div>
  `;
}

async function loadRecentRecords() {
  const data = await apiFetch('/records?page=1&page_size=10');
  const el = document.getElementById('recent-records');
  if (!data.records.length) {
    el.innerHTML = '<div class="empty-state">No records yet. Run import to begin.</div>';
    return;
  }
  el.innerHTML = data.records.map(r => recordRow(r)).join('');
}

function recordRow(r) {
  const pill = decisionPill(r.final_decision);
  return `
    <div class="record-row" onclick="openVerifyModal('${r.record_id}')">
      <div>
        <div class="record-title">${esc(r.title)}</div>
        <div class="record-meta">${r.authors || ''} · ${r.year || ''} · ${r.source_db || ''}</div>
      </div>
      <div>${pill}</div>
      <div style="font-size:11px; color:var(--text-muted);">${r.pipeline_stage || ''}</div>
      <div style="font-size:11px; color:var(--text-muted);">${r.agents_agree === false ? '⚡ Disagree' : ''}</div>
    </div>
  `;
}

function decisionPill(d) {
  const map = {
    'Included': 'pill-green',
    'Excluded': 'pill-red',
    'Needs Human Verification': 'pill-orange',
    'Full Text Needed': 'pill-blue',
  };
  return `<span class="pill ${map[d] || 'pill-gray'}">${d || 'Unknown'}</span>`;
}

// ─────────────────────────────────────────────
// HUMAN VERIFICATION
// ─────────────────────────────────────────────

async function loadVerifyQueue() {
  try {
    const data = await apiFetch('/records/uncertain/list');
    const el = document.getElementById('verify-list');

    if (!data.records.length) {
      el.innerHTML = '<div class="empty-state">🎉 No records need verification right now.</div>';
      return;
    }

    el.innerHTML = data.records.map(r => `
      <div class="verify-card" onclick="openVerifyModal('${r.record_id}')">
        <div>
          <div class="verify-card-title">${esc(r.title)}</div>
          <div class="verify-card-meta">${r.authors || ''} · ${r.year || ''} · ${r.source_db || ''}</div>
          ${r.uncertain_extraction_fields && r.uncertain_extraction_fields.length ?
            `<div class="verify-card-reason">⚠ Uncertain extraction fields: ${r.uncertain_extraction_fields.join(', ')}</div>` :
            `<div class="verify-card-reason">⚡ Agents disagree or low confidence</div>`
          }
        </div>
        <div>
          <button class="btn btn-sm btn-primary">Review →</button>
        </div>
      </div>
    `).join('');

    setBadge('badge-uncertain', data.count);
  } catch (e) {
    toast('error', 'Failed to load verification queue: ' + e.message);
  }
}

async function openVerifyModal(recordId) {
  currentRecordId = recordId;
  try {
    const rec = await apiFetch(`/records/${recordId}`);
    const modal = document.getElementById('verify-modal');

    // Title
    document.getElementById('modal-title').textContent =
      rec.dedup?.title || rec.raw?.title || 'Record';

    // Metadata
    const d = rec.dedup || rec.raw || {};
    document.getElementById('modal-metadata').innerHTML = `
      <span class="meta-key">Authors</span><span class="meta-val">${d.authors || '—'}</span>
      <span class="meta-key">Year</span><span class="meta-val">${d.year || '—'}</span>
      <span class="meta-key">Journal</span><span class="meta-val">${d.journal_venue || '—'}</span>
      <span class="meta-key">DOI</span><span class="meta-val">${d.doi || '—'}</span>
      <span class="meta-key">Source DB</span><span class="meta-val">${d.source_db || '—'}</span>
      <span class="meta-key">Stage</span><span class="meta-val">${rec.pipeline_stage || '—'}</span>
      <span class="meta-key">Decision</span><span class="meta-val">${rec.final_decision || '—'}</span>
    `;

    // Abstract
    document.getElementById('modal-abstract').textContent = d.abstract || '[No abstract]';

    // Agent decisions
    renderAgentDecisions(rec);

    // Extraction section
    renderExtractionSection(rec);

    modal.classList.remove('hidden');
  } catch (e) {
    toast('error', 'Failed to load record: ' + e.message);
  }
}

function renderAgentDecisions(rec) {
  const el = document.getElementById('modal-agent-decisions');
  const s = rec.screened;
  if (!s) { el.innerHTML = '<div class="agent-box">No screening data yet.</div>'; return; }

  const active = s.fulltext_screening || s.title_screening;
  if (!active) { el.innerHTML = '<div class="agent-box">No screening data yet.</div>'; return; }

  function agentBox(agent, label) {
    if (!agent) return '';
    const color = agent.decision === 'Included' ? 'var(--green)' :
                  agent.decision === 'Excluded' ? 'var(--red)' : 'var(--orange)';
    const pct = Math.round((agent.confidence || 0) * 100);
    return `
      <div class="agent-box">
        <div class="agent-box-header">${label} · <code>${agent.model_used}</code></div>
        <div class="agent-decision-label" style="color:${color}">${agent.decision}</div>
        <div class="confidence-bar">
          <div class="confidence-fill" style="width:${pct}%; background:${color}"></div>
        </div>
        <div style="font-size:11px; color:var(--text-dim); margin-bottom:6px;">Confidence: ${pct}%</div>
        <div class="agent-rationale">${esc(agent.rationale || '')}</div>
        ${agent.exclusion_code ? `<div class="pill pill-red mt-2">${agent.exclusion_code}</div>` : ''}
      </div>
    `;
  }

  el.innerHTML = agentBox(active.agent1, 'Agent 1') + agentBox(active.agent2, 'Agent 2');
}

function renderExtractionSection(rec) {
  const section = document.getElementById('modal-extraction-section');
  const grid = document.getElementById('modal-extraction-fields');

  if (!rec.extracted || !rec.extracted.extraction_final) {
    section.classList.add('hidden');
    return;
  }

  section.classList.remove('hidden');
  const ef = rec.extracted.extraction_final;
  const uncertain = ef.uncertain_fields || [];

  // Key fields to show
  const fields = [
    ['model_name', 'Model Name', 'text'],
    ['model_type', 'Model Type', 'text'],
    ['workflow_structure', 'Workflow Structure', 'text'],
    ['analytic_task', 'Analytic Task', 'text'],
    ['qualitative_approach', 'Qualitative Approach', 'text'],
    ['human_comparison', 'Human Comparison', 'boolean'],
    ['domain', 'Domain', 'text'],
    ['data_type', 'Data Type', 'text'],
    ['key_findings', 'Key Findings', 'textarea'],
    ['limitations_reported', 'Limitations', 'textarea'],
  ];

  grid.innerHTML = fields.map(([field, label, type]) => {
    const val = ef[field];
    const isUncertain = uncertain.includes(field);
    const displayVal = Array.isArray(val) ? val.join(', ') : (val ?? '');
    const inputEl = type === 'textarea'
      ? `<textarea id="ef-${field}" class="form-control" rows="2">${esc(String(displayVal))}</textarea>`
      : `<input type="text" id="ef-${field}" class="form-control" value="${esc(String(displayVal))}" />`;

    return `
      <div class="extraction-field ${isUncertain ? 'uncertain' : ''}">
        <label>${label} ${isUncertain ? '⚠' : ''}</label>
        ${inputEl}
      </div>
    `;
  }).join('');
}

function closeVerifyModal() {
  document.getElementById('verify-modal').classList.add('hidden');
  currentRecordId = null;
}

async function submitHumanDecision(decision) {
  const rationale = document.getElementById('human-rationale').value.trim();
  if (!rationale) {
    toast('error', 'Please provide a rationale before submitting.');
    return;
  }

  const reviewer = document.getElementById('human-reviewer').value.trim() || 'Reviewer';

  // Collect extraction corrections if visible
  let corrections = null;
  const section = document.getElementById('modal-extraction-section');
  if (!section.classList.contains('hidden')) {
    corrections = {};
    const fields = ['model_name', 'model_type', 'workflow_structure', 'analytic_task',
                    'qualitative_approach', 'domain', 'data_type', 'key_findings', 'limitations_reported'];
    fields.forEach(f => {
      const el = document.getElementById(`ef-${f}`);
      if (el) corrections[f] = el.value;
    });
  }

  try {
    await apiPost(`/records/${currentRecordId}/verify`, {
      decision,
      rationale,
      reviewer,
      corrections,
    });
    toast('success', `Decision saved: ${decision}`);
    closeVerifyModal();
    loadVerifyQueue();
    loadDashboard();
  } catch (e) {
    toast('error', 'Failed to save decision: ' + e.message);
  }
}

// ─────────────────────────────────────────────
// FULL TEXT NEEDED
// ─────────────────────────────────────────────

async function loadFulltextNeeded() {
  try {
    const data = await apiFetch('/records/fulltext-needed/list');
    const el = document.getElementById('fulltext-table');
    const select = document.getElementById('fulltext-record-select');

    setBadge('badge-fulltext', data.count);
    const retryBadge = document.getElementById('ft-retry-count');
    if (retryBadge) retryBadge.textContent = data.count;

    // Populate select
    select.innerHTML = '<option value="">— Select record —</option>' +
      data.records.map(r =>
        `<option value="${r.record_id}">${esc(r.title.substring(0, 60))} (${r.year})</option>`
      ).join('');

    if (!data.records.length) {
      el.innerHTML = '<div class="empty-state">No papers need full text right now.</div>';
      return;
    }

    el.innerHTML = `
      <div class="evidence-table-wrap">
        <table class="evidence-table">
          <thead>
            <tr>
              <th>Title</th><th>Authors</th><th>Year</th><th>DOI</th><th>Source</th>
            </tr>
          </thead>
          <tbody>
            ${data.records.map(r => `
              <tr>
                <td title="${esc(r.title)}">${esc(r.title.substring(0, 60))}${r.title.length > 60 ? '…' : ''}</td>
                <td>${esc(r.authors || '')}</td>
                <td>${r.year || ''}</td>
                <td><code>${r.doi || '—'}</code></td>
                <td>${r.source_db || ''}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  } catch (e) {
    toast('error', 'Failed to load full-text list: ' + e.message);
  }
}

async function uploadSinglePDF() {
  const recordId = document.getElementById('fulltext-record-select').value;
  const fileInput = document.getElementById('single-pdf-input');

  if (!recordId) { toast('error', 'Select a record first.'); return; }
  if (!fileInput.files.length) { toast('error', 'Select a PDF file first.'); return; }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('record_id', recordId);

  try {
    const res = await fetch(`${API}/pdfs/upload`, { method: 'POST', body: formData });
    if (!res.ok) throw new Error((await res.json()).detail);
    toast('success', 'PDF uploaded successfully!');
    fileInput.value = '';
    loadFulltextNeeded();
  } catch (e) {
    toast('error', 'Upload failed: ' + e.message);
  }
}

async function handlePDFUpload(event) {
  const files = Array.from(event.target.files);
  if (!files.length) return;

  const formData = new FormData();
  files.forEach(f => formData.append('files', f));

  try {
    const res = await fetch(`${API}/pdfs/upload-batch`, { method: 'POST', body: formData });
    const data = await res.json();
    toast('success', `Uploaded ${data.uploaded} PDFs`);
    loadFulltextNeeded();
  } catch (e) {
    toast('error', 'Batch upload failed: ' + e.message);
  }
}

// Drag and drop
const dropZone = document.getElementById('upload-zone');
if (dropZone) {
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = 'var(--accent)'; });
  dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = ''; });
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.style.borderColor = '';
    const input = document.getElementById('pdf-file-input');
    input.files = e.dataTransfer.files;
    handlePDFUpload({ target: input });
  });
}

// ─────────────────────────────────────────────
// INCLUDED / EXCLUDED PANELS
// ─────────────────────────────────────────────

async function loadIncluded() {
  try {
    const data = await apiFetch('/export/evidence-table');
    const el = document.getElementById('included-table');

    if (!data.rows.length) {
      el.innerHTML = '<div class="empty-state">No included studies yet.</div>';
      return;
    }

    const cols = ['study_id', 'title', 'authors', 'year', 'domain',
                  'model_name', 'analytic_task', 'workflow_structure', 'qa_score'];

    el.innerHTML = `
      <div class="evidence-table-wrap">
        <table class="evidence-table">
          <thead>
            <tr>${cols.map(c => `<th>${c.replace(/_/g, ' ')}</th>`).join('')}</tr>
          </thead>
          <tbody>
            ${data.rows.map(r => `
              <tr>
                ${cols.map(c => `<td title="${esc(String(r[c] ?? ''))}">${esc(String(r[c] ?? ''))}</td>`).join('')}
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  } catch (e) {
    toast('error', 'Failed to load included studies: ' + e.message);
  }
}

async function loadExcluded() {
  try {
    const data = await apiFetch('/records?decision=Excluded&page_size=100');
    const el = document.getElementById('excluded-table');

    if (!data.records.length) {
      el.innerHTML = '<div class="empty-state">No excluded records yet.</div>';
      return;
    }

    el.innerHTML = data.records.map(r => recordRow(r)).join('');
  } catch (e) {
    toast('error', 'Failed to load excluded records: ' + e.message);
  }
}

// ─────────────────────────────────────────────
// PIPELINE
// ─────────────────────────────────────────────

function logLine(msg) {
  const el = document.getElementById('pipeline-log');
  const ts = new Date().toLocaleTimeString();
  el.textContent += `[${ts}] ${msg}\n`;
  el.scrollTop = el.scrollHeight;
}

function clearLog() {
  document.getElementById('pipeline-log').textContent = '';
}

async function runStage(stage, limitInputId = null) {
  const limit = limitInputId
    ? (parseInt(document.getElementById(limitInputId)?.value) || null)
    : null;

  logLine(`Starting stage: ${stage}${limit ? ` (limit: ${limit})` : ''} ...`);

  try {
    const res = await apiPost('/pipeline/run', { stage, limit });
    logLine(`Stage ${stage} started in background. Status: ${res.status}`);
    toast('info', `Pipeline stage "${stage}" running...`);

    // For fulltext_download use the dedicated status endpoint for meaningful progress
    if (stage === 'fulltext_download') {
      const interval = setInterval(async () => {
        try {
          const status = await apiFetch('/fulltext/download-status');
          const r = status.last_run || {};
          const total = r.total_candidates ?? '?';
          const done = r.processed ?? 0;
          const dl = r.auto_downloaded ?? 0;
          if (!status.running && r.completed_at) {
            clearInterval(interval);
            logLine(`Download complete: ${dl} auto-downloaded, ${r.manual_needed ?? 0} need manual upload (${total} total candidates)`);
            loadDashboard();
            toast('success', `Done. ${dl} downloaded, ${r.manual_needed ?? 0} manual.`);
          } else {
            logLine(`Downloading: ${done}/${total} processed, ${dl} retrieved so far…`);
            loadDashboard();
          }
        } catch {}
      }, 5000);
      setTimeout(() => clearInterval(interval), 40 * 60 * 1000); // 40min max
      return;
    }

    // Generic stage polling
    const interval = setInterval(async () => {
      try {
        const status = await apiFetch('/pipeline/status');
        const p = status.prisma_counts;
        logLine(`Status: identified=${p.identified}, title_included=${p.title_abstract_included}, uncertain=${p.needs_human_verification}`);
        loadDashboard();
      } catch {}
    }, 3000);
    setTimeout(() => clearInterval(interval), 120000); // stop polling after 2min
  } catch (e) {
    logLine(`ERROR: ${e.message}`);
    toast('error', `Pipeline error: ${e.message}`);
  }
}

// ─────────────────────────────────────────────
// MODEL CONFIG
// ─────────────────────────────────────────────

async function loadConfig() {
  try {
    const cfg = await apiFetch('/config/models');
    setSelectValue('cfg-title-screening', cfg.model_title_screening);
    setSelectValue('cfg-fulltext-screening', cfg.model_fulltext_screening);
    setSelectValue('cfg-extraction', cfg.model_extraction);
    setSelectValue('cfg-qa-assessment', cfg.model_qa_assessment);
    setSelectValue('cfg-agent2-screening', cfg.model_agent2_screening);
    setSelectValue('cfg-agent2-extraction', cfg.model_agent2_extraction);
    const ci = document.getElementById('cfg-confidence');
    if (ci) ci.value = cfg.confidence_threshold;
  } catch (e) {
    console.warn('Config load error:', e);
  }
}

function configChanged() {
  // Visual indicator that config has unsaved changes
  const status = document.getElementById('config-status');
  if (status) {
    status.textContent = 'Unsaved changes';
    status.className = 'status-msg';
    status.classList.remove('hidden');
  }
}

async function saveConfig() {
  const updates = {
    model_title_screening: getVal('cfg-title-screening'),
    model_fulltext_screening: getVal('cfg-fulltext-screening'),
    model_extraction: getVal('cfg-extraction'),
    model_qa_assessment: getVal('cfg-qa-assessment'),
    model_agent2_screening: getVal('cfg-agent2-screening'),
    model_agent2_extraction: getVal('cfg-agent2-extraction'),
    confidence_threshold: parseFloat(getVal('cfg-confidence')) || 0.8,
  };

  try {
    await apiPatch('/config/models', updates);
    const status = document.getElementById('config-status');
    status.textContent = '✓ Configuration saved';
    status.className = 'status-msg success';
    status.classList.remove('hidden');
    toast('success', 'Model configuration saved');
    setTimeout(() => status.classList.add('hidden'), 3000);
  } catch (e) {
    toast('error', 'Failed to save config: ' + e.message);
  }
}

async function saveApiKey() {
  const key = document.getElementById('api-key-input').value.trim();
  if (!key) { toast('error', 'Enter an API key first.'); return; }
  // Write to backend (which writes to .env)
  try {
    await apiPatch('/config/models', { openai_api_key: key });
    document.getElementById('api-key-input').value = '';
    toast('success', 'API key saved to .env');
  } catch (e) {
    toast('error', 'Failed to save API key: ' + e.message);
  }
}

// ─────────────────────────────────────────────
// EXPORT
// ─────────────────────────────────────────────

async function loadExportPreview() {
  try {
    const p = await apiFetch('/export/prisma');
    const el = document.getElementById('prisma-preview');
    el.innerHTML = `
      <table>
        <thead><tr><th>PRISMA Stage</th><th>Count</th></tr></thead>
        <tbody>
          ${Object.entries(p).map(([k, v]) =>
            `<tr><td>${k.replace(/_/g, ' ')}</td><td><strong>${v}</strong></td></tr>`
          ).join('')}
        </tbody>
      </table>
    `;
  } catch (e) {
    console.warn('Export preview error:', e);
  }
}

async function exportCSV() {
  window.open(`${API}/export/evidence-table/csv`, '_blank');
}

async function downloadExport(type) {
  const urlMap = {
    csv: `${API}/export/evidence-table/csv`,
    json: `${API}/export/evidence-table`,
    prisma: `${API}/export/prisma`,
    full: `${API}/export/all-records`,
  };
  window.open(urlMap[type], '_blank');
}

// ─────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────

function refreshAll() {
  loadDashboard();
  loadVerifyQueue();
  toast('info', 'Refreshed');
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setBadge(id, count) {
  const el = document.getElementById(id);
  if (!el) return;
  if (count > 0) {
    el.textContent = count;
    el.style.display = '';
  } else {
    el.style.display = 'none';
  }
}

function setSelectValue(id, value) {
  const el = document.getElementById(id);
  if (!el || !value) return;
  // Add as option if not present
  let found = false;
  for (const opt of el.options) {
    if (opt.value === value) { found = true; break; }
  }
  if (!found) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = value;
    el.appendChild(opt);
  }
  el.value = value;
}

function getVal(id) {
  const el = document.getElementById(id);
  return el ? el.value : '';
}

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function toast(type, message) {
  const container = document.getElementById('toast-container');
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  const icons = { success: '✓', error: '✕', info: 'ℹ' };
  el.innerHTML = `<span>${icons[type] || ''}</span> ${esc(message)}`;
  container.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

// ─────────────────────────────────────────────
// SETUP WIZARD
// ─────────────────────────────────────────────

let _currentStep = 1;

function goStep(n) {
  document.querySelectorAll('.setup-step').forEach(s => s.classList.remove('active'));
  const target = document.getElementById(`step-${n}`);
  if (target) target.classList.add('active');

  document.querySelectorAll('.dot').forEach((d, i) => {
    d.classList.toggle('active', i + 1 <= n);
  });
  _currentStep = n;
}

function toggleSetupKeyVisibility() {
  const inp = document.getElementById('setup-api-key');
  inp.type = inp.type === 'password' ? 'text' : 'password';
}

async function saveSetupKey() {
  const key = document.getElementById('setup-api-key').value.trim();
  const baseUrl = document.getElementById('setup-base-url').value.trim();
  const statusEl = document.getElementById('setup-key-status');

  if (!key) { showSetupStatus('setup-key-status', 'error', 'Please enter your API key.'); return; }
  if (!key.startsWith('sk-')) { showSetupStatus('setup-key-status', 'error', 'API key must start with sk-'); return; }

  showSetupStatus('setup-key-status', '', 'Saving and testing key...');

  try {
    // Save key
    await apiPost('/config/api-key', { api_key: key, base_url: baseUrl || null });

    // Test key
    showSetupStatus('setup-key-status', '', 'Testing API connection...');
    const test = await apiFetch('/config/test-api-key');
    showSetupStatus('setup-key-status', 'success', `✓ API key valid. Connected to OpenAI.`);

    setTimeout(() => goStep(2), 1000);
  } catch (e) {
    showSetupStatus('setup-key-status', 'error', `Error: ${e.message}`);
  }
}

async function saveSetupModels() {
  const updates = {
    model_title_screening: getVal('s-title-a1'),
    model_agent2_screening: getVal('s-title-a2'),
    model_fulltext_screening: getVal('s-fulltext-a1'),
    model_extraction: getVal('s-extract-a1'),
    model_agent2_extraction: getVal('s-extract-a2'),
    confidence_threshold: parseFloat(getVal('s-confidence')) || 0.80,
  };

  try {
    await apiPatch('/config/models', updates);
    showSetupStatus('setup-model-status', 'success', '✓ Models saved.');
    setTimeout(() => goStep(3), 800);
  } catch (e) {
    showSetupStatus('setup-model-status', 'error', `Error: ${e.message}`);
  }
}

function finishSetup() {
  document.getElementById('setup-wizard').classList.add('hidden');
  loadDashboard();
}

function showSetupStatus(elId, type, msg) {
  const el = document.getElementById(elId);
  if (!el) return;
  el.textContent = msg;
  el.className = `status-msg ${type}`;
  el.classList.remove('hidden');
}

async function checkSetupRequired() {
  try {
    const status = await apiFetch('/config/setup-status');
    if (!status.ready) {
      document.getElementById('setup-wizard').classList.remove('hidden');
      goStep(1);
    } else {
      document.getElementById('setup-wizard').classList.add('hidden');
      loadDashboard();
    }
  } catch {
    // If API is not up yet, just show dashboard
    loadDashboard();
  }
}

// ─────────────────────────────────────────────
// FULL-TEXT AUTO-DOWNLOAD
// ─────────────────────────────────────────────

async function runFulltextDownload() {
  const email = document.getElementById('ft-contact-email')?.value?.trim() || '';
  const progressEl = document.getElementById('ft-download-progress');
  const fillEl = document.getElementById('ft-progress-fill');
  const labelEl = document.getElementById('ft-progress-label');
  const resultEl = document.getElementById('ft-download-result');

  if (progressEl) progressEl.classList.remove('hidden');
  if (labelEl) labelEl.textContent = 'Starting auto-download…';
  if (fillEl) fillEl.style.width = '5%';
  if (resultEl) resultEl.innerHTML = '';

  try {
    await apiPost('/fulltext/download', { email: email || null, limit: null });
    toast('info', 'Full-text auto-download started in background…');

    // Poll every 5s until complete
    const interval = setInterval(async () => {
      try {
        const status = await apiFetch('/fulltext/download-status');
        if (!status.running && status.last_run && status.last_run.completed_at) {
          clearInterval(interval);
          const r = status.last_run;
          if (fillEl) fillEl.style.width = '100%';
          if (labelEl) labelEl.textContent = 'Complete';
          if (resultEl) resultEl.innerHTML = `
            <div class="ft-result-box">
              <div class="ft-result-row">
                <span class="ft-result-stat ft-ok">✓ ${r.auto_downloaded ?? 0} auto-downloaded</span>
                <span class="ft-result-stat ft-warn">⚠ ${r.manual_needed ?? 0} need manual upload</span>
              </div>
              <p class="config-hint mt-2">Download the CSV below → retrieve PDFs → upload in Step 3.</p>
            </div>`;
          loadFulltextNeeded();
          loadDashboard();
          toast('success', `Done. ${r.auto_downloaded ?? 0} downloaded, ${r.manual_needed ?? 0} manual.`);
        } else if (status.running) {
          if (labelEl) labelEl.textContent = 'Downloading open-access PDFs… (may take several minutes)';
          const cur = parseFloat((fillEl?.style.width || '5').replace('%', '')) || 5;
          if (fillEl) fillEl.style.width = Math.min(90, cur + 1.5) + '%';
        }
      } catch {}
    }, 5000);
    setTimeout(() => clearInterval(interval), 40 * 60 * 1000); // 40min max
  } catch (e) {
    toast('error', 'Failed to start download: ' + e.message);
    if (progressEl) progressEl.classList.add('hidden');
  }
}

async function retryFulltextDownload() {
  const progressEl = document.getElementById('ft-retry-progress');
  const fillEl = document.getElementById('ft-retry-fill');
  const labelEl = document.getElementById('ft-retry-label');
  const resultEl = document.getElementById('ft-retry-result');

  if (progressEl) progressEl.classList.remove('hidden');
  if (labelEl) labelEl.textContent = 'Starting retry (arXiv · CORE · Direct URL)…';
  if (fillEl) fillEl.style.width = '5%';
  if (resultEl) resultEl.innerHTML = '';

  try {
    await apiPost('/fulltext/retry', {});
    toast('info', 'Retry started — trying arXiv, CORE, and direct URLs…');

    const interval = setInterval(async () => {
      try {
        const status = await apiFetch('/fulltext/retry-status');
        const r = status.last_run || {};
        const total = r.total_candidates ?? '?';
        const done = r.processed ?? 0;
        const dl = r.auto_downloaded ?? 0;
        if (!status.running && r.completed_at) {
          clearInterval(interval);
          const still = r.still_needed ?? status.still_needed_count ?? 0;
          if (fillEl) fillEl.style.width = '100%';
          if (labelEl) labelEl.textContent = 'Retry complete';
          if (resultEl) resultEl.innerHTML = `
            <div class="ft-result-box">
              <div class="ft-result-row">
                <span class="ft-result-stat ft-ok">✓ ${dl} newly retrieved</span>
                <span class="ft-result-stat ft-warn">⚠ ${still} still need manual upload</span>
              </div>
              ${still > 0 ? '<p class="config-hint mt-2">Download the updated CSV below → retrieve remaining PDFs → upload in Step 3.</p>' : '<p class="config-hint mt-2 text-ok">All records retrieved!</p>'}
            </div>`;
          // Update retry count badge
          const badge = document.getElementById('ft-retry-count');
          if (badge) badge.textContent = still;
          loadDashboard();
          toast('success', `Retry done. ${dl} newly retrieved, ${still} still manual.`);
        } else if (status.running) {
          if (labelEl) labelEl.textContent = `Retrying: ${done}/${total} processed, ${dl} newly retrieved…`;
          const cur = parseFloat((fillEl?.style.width || '5').replace('%', '')) || 5;
          if (fillEl) fillEl.style.width = Math.min(90, cur + 1.5) + '%';
        }
      } catch {}
    }, 5000);
    setTimeout(() => clearInterval(interval), 40 * 60 * 1000);
  } catch (e) {
    toast('error', 'Retry failed: ' + e.message);
    if (progressEl) progressEl.classList.add('hidden');
  }
}

function downloadManualList() {
  window.open(`${API}/fulltext/manual-list/csv`, '_blank');
}

async function savePrismaSnapshot() {
  const label = prompt('Label for this PRISMA snapshot:', 'after_human_verification') || 'manual';
  try {
    const res = await apiPost('/prisma/snapshot', { label });
    toast('success', `PRISMA snapshot saved: "${res.snapshot.label}"`);
  } catch (e) {
    toast('error', 'Failed to save snapshot: ' + e.message);
  }
}

// ─────────────────────────────────────────────
// SECOND-PASS REVIEW
// ─────────────────────────────────────────────

let secondPassPage = 1;
let secondPassPendingId = null;   // record_id awaiting exclude submission
let secondPassTotalReviewed = 0;
let secondPassTotal = 0;

async function loadSecondPass(page = 1) {
  secondPassPage = page;
  const filter = document.getElementById('sp-filter')?.value || 'all';
  const reviewedParam = filter === 'no' ? '&reviewed=no' : '';

  try {
    const data = await apiFetch(`/records/second-pass/list?page=${page}&page_size=50${reviewedParam}`);
    secondPassTotal = data.total;

    // Update badge (total unreviewed)
    const unreviewed = data.records.filter(r => !r.human_verified).length;
    setBadge('badge-second-pass', data.total);

    // Progress bar
    const reviewed = await apiFetch('/records/second-pass/list?page=1&page_size=1&reviewed=no');
    const unreviewedTotal = reviewed.total;
    const doneCount = secondPassTotal - unreviewedTotal;
    const pct = secondPassTotal > 0 ? Math.round((doneCount / secondPassTotal) * 100) : 0;
    const label = document.getElementById('sp-progress-label');
    const fill = document.getElementById('sp-progress-fill');
    if (label) label.textContent = `${doneCount} / ${secondPassTotal} reviewed (${pct}%)`;
    if (fill) fill.style.width = pct + '%';

    // Render cards
    const el = document.getElementById('sp-list');
    if (!data.records.length) {
      el.innerHTML = '<div class="empty-state">No included records found. Run title screening first.</div>';
      document.getElementById('sp-pagination').innerHTML = '';
      return;
    }

    el.innerHTML = data.records.map(r => renderSecondPassCard(r)).join('');

    // Pagination
    const totalPages = Math.ceil(data.total / 50);
    renderSecondPassPagination(page, totalPages);
  } catch (e) {
    toast('error', 'Failed to load second-pass records: ' + e.message);
  }
}

function renderSecondPassCard(r) {
  const a1pct = Math.round((r.agent1_confidence || 0) * 100);
  const a2pct = Math.round((r.agent2_confidence || 0) * 100);
  const a1color = r.agent1_decision === 'Included' ? 'var(--green)' :
                  r.agent1_decision === 'Excluded' ? 'var(--red)' : 'var(--orange)';
  const a2color = r.agent2_decision === 'Included' ? 'var(--green)' :
                  r.agent2_decision === 'Excluded' ? 'var(--red)' : 'var(--orange)';
  const abstractPreview = r.abstract ? r.abstract.substring(0, 320) + (r.abstract.length > 320 ? '…' : '') : '[No abstract]';
  const verifiedBadge = r.human_verified
    ? '<span class="pill pill-green" style="font-size:10px">✓ Reviewed</span>' : '';

  return `
    <div class="sp-card ${r.human_verified ? 'sp-card-reviewed' : ''}" id="sp-card-${r.record_id}">
      <div class="sp-card-header">
        <div class="sp-card-title">${esc(r.title)}</div>
        <div class="sp-card-meta">${esc(r.authors || '')} · ${r.year || ''} · <em>${esc(r.journal || r.source_db || '')}</em> ${verifiedBadge}</div>
      </div>

      <div class="sp-card-abstract" id="abstract-${r.record_id}">
        <span class="sp-abstract-text">${esc(abstractPreview)}</span>
        ${r.abstract && r.abstract.length > 320
          ? `<button class="sp-expand-btn" onclick="toggleAbstract('${r.record_id}', ${JSON.stringify(esc(r.abstract))})">Show more</button>` : ''}
      </div>

      <div class="sp-agents">
        <div class="sp-agent-section-label">First-Pass AI Screen</div>
        <div class="sp-agent-box">
          <div class="sp-agent-label" style="color:${a1color}">Agent 1: ${esc(r.agent1_decision)} (${a1pct}%)</div>
          <div class="confidence-bar"><div class="confidence-fill" style="width:${a1pct}%; background:${a1color}"></div></div>
          <div class="sp-agent-rationale">${esc(r.agent1_rationale || '—')}</div>
        </div>
        <div class="sp-agent-box">
          <div class="sp-agent-label" style="color:${a2color}">Agent 2: ${esc(r.agent2_decision)} (${a2pct}%)</div>
          <div class="confidence-bar"><div class="confidence-fill" style="width:${a2pct}%; background:${a2color}"></div></div>
          <div class="sp-agent-rationale">${esc(r.agent2_rationale || '—')}</div>
        </div>
        ${r.sp_done ? renderSecondPassAgentBoxes(r) : spNotRunBanner()}
      </div>

      <div class="sp-card-actions">
        <button class="btn btn-success btn-sm" onclick="secondPassKeep('${r.record_id}')">✓ Keep</button>
        <button class="btn btn-danger btn-sm" onclick="secondPassExclude('${r.record_id}')">✗ Exclude</button>
        <button class="btn btn-ghost btn-sm" onclick="openVerifyModal('${r.record_id}')" title="Open full review modal">Full Review</button>
        ${r.doi ? `<a class="btn btn-ghost btn-sm" href="https://doi.org/${r.doi}" target="_blank">DOI ↗</a>` : ''}
      </div>
    </div>
  `;
}

function spNotRunBanner() {
  return `<div class="sp-agent-section-label sp-not-run" style="grid-column:1/-1">
    ⚠ Second-pass AI screening not yet run — go to <strong>Run Pipeline → Stage 3b</strong>
  </div>`;
}

function renderSecondPassAgentBoxes(r) {
  const sp1pct = Math.round((r.sp_agent1_confidence || 0) * 100);
  const sp2pct = Math.round((r.sp_agent2_confidence || 0) * 100);
  const sp1color = r.sp_agent1_decision === 'Included' ? 'var(--green)' :
                   r.sp_agent1_decision === 'Excluded' ? 'var(--red)' : 'var(--orange)';
  const sp2color = r.sp_agent2_decision === 'Included' ? 'var(--green)' :
                   r.sp_agent2_decision === 'Excluded' ? 'var(--red)' : 'var(--orange)';
  return `
    <div class="sp-agent-section-label sp-second-pass-label" style="grid-column:1/-1">Second-Pass AI Screen (Strict)</div>
    <div class="sp-agent-box sp-agent-box-strict">
      <div class="sp-agent-label" style="color:${sp1color}">
        Agent 1: ${esc(r.sp_agent1_decision || '—')} (${sp1pct}%)
        ${r.sp_agent1_ec ? `<span class="pill pill-red" style="font-size:10px;margin-left:4px">${esc(r.sp_agent1_ec)}</span>` : ''}
      </div>
      <div class="confidence-bar"><div class="confidence-fill" style="width:${sp1pct}%; background:${sp1color}"></div></div>
      <div class="sp-agent-rationale">${esc(r.sp_agent1_rationale || '—')}</div>
    </div>
    <div class="sp-agent-box sp-agent-box-strict">
      <div class="sp-agent-label" style="color:${sp2color}">
        Agent 2: ${esc(r.sp_agent2_decision || '—')} (${sp2pct}%)
        ${r.sp_agent2_ec ? `<span class="pill pill-red" style="font-size:10px;margin-left:4px">${esc(r.sp_agent2_ec)}</span>` : ''}
      </div>
      <div class="confidence-bar"><div class="confidence-fill" style="width:${sp2pct}%; background:${sp2color}"></div></div>
      <div class="sp-agent-rationale">${esc(r.sp_agent2_rationale || '—')}</div>
    </div>
  `;
}

function toggleAbstract(recordId, fullText) {
  const el = document.querySelector(`#abstract-${recordId} .sp-abstract-text`);
  const btn = document.querySelector(`#abstract-${recordId} .sp-expand-btn`);
  if (!el) return;
  if (btn.textContent === 'Show more') {
    el.textContent = fullText;
    btn.textContent = 'Show less';
  } else {
    el.textContent = fullText.substring(0, 320) + '…';
    btn.textContent = 'Show more';
  }
}

async function secondPassKeep(recordId) {
  try {
    await apiPost(`/records/${recordId}/verify`, {
      decision: 'Included',
      rationale: 'Human confirmed after second-pass review.',
      reviewer: document.getElementById('human-reviewer')?.value?.trim() || 'Reviewer',
    });
    // Mark the card as reviewed visually
    const card = document.getElementById(`sp-card-${recordId}`);
    if (card) {
      card.classList.add('sp-card-reviewed');
      card.querySelector('.sp-card-actions').innerHTML =
        '<span class="pill pill-green">✓ Kept</span>';
    }
    toast('success', 'Kept as included.');
    loadDashboard();
  } catch (e) {
    toast('error', 'Failed to save decision: ' + e.message);
  }
}

function secondPassExclude(recordId) {
  secondPassPendingId = recordId;
  document.getElementById('sp-exclude-rationale').value = '';
  document.getElementById('sp-exclude-form').classList.remove('hidden');
}

function cancelSecondPassExclude() {
  secondPassPendingId = null;
  document.getElementById('sp-exclude-form').classList.add('hidden');
}

async function submitSecondPassExclude() {
  const rationale = document.getElementById('sp-exclude-rationale').value.trim();
  if (!rationale) { toast('error', 'Please provide a rationale.'); return; }
  const ec = document.getElementById('sp-ec-code').value;

  try {
    await apiPost(`/records/${secondPassPendingId}/verify`, {
      decision: 'Excluded',
      rationale: `[${ec}] ${rationale}`,
      reviewer: document.getElementById('human-reviewer')?.value?.trim() || 'Reviewer',
    });
    const card = document.getElementById(`sp-card-${secondPassPendingId}`);
    if (card) {
      card.classList.add('sp-card-reviewed', 'sp-card-excluded');
      card.querySelector('.sp-card-actions').innerHTML =
        `<span class="pill pill-red">✗ Excluded · ${esc(ec)}</span>`;
    }
    cancelSecondPassExclude();
    toast('success', `Excluded (${ec}).`);
    loadDashboard();
  } catch (e) {
    toast('error', 'Failed to save decision: ' + e.message);
  }
}

function renderSecondPassPagination(page, totalPages) {
  const el = document.getElementById('sp-pagination');
  if (totalPages <= 1) { el.innerHTML = ''; return; }
  const pages = [];
  if (page > 1) pages.push(`<button class="btn btn-ghost btn-sm" onclick="loadSecondPass(${page - 1})">← Prev</button>`);
  pages.push(`<span class="sp-page-info">Page ${page} / ${totalPages}</span>`);
  if (page < totalPages) pages.push(`<button class="btn btn-ghost btn-sm" onclick="loadSecondPass(${page + 1})">Next →</button>`);
  el.innerHTML = pages.join('');
}

async function markAllIncludedForReview() {
  if (!confirm('This will move all 1 038 AI-included records into the Human Verification queue. Continue?')) return;
  try {
    const res = await apiPost('/pipeline/run', { stage: 'mark_included_for_review' });
    toast('success', `${res.stats?.updated ?? '?'} records queued for human verification.`);
    loadDashboard();
    loadVerifyQueue();
    loadSecondPass(1);
  } catch (e) {
    toast('error', 'Failed: ' + e.message);
  }
}

// ─────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  checkSetupRequired();
});
