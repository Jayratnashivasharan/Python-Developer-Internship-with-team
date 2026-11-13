(function () {
  const h = GA.h, num = GA.num;
  const root = document.getElementById('job-root');
  const id = root.dataset.id;
  let job = JSON.parse(document.getElementById('job-data').textContent);
  const $ = i => document.getElementById(i);
  const STAGES = {
    queued: 'Waiting in queue…', simulating: 'Simulating genome and reads…', indexing: 'Building BWT/FM-index…',
    aligning: 'Aligning reads…', calling: 'Calling variants and writing report…',
    benchmarking: 'Running benchmarks (this can take a minute)…', done: 'Completed', failed: 'Failed'
  };
  let resultsFor = null;

  /* ---------------------------------------------------------- live view */
  function rateNow(hist) {
    if (hist.length < 2) return 0;
    const b = hist[hist.length - 1], a = hist[Math.max(0, hist.length - 6)];
    const dt = b[0] - a[0];
    return dt > 0 ? (b[1] - a[1]) / dt : 0;
  }
  function tile(label, value, sub) {
    return h('div', {class: 'stat'}, h('div', {class: 'stat-label', text: label}),
      h('div', {class: 'stat-value', text: value}), sub ? h('div', {class: 'stat-sub', text: sub}) : null);
  }
  function renderLive() {
    const box = $('live'), L = job.live;
    if (!L || job.stage !== 'aligning') { box.hidden = true; return; }
    const busy = L.elapsed_s > 0 ? 100 * L.kernel_seconds / L.elapsed_s : 0;
    box.replaceChildren(
      tile('Reads processed', num(L.reads_total), L.percent + '% of file'),
      tile('Throughput now', num(rateNow(job.history || [])) + ' /s', 'avg ' + num(L.reads_per_sec) + ' reads/s'),
      tile('Seeded → aligner', num(L.reads_seeded), num(L.reads_filtered) + ' filtered'),
      tile('Aligned', num(L.reads_aligned), num(L.reads_lowscore) + ' low score'),
      tile('Backend', (L.backend || '').toUpperCase(), L.device || ''),
      tile('Kernel busy', busy.toFixed(0) + '%', 'queue depth ' + L.queue_depth),
      tile('Batches', num(L.batches), 'GPU/CPU mismatches ' + L.gpu_cpu_mismatch),
      tile('Variant events', num(L.variant_events), 'before calling'));
    box.hidden = false;
  }
  function drawChart() {
    const hist = job.history || [], wrap = $('chart-wrap');
    if (hist.length < 3 || job.kind !== 'align') { wrap.hidden = true; return; }
    wrap.hidden = false;
    const c = $('chart'), dpr = window.devicePixelRatio || 1, W = c.clientWidth || 600, H = 140;
    c.style.height = H + 'px'; c.width = W * dpr; c.height = H * dpr;
    const ctx = c.getContext('2d'); if (!ctx) return; ctx.scale(dpr, dpr);
    const pts = [];
    for (let i = 1; i < hist.length; i++) {
      const dt = hist[i][0] - hist[i - 1][0];
      if (dt > 0) pts.push([hist[i][0], (hist[i][1] - hist[i - 1][1]) / dt]);
    }
    if (pts.length < 2) return;
    const sm = pts.map((p, i) => {
      let s = 0, n = 0;
      for (let k = Math.max(0, i - 2); k <= i; k++) { s += pts[k][1]; n++; }
      return [p[0], s / n];
    });
    const css = getComputedStyle(document.body);
    const col = css.getPropertyValue('--primary').trim() || '#0f766e', grid = css.getPropertyValue('--border').trim() || '#ccc';
    const mut = css.getPropertyValue('--muted').trim() || '#777';
    const maxY = Math.max(1, ...sm.map(p => p[1])) * 1.15, minX = sm[0][0], maxX = sm[sm.length - 1][0] || 1;
    const pl = 46, pb = 18, pt = 6, pr = 6;
    const X = x => pl + (x - minX) / Math.max(1e-9, maxX - minX) * (W - pl - pr);
    const Y = y => H - pb - (y / maxY) * (H - pb - pt);
    ctx.clearRect(0, 0, W, H); ctx.font = '11px system-ui'; ctx.fillStyle = mut; ctx.strokeStyle = grid; ctx.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      const y = Y(maxY * i / 3.45);
      ctx.beginPath(); ctx.moveTo(pl, y); ctx.lineTo(W - pr, y); ctx.stroke();
      ctx.fillText(num(maxY * i / 3.45), 2, y + 4);
    }
    ctx.fillText(maxX.toFixed(0) + 's', W - 34, H - 4);
    ctx.beginPath(); sm.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])));
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();
    ctx.lineTo(X(maxX), Y(0)); ctx.lineTo(X(minX), Y(0)); ctx.closePath();
    ctx.globalAlpha = 0.15; ctx.fillStyle = col; ctx.fill(); ctx.globalAlpha = 1;
  }

  /* ------------------------------------------------------------ results */
  function section(title, ...kids) { return h('section', {class: 'card'}, h('h2', {text: title}), ...kids); }
  function pct(v) { return v === null || v === undefined ? '-' : (v * 100).toFixed(2) + '%'; }

  function buildAlignResults() {
    const r = job.result, s = r.stats, box = $('results');
    const aligned = s.reads_total ? 100 * s.reads_aligned / s.reads_total : 0;
    box.replaceChildren(
      h('section', {class: 'grid cols-4'},
        tile('Reads', num(s.reads_total), num(s.bases) + ' bases'),
        tile('Aligned', aligned.toFixed(1) + '%', num(s.reads_aligned) + ' reads'),
        tile('Throughput', num(s.reads_per_sec) + ' /s', 'in ' + GA.dur(s.elapsed_s) + ' on ' + (s.backend || '').toUpperCase()),
        tile('Variants', num(r.n_variants), r.variant_summary.SNP + ' SNP · ' + r.variant_summary.INS + ' INS · ' + r.variant_summary.DEL + ' DEL')));
    if (r.truth_eval) {
      const rows = Object.entries(r.truth_eval).map(([k, v]) => [k, num(v.truth), num(v.called), num(v.tp), pct(v.precision), pct(v.recall)]);
      box.append(section('Accuracy vs. truth set', GA.table(['Class', 'Truth', 'Called', 'True positives', 'Precision', 'Recall'], rows)));
    }
    const vs = r.variant_summary, mx = Math.max(1, vs.SNP, vs.INS, vs.DEL);
    box.append(section('Variant summary',
      h('div', {class: 'bars'}, ['SNP', 'INS', 'DEL'].map(t => h('div', {class: 'bar-row'},
        h('span', {class: 'badge ' + t, text: t}),
        h('div', {class: 'bar-track'}, h('div', {class: 'bar-fill', style: 'width:' + (100 * vs[t] / mx) + '%'})),
        h('span', {text: num(vs[t])})))),
      h('div', {class: 'muted small', text: 'Ti/Tv ratio: ' + (vs.ti_tv === null || vs.ti_tv === undefined ? 'n/a' : vs.ti_tv)})));
    box.append(buildVariantsSection());
    const al = h('div', {text: 'Loading…', class: 'muted'});
    box.append(section('Alignment preview (first 15 reads)', al));
    fetch('/api/jobs/' + id + '/alignments').then(x => x.json()).then(d => {
      const rows = (d.rows || []).slice(0, 15).map(a => [h('span', {class: 'mono', text: a.read}), a.contig, num(a.pos), a.strand, a.score, h('span', {class: 'mono', text: a.cigar})]);
      al.replaceWith(rows.length ? GA.table(['Read', 'Contig', 'Position', 'Strand', 'Score', 'CIGAR'], rows) : h('p', {class: 'muted', text: 'No alignments.'}));
    }).catch(() => { al.textContent = 'Could not load alignments.'; });
    box.append(downloads(), paramsCard());
  }

  function buildVariantsSection() {
    const st = {offset: 0, limit: 15, type: 'all', q: ''};
    const holder = h('div', {}, h('p', {class: 'muted', text: 'Loading…'}));
    const pager = h('div', {class: 'pager'});
    const type = h('select', {'aria-label': 'Variant type'}, ['all', 'SNP', 'INS', 'DEL'].map(t => h('option', {value: t, text: t === 'all' ? 'All types' : t})));
    const q = h('input', {type: 'search', placeholder: 'Search contig / position / allele', 'aria-label': 'Search variants'});
    let timer;
    function load() {
      const url = '/api/jobs/' + id + '/variants?offset=' + st.offset + '&limit=' + st.limit +
        '&type=' + encodeURIComponent(st.type) + '&q=' + encodeURIComponent(st.q);
      fetch(url).then(x => x.json()).then(d => {
        if (d.error) { holder.replaceChildren(h('p', {class: 'muted', text: d.error})); pager.replaceChildren(); return; }
        const rows = d.rows.map(v => [v.chrom, num(v.pos), h('span', {class: 'mono', text: v.ref}), h('span', {class: 'mono', text: v.alt}),
          h('span', {class: 'badge ' + v.type, text: v.type}), v.qual, v.dp, v.ac, v.af, v.gt]);
        holder.replaceChildren(rows.length ? GA.table(['Contig', 'Pos', 'Ref', 'Alt', 'Type', 'Qual', 'Depth', 'Alt reads', 'AF', 'GT'], rows)
                                           : h('p', {class: 'muted', text: 'No variants match.'}));
        const from = d.total ? d.offset + 1 : 0, to = Math.min(d.total, d.offset + d.limit);
        pager.replaceChildren(
          h('span', {class: 'muted small', text: from + '–' + to + ' of ' + num(d.total)}),
          h('button', {class: 'btn small', disabled: d.offset <= 0, onclick: () => { st.offset = Math.max(0, st.offset - st.limit); load(); }, text: '← Prev'}),
          h('button', {class: 'btn small', disabled: to >= d.total, onclick: () => { st.offset += st.limit; load(); }, text: 'Next →'}));
      }).catch(() => holder.replaceChildren(h('p', {class: 'muted', text: 'Could not load variants.'})));
    }
    type.addEventListener('change', () => { st.type = type.value; st.offset = 0; load(); });
    q.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => { st.q = q.value.trim(); st.offset = 0; load(); }, 300); });
    load();
    return section('Variants', h('div', {class: 'toolbar'}, type, q), holder, pager);
  }

  function downloads() {
    const files = (job.result.files || []);
    return section('Downloads', h('div', {class: 'files'}, files.map(f =>
      h('a', {class: 'btn', href: '/jobs/' + id + '/download/' + encodeURIComponent(f.name), text: '⬇ ' + f.label}))));
  }
  function paramsCard() {
    const skip = new Set(['reference_file', 'reads_file', 'truth_file']);
    const rows = Object.entries(job.params || {}).filter(([k]) => !skip.has(k)).map(([k, v]) => [k.replace(/_/g, ' '), String(v)]);
    return section('Run parameters', GA.table(['Parameter', 'Value'], rows));
  }

  function buildBenchResults() {
    const r = job.result, box = $('results'), k = r.kernels, e = r.end_to_end;
    const krows = Object.entries(k).map(([name, v]) => [name, num(v.pairs), v.seconds, num(v.alignments_per_sec), num(v.cells_per_sec),
      v.identical_to_cpu === undefined ? '-' : (v.identical_to_cpu ? '✔ yes' : '✘ NO')]);
    box.replaceChildren(section('Smith-Waterman kernel throughput (100 bp read × 140 bp window)',
      GA.table(['Backend', 'Pairs', 'Seconds', 'Alignments/s', 'Cells/s', 'Identical to CPU'], krows)));
    const g = e.gpualign, erows = [['gpualign (' + g.backend + ')', g.seconds, num(g.reads_per_sec), 'alignment + traceback + VCF']];
    for (const t of ['bwa_mem', 'bowtie2']) {
      const v = e[t];
      erows.push(typeof v === 'string' ? [t, '-', '-', v] : [t, v.seconds, num(v.reads_per_sec), v.threads + ' threads, index excluded']);
    }
    box.append(section('End-to-end pipeline (' + num(g.reads) + ' reads)', GA.table(['Tool', 'Seconds', 'Reads/s', 'Note'], erows),
      h('p', {class: 'muted small', text: 'Small synthetic genome: external tools are mostly start-up bound at this size. Use a larger genome on a GPU host for a meaningful comparison.'})));
    box.append(downloads());
  }

  /* -------------------------------------------------------------- main */
  function render() {
    $('job-title').textContent = job.name;
    const b = $('status-badge'); b.className = 'badge ' + job.status; b.textContent = job.status;
    const bar = $('progress-bar'), failed = job.status === 'failed';
    $('stage-text').textContent = failed ? 'Failed' : (job.status === 'done' ? 'Completed in ' + GA.dur(job.elapsed) : (STAGES[job.stage] || job.stage));
    const indet = job.status === 'queued' || (job.status === 'running' && job.kind === 'benchmark');
    bar.className = 'progress-bar' + (failed ? ' failed' : '') + (indet ? ' indeterminate' : '');
    bar.style.width = indet ? '' : (failed ? '100' : job.progress) + '%';
    $('progress-pct').textContent = indet ? '' : (failed ? '' : Math.round(job.progress) + '%');
    renderLive(); drawChart();
    const eb = $('error-box');
    if (failed) {
      eb.replaceChildren(h('strong', {text: 'This job failed. '}), job.error || 'Unknown error', ' ',
        h('a', {href: '/jobs/' + id + '/log', target: '_blank', rel: 'noopener', text: 'View technical log'}));
      eb.hidden = false;
    } else eb.hidden = true;
    if (job.status === 'done' && resultsFor !== 'done' && job.result) {
      resultsFor = 'done';
      if (job.kind === 'benchmark') buildBenchResults(); else buildAlignResults();
    }
  }
  async function poll() {
    try {
      const r = await fetch('/api/jobs/' + id, {cache: 'no-store'});
      if (r.ok) { job = await r.json(); render(); }
    } catch (e) { /* transient network error: keep polling */ }
    if (job.status === 'queued' || job.status === 'running') setTimeout(poll, 1000);
  }
  window.addEventListener('resize', drawChart);
  render();
  if (job.status === 'queued' || job.status === 'running') setTimeout(poll, 700);
})();
