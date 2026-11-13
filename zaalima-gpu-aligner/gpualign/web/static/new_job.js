(function () {
  const form = document.getElementById('job-form');
  const demo = document.getElementById('demo-fields'), upl = document.getElementById('upload-fields');
  const errBox = document.getElementById('form-errors'), btn = document.getElementById('submit-btn');
  const pw = document.getElementById('upload-progress'), bar = document.getElementById('upload-bar'), txt = document.getElementById('upload-text');

  function sync() {
    const src = form.elements['source'].value;
    demo.hidden = src !== 'demo'; upl.hidden = src !== 'upload';
  }
  form.querySelectorAll('input[name=source]').forEach(r => r.addEventListener('change', sync));
  form.querySelectorAll('[data-preset]').forEach(b => b.addEventListener('click', () => {
    const [g, c] = b.dataset.preset.split(',');
    form.elements['genome_size'].value = g; form.elements['coverage'].value = c;
  }));
  sync();

  form.addEventListener('submit', async ev => {
    ev.preventDefault();
    const src = form.elements['source'].value, errs = [];
    if (src === 'upload') {
      if (!form.elements['reference'].files.length) errs.push('Choose a reference FASTA file.');
      if (!form.elements['reads'].files.length) errs.push('Choose a reads FASTQ file.');
    }
    if (errs.length) { GA.showErrors(errBox, errs); return; }
    GA.showErrors(errBox, []);
    const fd = new FormData(form);
    if (src === 'demo') { fd.delete('reference'); fd.delete('reads'); fd.delete('truth'); }
    btn.disabled = true; btn.textContent = 'Submitting…';
    if (src === 'upload') pw.hidden = false;
    try {
      const r = await GA.postForm('/api/jobs', fd, (l, t) => {
        const p = Math.round(100 * l / t); bar.style.width = p + '%';
        txt.textContent = p < 100 ? 'Uploading… ' + p + '% (' + (l / 1048576).toFixed(1) + ' / ' + (t / 1048576).toFixed(1) + ' MB)' : 'Validating files…';
      });
      if (r.status === 201) { location.href = r.data.url; return; }
      GA.showErrors(errBox, r.data.errors || ['Could not start the job.']);
    } catch (e) { GA.showErrors(errBox, [e.message]); }
    btn.disabled = false; btn.textContent = 'Start alignment'; pw.hidden = true; bar.style.width = '0';
  });
})();
