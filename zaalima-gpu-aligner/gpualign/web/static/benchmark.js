(function () {
  const form = document.getElementById('bench-form');
  const box = document.getElementById('bench-errors'), btn = document.getElementById('bench-btn');
  form.addEventListener('submit', async ev => {
    ev.preventDefault();
    GA.showErrors(box, []); btn.disabled = true; btn.textContent = 'Starting…';
    try {
      const r = await GA.postForm('/api/benchmark', new FormData(form));
      if (r.status === 201) { location.href = r.data.url; return; }
      GA.showErrors(box, r.data.errors || ['Could not start the benchmark.']);
    } catch (e) { GA.showErrors(box, [e.message]); }
    btn.disabled = false; btn.textContent = 'Run benchmark';
  });
})();
