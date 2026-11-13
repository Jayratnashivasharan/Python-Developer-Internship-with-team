(function () {
  const btn = document.getElementById('quick-demo');
  if (!btn) return;
  const box = document.getElementById('quick-error');
  btn.addEventListener('click', async () => {
    btn.disabled = true; btn.textContent = 'Starting…';
    const fd = new FormData();
    fd.append('source', 'demo'); fd.append('name', 'Quick demo');
    fd.append('genome_size', '50000'); fd.append('coverage', '15');
    try {
      const r = await GA.postForm('/api/jobs', fd);
      if (r.status === 201) { location.href = r.data.url; return; }
      box.textContent = (r.data.errors || ['Could not start the job.']).join(' '); box.hidden = false;
    } catch (e) { box.textContent = e.message; box.hidden = false; }
    btn.disabled = false; btn.textContent = '⚡ Run quick demo';
  });
})();
