/* Shared helpers (no dependencies). */
window.GA = (function () {
  const nf = new Intl.NumberFormat();
  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') e.className = v;
      else if (k === 'text') e.textContent = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid === null || kid === undefined || kid === false) continue;
      e.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return e;
  }
  function num(n, d) {
    if (n === null || n === undefined || isNaN(n)) return '-';
    return d ? Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}) : nf.format(Math.round(n));
  }
  function dur(s) {
    s = Number(s || 0);
    return s < 60 ? s.toFixed(1) + 's' : Math.floor(s / 60) + 'm ' + Math.floor(s % 60) + 's';
  }
  function showErrors(box, errors) {
    box.replaceChildren();
    if (!errors || !errors.length) { box.hidden = true; return; }
    box.append(h('strong', {text: 'Please fix the following:'}), h('ul', {}, errors.map(e => h('li', {text: e}))));
    box.hidden = false;
    if (box.scrollIntoView) box.scrollIntoView({behavior: 'smooth', block: 'center'});
  }
  /* POST a FormData with upload progress. Resolves {status, data}. */
  function postForm(url, formData, onProgress) {
    return new Promise((resolve, reject) => {
      const x = new XMLHttpRequest();
      x.open('POST', url);
      x.responseType = 'json';
      x.upload.onprogress = ev => { if (ev.lengthComputable && onProgress) onProgress(ev.loaded, ev.total); };
      x.onload = () => resolve({status: x.status, data: x.response || {errors: ['Unexpected server response (HTTP ' + x.status + ').']}});
      x.onerror = () => reject(new Error('Network error - is the server running?'));
      x.send(formData);
    });
  }
  function table(headers, rows, cls) {
    return h('div', {class: 'table-wrap'}, h('table', {class: 'table ' + (cls || '')},
      h('thead', {}, h('tr', {}, headers.map(x => h('th', {text: x})))),
      h('tbody', {}, rows.map(r => h('tr', {}, r.map(c => h('td', {}, c)))))));
  }
  document.addEventListener('DOMContentLoaded', () => {
    const t = document.getElementById('navtoggle'), n = document.getElementById('nav');
    if (t && n) t.addEventListener('click', () => { const o = n.classList.toggle('open'); t.setAttribute('aria-expanded', o); });
    document.querySelectorAll('form[data-confirm]').forEach(f =>
      f.addEventListener('submit', ev => { if (!confirm(f.dataset.confirm)) ev.preventDefault(); }));
  });
  return {h, num, dur, showErrors, postForm, table};
})();
