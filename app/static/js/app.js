// Frontend logic for FRA OCR System

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');

const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');

const resultsSection = document.getElementById('resultsSection');
const resultGrid = document.getElementById('resultGrid');
const rawTextEl = document.getElementById('rawText');
const exportBtn = document.getElementById('exportBtn');
const newScanBtn = document.getElementById('newScanBtn');

const documentList = document.getElementById('documentList');

let selectedFile = null;
let lastDocumentId = null;

function showProgress(percent, text) {
  progressSection.style.display = 'block';
  progressFill.style.width = `${percent}%`;
  progressText.innerText = text || 'Processing...';
}

function hideProgress() {
  progressSection.style.display = 'none';
  progressFill.style.width = '0%';
}

function resetResults() {
  resultsSection.style.display = 'none';
  resultGrid.innerHTML = '';
  rawTextEl.textContent = '';
}

function renderResults(extracted, rawText) {
  resultsSection.style.display = 'block';
  resultGrid.innerHTML = '';

  const order = ['patta_id','holder_name','relative_name','village','district','state','area','issue_date','claim_type','status'];
  order.forEach((key) => {
    const field = extracted[key];
    if (!field) return;

    const item = document.createElement('div');
    item.className = 'result-item';

    const label = document.createElement('div');
    label.className = 'result-label';
    label.textContent = key.replace('_', ' ').toUpperCase();

    const value = document.createElement('div');
    value.className = 'result-value';
    value.textContent = field.value ?? '—';

    const conf = document.createElement('span');
    const c = field.confidence ?? 0;
    conf.className = 'confidence-badge ' + (c >= 0.8 ? 'confidence-high' : c >= 0.6 ? 'confidence-medium' : 'confidence-low');
    conf.textContent = `conf ${c.toFixed(2)}`;

    value.appendChild(conf);

    item.appendChild(label);
    item.appendChild(value);

    if (field.note) {
      const note = document.createElement('div');
      note.className = 'result-label';
      note.textContent = `note: ${field.note}`;
      item.appendChild(note);
    }

    resultGrid.appendChild(item);
  });

  rawTextEl.textContent = rawText || '';
}

function refreshHistory() {
  fetch('/api/documents')
    .then(res => res.json())
    .then(data => {
      documentList.innerHTML = '';
      if (!data.success) return;

      data.documents.forEach(doc => {
        const el = document.createElement('div');
        el.className = 'document-item';

        const info = document.createElement('div');
        info.className = 'document-info';
        const name = document.createElement('div');
        name.className = 'document-name';
        name.textContent = doc.name;
        const meta = document.createElement('div');
        meta.className = 'document-meta';
        meta.textContent = `ID: ${doc.id}`;
        info.appendChild(name);
        info.appendChild(meta);

        el.appendChild(info);

        el.addEventListener('click', () => {
          fetch(`/api/document/${doc.id}`)
            .then(res => res.json())
            .then(d => {
              if (!d.success) return;
              lastDocumentId = d.document.document_id;
              const extracted = { ...d.document.regex_fields, ...d.document.gemini_fields };
              renderResults(extracted, d.document.preview);
            });
        });

        documentList.appendChild(el);
      });
    });
}

// Upload interactions
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));

uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) {
    selectedFile = file;
    uploadBtn.disabled = false;
    uploadArea.querySelector('p').textContent = `Selected: ${file.name}`;
  }
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    selectedFile = file;
    uploadBtn.disabled = false;
    uploadArea.querySelector('p').textContent = `Selected: ${file.name}`;
  }
});

uploadBtn.addEventListener('click', () => {
  if (!selectedFile) return;

  resetResults();
  showProgress(10, 'Uploading file...');

  const formData = new FormData();
  formData.append('file', selectedFile);

  // Collect language hints
  const langs = Array.from(document.querySelectorAll('.language-selection input:checked')).map(el => el.value);
  formData.append('languages', langs.join(','));

  fetch('/api/upload', {
    method: 'POST',
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    if (data.error) {
      hideProgress();
      alert('Error: ' + (data.error || 'Unknown error'));
      return;
    }

    showProgress(90, 'Finalizing...');
    lastDocumentId = data.document_id;
    const extracted = { ...data.regex_fields, ...data.gemini_fields };
    renderResults(extracted, data.preview);
    showProgress(100, 'Completed');
    setTimeout(hideProgress, 1000);
    refreshHistory();
  })
  .catch(err => {
    hideProgress();
    alert('Upload failed: ' + err.message);
  });
});

exportBtn.addEventListener('click', () => {
  if (!lastDocumentId) return;
  showProgress(50, 'Generating PDF report...');
  fetch(`/api/export/${lastDocumentId}/pdf`)
    .then(res => res.json())
    .then(data => {
      hideProgress();
      if (data.success) {
        // Trigger download
        window.open(`/api/download/${lastDocumentId}.pdf`, '_blank');
      } else {
        alert('Export failed: ' + (data.error || 'Unknown error'));
      }
    })
    .catch(err => {
      hideProgress();
      alert('Export failed: ' + err.message);
    });
});

newScanBtn.addEventListener('click', () => {
  selectedFile = null;
  uploadBtn.disabled = true;
  uploadArea.querySelector('p').textContent = 'Drop your FRA document here or click to browse';
  resetResults();
});

// Initial load
refreshHistory();

