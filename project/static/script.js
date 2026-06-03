/* ═══════════════════════════════════════════════════════════
   FakeDetect AI  –  Core Logic
   Handles: Predictions, History, and UI Transitions
═══════════════════════════════════════════════════════════ */

'use strict';

// ── DOM Elements ──
const messageInput   = document.getElementById('messageInput');
const charCounter    = document.getElementById('charCounter');
const analyseBtn     = document.getElementById('analyseBtn');
const clearBtn       = document.getElementById('clearBtn');
const resultCard     = document.getElementById('resultCard');
const resultContent  = document.getElementById('resultContent');
const resultVerdict  = document.getElementById('resultVerdict');
const resultSub      = document.getElementById('resultSub');
const resultIcon     = document.getElementById('resultIcon');
const confidenceValue= document.getElementById('confidenceValue');
const progressBar    = document.getElementById('progressBar');
const copyBtn        = document.getElementById('copyBtn');
const refreshHistory = document.getElementById('refreshHistory');
const historyTable   = document.getElementById('historyTable');
const emptyHistory   = document.getElementById('emptyHistory');
const toast          = document.getElementById('toast');

// ── State ──
let lastData = null;
const MAX_LEN = 500;

// ── Character Counter ──
messageInput.addEventListener('input', () => {
    const len = messageInput.value.length;
    charCounter.textContent = `${len} / ${MAX_LEN}`;
    charCounter.style.color = len >= MAX_LEN ? '#ff4d4d' : 'var(--text-dim)';
});

// ── Clear Action ──
clearBtn.addEventListener('click', () => {
    messageInput.value = '';
    charCounter.textContent = `0 / ${MAX_LEN}`;
    charCounter.style.color = 'var(--text-dim)';
    resultCard.classList.add('hidden');
    messageInput.focus();
});

// ── Analysis Logic ──
analyseBtn.addEventListener('click', async () => {
    const message = messageInput.value.trim();
    
    if (!message) {
        showToast("Please enter a message to analyze.");
        return;
    }

    setLoading(true);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error || "Server Error");

        lastData = data;
        displayResult(data);
        loadHistory();

    } catch (err) {
        showToast(`Error: ${err.message}`);
    } finally {
        setLoading(false);
    }
});

function setLoading(state) {
    if (state) {
        analyseBtn.classList.add('loading');
        analyseBtn.disabled = true;
    } else {
        analyseBtn.classList.remove('loading');
        analyseBtn.disabled = false;
    }
}

function displayResult(data) {
    const isSpam = data.is_spam;
    
    resultCard.classList.remove('hidden');
    resultContent.className = `result-card ${isSpam ? 'spam-theme' : 'ham-theme'}`;
    
    resultVerdict.textContent = isSpam ? "SPAM DETECTED" : "SAFE MESSAGE";
    resultSub.textContent = isSpam 
        ? "This message matches known patterns of spam or phishing. Please be cautious." 
        : "No malicious patterns detected. This message appears to be safe.";
    
    resultIcon.innerHTML = isSpam 
        ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0zM12 9v4M12 17h.01"/></svg>'
        : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/></svg>';

    // Confidence Animation
    confidenceValue.textContent = `0%`;
    progressBar.style.width = `0%`;
    
    setTimeout(() => {
        confidenceValue.textContent = `${data.confidence}%`;
        progressBar.style.width = `${data.confidence}%`;
    }, 100);

    resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ── Copy Function ──
copyBtn.addEventListener('click', () => {
    if (!lastData) return;
    const text = `FakeDetect AI Report\nVerdict: ${lastData.prediction}\nConfidence: ${lastData.confidence}%\nStatus: ${lastData.is_spam ? 'Dangerous' : 'Safe'}`;
    navigator.clipboard.writeText(text).then(() => showToast("Report copied to clipboard!"));
});

// ── History Logic ──
async function loadHistory() {
    try {
        const res = await fetch('/history');
        const data = await res.json();
        
        if (data.length === 0) {
            emptyHistory.classList.remove('hidden');
            historyTable.classList.add('hidden');
            return;
        }

        emptyHistory.classList.add('hidden');
        historyTable.classList.remove('hidden');

        historyTable.innerHTML = data.map(row => `
            <div class="history-row">
                <div class="log-msg" title="${escapeHtml(row.Message)}">${escapeHtml(row.Message)}</div>
                <div class="log-label ${row.Prediction === 'SPAM' ? 'log-spam' : 'log-ham'}">${row.Prediction}</div>
                <div class="log-ts">${row.Timestamp}</div>
            </div>
        `).join('');

    } catch (err) {
        console.error("History failed:", err);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

refreshHistory.addEventListener('click', () => {
    refreshHistory.textContent = "Loading...";
    loadHistory().finally(() => {
        setTimeout(() => refreshHistory.textContent = "Refresh Log", 500);
    });
});

// ── Toast Utility ──
function showToast(msg) {
    toast.textContent = msg;
    toast.style.display = 'block';
    setTimeout(() => {
        toast.style.display = 'none';
    }, 3000);
}

// ── Init ──
loadHistory();
messageInput.focus();
