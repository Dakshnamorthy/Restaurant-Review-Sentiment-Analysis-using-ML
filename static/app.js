document.addEventListener('DOMContentLoaded', () => {

    let debounceTimer = null;

    const el = {
        input:         document.getElementById('analyzer-input'),
        charCount:     document.getElementById('analyzer-char-count'),
        btnClear:      document.getElementById('btn-analyzer-clear'),
        btnRun:        document.getElementById('btn-analyzer-run'),
        placeholder:   document.getElementById('analyzer-result-placeholder'),
        resultDisplay: document.getElementById('analyzer-result-display'),
        badge:         document.getElementById('sentiment-indicator-badge'),
        icon:          document.getElementById('sentiment-icon'),
        sentText:      document.getElementById('sentiment-text'),
        confPct:       document.getElementById('confidence-percentage'),
        confBar:       document.getElementById('confidence-bar'),
        probNeg:       document.getElementById('prob-neg-val'),
        probPos:       document.getElementById('prob-pos-val'),
        highlightBox:  document.getElementById('highlighted-text-box'),
        preprocBox:    document.getElementById('preprocessed-text-box'),
        templateTags:  document.querySelectorAll('.template-tag'),
        tooltip:       document.getElementById('coeff-tooltip')
    };

    // ── Helpers ──────────────────────────────────────────────────
    function debounce(fn, ms) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(fn, ms);
    }

    function escapeHTML(str) {
        return str.replace(/[&<>'"]/g, t =>
            ({ '&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;' }[t]));
    }

    function showTooltip(e, word, weight) {
        const sign   = weight > 0 ? '+' : '';
        const label  = weight > 0 ? 'Positive' : 'Negative';
        const cls    = weight > 0 ? 'text-teal' : 'text-rose';
        el.tooltip.innerHTML = `
            <div class="tooltip-title">"${escapeHTML(word)}"</div>
            <div>Coefficient: <strong class="${cls}">${sign}${weight.toFixed(4)}</strong></div>
            <div style="font-size:.63rem;color:#9ca3af;margin-top:3px">
                Pulls prediction → <strong>${label}</strong>
            </div>`;
        el.tooltip.classList.remove('hidden');
        el.tooltip.style.left = `${e.clientX + 14}px`;
        el.tooltip.style.top  = `${e.clientY + 14}px`;
    }

    function hideTooltip() { el.tooltip.classList.add('hidden'); }

    // ── Render highlighted text ───────────────────────────────────
    function renderHighlights(originalText, contributions) {
        el.highlightBox.innerHTML = '';

        const map = {};
        contributions.forEach(c => { map[c.word.toLowerCase()] = c.weight; });

        originalText.split(/(\s+|[^a-zA-Z]+)/).forEach(token => {
            const key = token.toLowerCase().trim();
            if (key && key in map) {
                const w    = map[key];
                const span = document.createElement('span');
                span.className = `highlight-word ${w > 0 ? 'pos-highlight' : 'neg-highlight'}`;
                span.innerText = token;
                span.addEventListener('mousemove', e => showTooltip(e, token, w));
                span.addEventListener('mouseleave', hideTooltip);
                el.highlightBox.appendChild(span);
            } else {
                el.highlightBox.appendChild(document.createTextNode(token));
            }
        });
    }

    // ── Run prediction ────────────────────────────────────────────
    async function predict() {
        const text = el.input.value.trim();
        if (!text) { clearResults(); return; }

        el.btnRun.disabled = true;
        el.btnRun.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing…';

        try {
            const res  = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            if (!res.ok) throw new Error('API error');
            const data = await res.json();

            // Show result pane
            el.placeholder.classList.add('hidden');
            el.resultDisplay.classList.remove('hidden');

            // Sentiment badge
            const isPos = data.sentiment === 'Positive';
            el.sentText.innerText = data.sentiment;
            el.badge.className   = `sentiment-indicator ${isPos ? 'positive-indicator' : 'negative-indicator'}`;
            el.icon.className    = `fa-solid ${isPos ? 'fa-face-smile' : 'fa-face-frown'}`;

            // Confidence bar
            const pct = (data.confidence * 100).toFixed(1);
            el.confPct.innerText        = `${pct}%`;
            el.confBar.style.width      = `${pct}%`;
            el.confBar.className        = `gauge-bar-inner ${isPos ? 'positive-indicator-bar' : 'negative-indicator-bar'}`;

            // Probabilities
            el.probNeg.innerText = `${(data.probability_negative * 100).toFixed(1)}%`;
            el.probPos.innerText = `${(data.probability_positive * 100).toFixed(1)}%`;

            // Highlights & preprocessed
            renderHighlights(text, data.word_contributions);
            el.preprocBox.innerText = data.cleaned_text || '(empty after preprocessing)';

        } catch (err) {
            console.error(err);
        } finally {
            el.btnRun.disabled  = false;
            el.btnRun.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> Analyze';
        }
    }

    function clearResults() {
        el.input.value           = '';
        el.charCount.innerText   = '0 characters';
        el.resultDisplay.classList.add('hidden');
        el.placeholder.classList.remove('hidden');
    }

    // ── Event listeners ───────────────────────────────────────────
    el.input.addEventListener('input', () => {
        el.charCount.innerText = `${el.input.value.length} characters`;
        debounce(predict, 500);
    });

    el.btnRun.addEventListener('click', predict);
    el.btnClear.addEventListener('click', clearResults);

    el.templateTags.forEach(tag => {
        tag.addEventListener('click', () => {
            el.input.value         = tag.dataset.text;
            el.charCount.innerText = `${el.input.value.length} characters`;
            predict();
        });
    });
});
