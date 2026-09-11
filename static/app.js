document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const spinner = analyzeBtn.querySelector('.spinner');
    const errorMessage = document.getElementById('error-message');
    
    // Results DOM Elements
    const resultsSection = document.getElementById('results-section');
    const predictionEmoji = document.getElementById('prediction-emoji');
    const predictionTitle = document.getElementById('prediction-title');
    const qualityBadge = document.getElementById('quality-badge');
    const topConfidenceVal = document.getElementById('top-confidence-val');
    const topConfidenceFill = document.getElementById('top-confidence-fill');
    const top5List = document.getElementById('top5-list');

    let currentFile = null;

    // Emoji mapping
    const emojiMap = {
        'apple': '🍎',
        'banana': '🍌',
        'guava': '🍈',
        'lime': '🍋',
        'orange': '🍊',
        'pomegranate': '🫐',
        'default': '🍏'
    };

    // Drag & Drop Handlers
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('active');
    });

    uploadZone.addEventListener('dragenter', (e) => {
        e.preventDefault();
        uploadZone.classList.add('active');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('active');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('active');
        
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showError('Please upload a valid image file.');
            return;
        }

        currentFile = file;
        hideError();
        resultsSection.classList.add('hidden');
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // Set loading state
        analyzeBtn.disabled = true;
        btnText.classList.add('hidden');
        spinner.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        hideError();

        const formData = new FormData();
        formData.append('image', currentFile);

        try {
            // Mock API delay for effect if actual API fails or takes time
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}`);
            }

            const data = await response.json();
            showResults(data);
        } catch (error) {
            console.error('API Error:', error);
            // Fallback for visual testing if API is not yet available
            showError('Failed to connect to the prediction API. Check console for details.');
        } finally {
            // Remove loading state
            analyzeBtn.disabled = false;
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
        }
    });

    function formatClassName(className) {
        // e.g., "apple_good" -> "Apple — Good"
        const parts = className.split('_');
        if (parts.length >= 2) {
            const fruit = parts[0].charAt(0).toUpperCase() + parts[0].slice(1);
            const quality = parts[1].charAt(0).toUpperCase() + parts[1].slice(1);
            return { fruit, quality, formatted: `${fruit} — ${quality}` };
        }
        return { fruit: className, quality: 'Unknown', formatted: className };
    }

    function getQualityColor(quality) {
        const q = quality.toLowerCase();
        if (q.includes('good')) return 'var(--success)';
        if (q.includes('bad')) return 'var(--danger)';
        if (q.includes('mixed')) return 'var(--warning)';
        return 'var(--accent-primary)'; // default
    }

    function getQualityClass(quality) {
        const q = quality.toLowerCase();
        if (q.includes('good')) return 'quality-good';
        if (q.includes('bad')) return 'quality-bad';
        if (q.includes('mixed')) return 'quality-mixed';
        return '';
    }

    function getEmoji(fruitName) {
        const f = fruitName.toLowerCase();
        for (const [key, value] of Object.entries(emojiMap)) {
            if (f.includes(key)) return value;
        }
        return emojiMap.default;
    }

    function showResults(data) {
        if (!data || !data.predictions || data.predictions.length === 0) {
            showError('No predictions returned from the API.');
            return;
        }

        const topPrediction = data.predictions[0];
        const { fruit, quality, formatted } = formatClassName(topPrediction.class);
        const topConfPercentage = (topPrediction.confidence * 100).toFixed(1);

        // Update Top Prediction UI
        predictionEmoji.textContent = getEmoji(fruit);
        predictionTitle.textContent = fruit;
        
        qualityBadge.textContent = quality;
        qualityBadge.className = `quality-badge ${getQualityClass(quality)}`;
        
        topConfidenceVal.textContent = `${topConfPercentage}%`;
        
        // Reset width before animating
        topConfidenceFill.style.width = '0%';
        topConfidenceFill.style.backgroundColor = getQualityColor(quality);
        
        // Clear top 5 list
        top5List.innerHTML = '';
        
        // Populate top 5
        const top5 = data.predictions.slice(0, 5);
        top5.forEach((pred, index) => {
            const parsed = formatClassName(pred.class);
            const confPercentage = (pred.confidence * 100).toFixed(1);
            const color = getQualityColor(parsed.quality);
            
            const card = document.createElement('div');
            card.className = 'result-card';
            card.style.animationDelay = `${0.1 * index}s`;
            
            card.innerHTML = `
                <div class="result-header">
                    <span class="result-name">${parsed.formatted}</span>
                    <span class="result-conf">${confPercentage}%</span>
                </div>
                <div class="confidence-bar-container small">
                    <div class="confidence-bar-fill" style="background-color: ${color}; width: 0%;" data-target="${confPercentage}%"></div>
                </div>
            `;
            
            top5List.appendChild(card);
        });

        // Show section
        resultsSection.classList.remove('hidden');

        // Trigger animations next frame
        requestAnimationFrame(() => {
            setTimeout(() => {
                topConfidenceFill.style.width = `${topConfPercentage}%`;
                
                const fills = top5List.querySelectorAll('.confidence-bar-fill');
                fills.forEach(fill => {
                    fill.style.width = fill.getAttribute('data-target');
                });
            }, 50);
        });
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorMessage.classList.remove('hidden');
        
        // Auto dismiss after 5 seconds
        setTimeout(hideError, 5000);
    }

    function hideError() {
        errorMessage.classList.add('hidden');
    }
});
