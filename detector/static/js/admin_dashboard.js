/**
 * VeriTruth Enterprise Admin Analytics Dashboard JS
 * Handlers for Tab Switching, Chart.js Initialization, Table Search/Filter, CSV & PDF Export
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Sidebar Single Page Tab Navigation
    const navItems = document.querySelectorAll('.nav-item[data-tab]');
    const tabPanes = document.querySelectorAll('.tab-pane');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetTab = item.getAttribute('data-tab');

            // Update active state in nav
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            // Display target pane
            tabPanes.forEach(pane => {
                pane.classList.remove('active');
                if (pane.id === `tab-${targetTab}`) {
                    pane.classList.add('active');
                }
            });
        });
    });

    // 2. Initialize Charts using Chart.js
    initPredictionDistributionChart();
    initDailyTrendChart();
    initWeeklyTrafficChart();
    initInputTypeChart();

    // 3. Activity Table Live Search & Filter
    const searchInput = document.getElementById('tableSearchInput');
    const filterSelect = document.getElementById('tableFilterSelect');
    const tableRows = document.querySelectorAll('#activityTableBody tr');

    function filterTable() {
        const query = (searchInput ? searchInput.value : '').toLowerCase();
        const filterVal = (filterSelect ? filterSelect.value : 'ALL').toUpperCase();

        tableRows.forEach(row => {
            const text = row.textContent.toLowerCase();
            const predictionCell = row.querySelector('.prediction-cell');
            const predictionText = predictionCell ? predictionCell.textContent.toUpperCase() : '';

            const matchesSearch = text.includes(query);
            const matchesFilter = (filterVal === 'ALL') || predictionText.includes(filterVal);

            if (matchesSearch && matchesFilter) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        });
    }

    if (searchInput) searchInput.addEventListener('input', filterTable);
    if (filterSelect) filterSelect.addEventListener('change', filterTable);

    // 4. Export CSV Functionality
    const exportCsvBtn = document.getElementById('exportCsvBtn');
    if (exportCsvBtn) {
        exportCsvBtn.addEventListener('click', () => {
            exportTableToCSV('veritruth_prediction_analytics.csv');
        });
    }

    // 5. Export PDF Functionality
    const exportPdfBtn = document.getElementById('exportPdfBtn');
    if (exportPdfBtn) {
        exportPdfBtn.addEventListener('click', () => {
            window.print();
        });
    }

    // 6. Clickable Row Inspection Modal Handler
    const modalOverlay = document.getElementById('predictionModalOverlay');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const clickableRows = document.querySelectorAll('.clickable-row');

    clickableRows.forEach(row => {
        row.addEventListener('click', () => {
            const ds = row.dataset;
            document.getElementById('modal-id').textContent = ds.id ? `#${ds.id}` : '-';
            document.getElementById('modal-username').textContent = ds.username || '-';
            document.getElementById('modal-inputtype').textContent = ds.inputtype || '-';
            document.getElementById('modal-prediction').textContent = ds.prediction || '-';
            document.getElementById('modal-confidence').textContent = ds.confidence ? `${ds.confidence}%` : '-';
            document.getElementById('modal-trust').textContent = ds.trust ? `${ds.trust}%` : '-';
            document.getElementById('modal-bias').textContent = ds.bias ? `${ds.bias}%` : '-';
            document.getElementById('modal-source').textContent = ds.source ? `${ds.source}% ${ds.sourcename ? '(' + ds.sourcename + ')' : ''}` : '-';
            document.getElementById('modal-similarity').textContent = ds.similarity ? `${ds.similarity}%` : '-';
            document.getElementById('modal-author').textContent = ds.author || '-';
            document.getElementById('modal-publishdate').textContent = ds.publishdate || '-';
            document.getElementById('modal-createdat').textContent = ds.createdat || '-';
            document.getElementById('modal-title').textContent = ds.title || '-';
            document.getElementById('modal-url').textContent = ds.url || '-';
            document.getElementById('modal-file').textContent = ds.file || '-';
            document.getElementById('modal-ocr').textContent = ds.ocr || '-';
            document.getElementById('modal-article').textContent = ds.article || '-';
            document.getElementById('modal-explanation').textContent = ds.explanation || '-';

            // Image Preview Handling
            const imgContainer = document.getElementById('modal-image-preview-container');
            const imgEl = document.getElementById('modal-image-img');
            const imgLink = document.getElementById('modal-image-link');

            if (ds.imageurl && ds.imageurl.trim() !== '') {
                if (imgEl) imgEl.src = ds.imageurl;
                if (imgLink) imgLink.href = ds.imageurl;
                if (imgContainer) imgContainer.style.display = 'block';
            } else {
                if (imgContainer) imgContainer.style.display = 'none';
                if (imgEl) imgEl.src = '';
            }

            if (modalOverlay) modalOverlay.classList.add('active');
        });
    });

    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', () => {
            if (modalOverlay) modalOverlay.classList.remove('active');
        });
    }

    if (modalOverlay) {
        modalOverlay.addEventListener('click', (e) => {
            if (e.target === modalOverlay) {
                modalOverlay.classList.remove('active');
            }
        });
    }
});

// Chart 1: Real vs Fake News Distribution (Doughnut)
function initPredictionDistributionChart() {
    const ctx = document.getElementById('predictionDistributionChart');
    if (!ctx) return;

    const realCount = parseInt(ctx.dataset.real || 0);
    const fakeCount = parseInt(ctx.dataset.fake || 0);

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Real News', 'Fake News'],
            datasets: [{
                data: [realCount, fakeCount],
                backgroundColor: ['#10B981', '#EF4444'],
                hoverBackgroundColor: ['#059669', '#DC2626'],
                borderWidth: 2,
                borderColor: '#FFFFFF'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: { family: 'Inter', size: 12 },
                        padding: 16
                    }
                }
            },
            cutout: '70%'
        }
    });
}

// Chart 2: Daily Predictions Line Chart (7 Days)
function initDailyTrendChart() {
    const ctx = document.getElementById('dailyTrendChart');
    if (!ctx) return;

    const days = JSON.parse(ctx.dataset.days || '["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]');
    const values = JSON.parse(ctx.dataset.values || '[12, 19, 15, 25, 22, 30, 28]');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: days,
            datasets: [{
                label: 'Predictions',
                data: values,
                borderColor: '#0B5CAD',
                backgroundColor: 'rgba(11, 92, 173, 0.08)',
                borderWidth: 2.5,
                fill: true,
                tension: 0.35,
                pointRadius: 4,
                pointBackgroundColor: '#0B5CAD'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { family: 'Inter', size: 11 } }
                },
                y: {
                    grid: { color: '#E8EDF3' },
                    ticks: { font: { family: 'Inter', size: 11 } }
                }
            }
        }
    });
}

// Chart 3: Weekly Platform Traffic (Bar Chart)
function initWeeklyTrafficChart() {
    const ctx = document.getElementById('weeklyTrafficChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
            datasets: [{
                label: 'Visitors',
                data: [1420, 1850, 2200, 2650],
                backgroundColor: '#3B82F6',
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false } },
                y: { grid: { color: '#E8EDF3' } }
            }
        }
    });
}

// Chart 4: Input Types Breakdown (Doughnut)
function initInputTypeChart() {
    const ctx = document.getElementById('inputTypeChart');
    if (!ctx) return;

    const urlCount = parseInt(ctx.dataset.url || 0);
    const textCount = parseInt(ctx.dataset.text || 0);
    const fileCount = parseInt(ctx.dataset.file || 0);

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['URL Extraction', 'Text Input', 'File / OCR'],
            datasets: [{
                data: [urlCount, textCount, fileCount],
                backgroundColor: ['#0B5CAD', '#F59E0B', '#8B5CF6'],
                borderWidth: 2,
                borderColor: '#FFFFFF'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            },
            cutout: '65%'
        }
    });
}

// Helper: Export HTML Table data to CSV file
function exportTableToCSV(filename) {
    const csv = [];
    const rows = document.querySelectorAll('#activityTableBody tr');
    const headers = document.querySelectorAll('#activityTableHead th');

    // Extract headers
    const headerRow = [];
    headers.forEach(h => headerRow.push(`"${h.textContent.trim().replace(/"/g, '""')}"`));
    csv.push(headerRow.join(','));

    // Extract rows
    rows.forEach(row => {
        if (row.style.display !== 'none') {
            const rowData = [];
            row.querySelectorAll('td').forEach(cell => {
                rowData.push(`"${cell.textContent.trim().replace(/"/g, '""')}"`);
            });
            csv.push(rowData.join(','));
        }
    });

    // Download CSV
    const csvFile = new Blob([csv.join('\n')], { type: 'text/csv' });
    const downloadLink = document.createElement('a');
    downloadLink.download = filename;
    downloadLink.href = window.URL.createObjectURL(csvFile);
    downloadLink.style.display = 'none';
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
}
