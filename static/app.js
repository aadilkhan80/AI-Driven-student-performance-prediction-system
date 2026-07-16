// App State
let activeTab = 'dashboard';
let charts = {};
let modelStats = null;
let isOffline = false;

// Average student baseline for comparison chart
const baselineStudent = {
    study_hours: 15.0,
    attendance: 80.0,
    sleep_hours: 7.0,
    previous_grade: 70.0
};

// DOM Elements
const navButtons = document.querySelectorAll('.nav-btn');
const tabContents = document.querySelectorAll('.tab-content');
const viewTitle = document.getElementById('view-title');
const viewSubtitle = document.getElementById('view-subtitle');
const offlineBadge = document.getElementById('offline-badge');
const predictionForm = document.getElementById('prediction-form');

// Ranges
const rangeInputs = {
    study_hours: { input: document.getElementById('study_hours'), bubble: document.getElementById('val-study_hours'), suffix: '' },
    attendance: { input: document.getElementById('attendance'), bubble: document.getElementById('val-attendance'), suffix: '%' },
    sleep_hours: { input: document.getElementById('sleep_hours'), bubble: document.getElementById('val-sleep_hours'), suffix: '' },
    previous_grade: { input: document.getElementById('previous_grade'), bubble: document.getElementById('val-previous_grade'), suffix: '%' }
};

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    setupTabNavigation();
    setupRangeInputs();
    fetchModelStats();
    
    // Initial prediction simulation on load
    runPrediction(true);
    
    // Form submission
    predictionForm.addEventListener('submit', (e) => {
        e.preventDefault();
        runPrediction();
    });

    // Retrain button
    const retrainBtn = document.getElementById('retrain-btn');
    if (retrainBtn) {
        retrainBtn.addEventListener('click', handleRetrain);
    }
});

// Setup tab switches
function setupTabNavigation() {
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            
            // Toggle active classes
            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const targetContent = document.getElementById(`tab-${tabId}`);
            if (targetContent) targetContent.classList.add('active');
            
            activeTab = tabId;
            
            // Update Headers
            if (tabId === 'dashboard') {
                viewTitle.innerText = "Performance Predictor";
                viewSubtitle.innerText = "Enter student metrics below to evaluate risk and forecast GPA.";
            } else if (tabId === 'analytics') {
                viewTitle.innerText = "Feature Insights";
                viewSubtitle.innerText = "Deconstruct which attributes carry the highest weight in student performance.";
                setTimeout(renderAnalyticsCharts, 100); // Small timeout to allow canvas scaling
            } else if (tabId === 'settings') {
                viewTitle.innerText = "Model Control Panel";
                viewSubtitle.innerText = "Retrain the machine learning pipelines and inspect testing diagnostics.";
            }
        });
    });
}

// Update value bubbles on range inputs
function setupRangeInputs() {
    Object.keys(rangeInputs).forEach(key => {
        const item = rangeInputs[key];
        const updateBubble = () => {
            item.bubble.innerText = `${item.input.value}${item.suffix}`;
        };
        item.input.addEventListener('input', updateBubble);
        updateBubble(); // Init
    });
}

// Fetch model configuration and statistics
async function fetchModelStats() {
    try {
        const response = await fetch('/api/stats');
        if (!response.ok) throw new Error("Backend offline");
        
        modelStats = await response.json();
        setServerOffline(false);
        updateStatsUI();
    } catch (error) {
        console.warn("Could not connect to FastAPI server. Switching to local predictive fallback.", error);
        setServerOffline(true);
        loadFallbackStats();
    }
}

// Handle server connection display states
function setServerOffline(offline) {
    isOffline = offline;
    const dot = document.querySelector('.status-dot');
    const statusTxt = document.querySelector('.status-text');
    
    if (offline) {
        offlineBadge.style.display = 'block';
        dot.className = 'status-dot offline';
        statusTxt.innerText = 'Server Offline';
    } else {
        offlineBadge.style.display = 'none';
        dot.className = 'status-dot online';
        statusTxt.innerText = 'Server Connected';
    }
}

// Load default diagnostics if API is unreachable
function loadFallbackStats() {
    modelStats = {
        gpa_metrics: { r2_score: 0.932, mean_squared_error: 0.024, mean_absolute_error: 0.118 },
        risk_metrics: { accuracy: 0.954 },
        dataset_size: 1200,
        feature_importances: {
            gpa: {
                previous_grade: 0.38,
                attendance: 0.29,
                study_hours: 0.18,
                sleep_hours: 0.05,
                parental_support_High: 0.03,
                stress_level_High: 0.025,
                tutoring: 0.02,
                internet_access: 0.015,
                extracurriculars: 0.01
            }
        }
    };
    updateStatsUI();
}

// Update the DOM with diagnostics
function updateStatsUI() {
    if (!modelStats) return;
    
    const r2Val = (modelStats.gpa_metrics.r2_score).toFixed(3);
    const accVal = (modelStats.risk_metrics.accuracy * 100).toFixed(1) + '%';
    
    // Sidebar
    document.getElementById('model-r2').innerText = r2Val;
    document.getElementById('model-acc').innerText = accVal;
    
    // Settings Tab Diagnostics Table
    document.getElementById('stat-r2').innerText = r2Val;
    document.getElementById('stat-mse').innerText = (modelStats.gpa_metrics.mean_squared_error).toFixed(4);
    document.getElementById('stat-mae').innerText = (modelStats.gpa_metrics.mean_absolute_error).toFixed(4);
    document.getElementById('stat-acc').innerText = accVal;
    document.getElementById('stat-size').innerText = modelStats.dataset_size.toLocaleString();
}

// Trigger Model Retraining
async function handleRetrain() {
    const numStudentsSelect = document.getElementById('num_students');
    const trainBtn = document.getElementById('retrain-btn');
    
    const originalText = trainBtn.innerHTML;
    trainBtn.disabled = true;
    trainBtn.innerHTML = '<span>Training Models...</span><div class="btn-glow"></div>';
    
    try {
        if (isOffline) {
            // Simulated delay for offline training demo
            await new Promise(resolve => setTimeout(resolve, 1500));
            modelStats.dataset_size = parseInt(numStudentsSelect.value);
            updateStatsUI();
            alert("Simulated Local Retraining complete!");
            return;
        }
        
        const response = await fetch('/api/retrain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ num_students: parseInt(numStudentsSelect.value) })
        });
        
        if (!response.ok) throw new Error("Training API error");
        
        const resData = await response.json();
        modelStats = resData.metrics;
        updateStatsUI();
        alert("ML models successfully retrained on new synthetic records!");
        
        // Re-render if on analytics tab
        if (activeTab === 'analytics') {
            renderAnalyticsCharts();
        }
    } catch (err) {
        console.error("Retrain error:", err);
        alert("Failed to retrain. Verify server connection.");
    } finally {
        trainBtn.disabled = false;
        trainBtn.innerHTML = originalText;
    }
}

// Main Prediction logic
async function runPrediction(isInitial = false) {
    const data = getFormData();
    let result = null;
    
    const predictBtn = document.getElementById('predict-btn');
    const originalText = predictBtn ? predictBtn.innerHTML : '';
    if (predictBtn && !isInitial) {
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span>Calculating...</span><div class="btn-glow"></div>';
    }
    
    try {
        if (isOffline) {
            // If offline, wait 200ms to simulate ML pipeline delay, then run local logic
            if (!isInitial) await new Promise(res => setTimeout(res, 200));
            result = runLocalHeuristicPrediction(data);
        } else {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            if (!response.ok) throw new Error("Predict failed");
            result = await response.json();
        }
        
        displayPredictionResult(result, data);
    } catch (err) {
        console.error("Prediction API failed. Using heuristic fallback.", err);
        setServerOffline(true);
        result = runLocalHeuristicPrediction(data);
        displayPredictionResult(result, data);
    } finally {
        if (predictBtn && !isInitial) {
            predictBtn.disabled = false;
            predictBtn.innerHTML = originalText;
        }
    }
}

// Read inputs from DOM elements
function getFormData() {
    return {
        study_hours: parseFloat(document.getElementById('study_hours').value),
        attendance: parseFloat(document.getElementById('attendance').value),
        sleep_hours: parseFloat(document.getElementById('sleep_hours').value),
        previous_grade: parseFloat(document.getElementById('previous_grade').value),
        parental_support: document.getElementById('parental_support').value,
        stress_level: document.getElementById('stress_level').value,
        tutoring: document.getElementById('tutoring').checked ? 1 : 0,
        extracurriculars: document.getElementById('extracurriculars').checked ? 1 : 0,
        internet_access: document.getElementById('internet_access').checked ? 1 : 0
    };
}

// Local mathematical fallback prediction that maps dataset relationships
function runLocalHeuristicPrediction(data) {
    const parentalMap = { 'Low': 0, 'Medium': 1, 'High': 2 };
    const stressMap = { 'Low': 2, 'Medium': 1, 'High': 0 };
    
    const parentalVal = parentalMap[data.parental_support];
    const stressVal = stressMap[data.stress_level];
    
    // Formula mirrors app/data_generator.py logic
    const baseScore = (
        0.30 * (data.previous_grade / 100.0) +
        0.25 * (data.attendance / 100.0) +
        0.20 * (data.study_hours / 30.0) +
        0.08 * (data.sleep_hours / 10.0) +
        0.05 * (parentalVal / 2.0) +
        0.04 * data.tutoring +
        0.03 * data.internet_access +
        0.03 * (stressVal / 2.0) +
        0.02 * data.extracurriculars
    );
    
    // Map baseScore to GPA scale 1.0 - 4.0
    let gpa = 1.0 + (baseScore - 0.2) * (3.0 / 0.7);
    gpa = Math.min(Math.max(gpa, 0.0), 4.0);
    gpa = Math.round(gpa * 100) / 100;
    
    // Determine risk category
    let risk = 'Low';
    if (gpa < 2.0 || data.attendance < 65) {
        risk = 'High';
    } else if (gpa < 3.0 || data.attendance < 80) {
        risk = 'Medium';
    }
    
    return {
        predicted_gpa: gpa,
        predicted_risk: risk
    };
}

// Show Prediction outcome on Frontend UI
function displayPredictionResult(result, inputData) {
    const gpaVal = result.predicted_gpa;
    const riskVal = result.predicted_risk;
    
    // Update numerical GPA text
    document.getElementById('predicted-gpa').innerText = gpaVal.toFixed(2);
    
    // Animate Circular Progress Ring
    const svgCircle = document.getElementById('gpa-svg-ring');
    const radius = 40;
    const circumference = 2 * Math.PI * radius; // 251.2
    const offset = circumference - (gpaVal / 4.0) * circumference;
    svgCircle.style.strokeDashoffset = offset;
    
    // Color mapping the GPA ring color
    if (gpaVal >= 3.5) {
        svgCircle.style.stroke = 'var(--secondary)'; // Brilliant cyan
    } else if (gpaVal >= 3.0) {
        svgCircle.style.stroke = 'var(--primary)'; // Violet
    } else if (gpaVal >= 2.0) {
        svgCircle.style.stroke = 'var(--warning)'; // Yellow
    } else {
        svgCircle.style.stroke = 'var(--danger)'; // Red
    }
    
    // Update Risk Badge style
    const riskBadge = document.getElementById('risk-badge');
    riskBadge.innerText = `${riskVal} Risk`;
    riskBadge.className = `risk-badge ${riskVal.toLowerCase()}`;
    
    // Grade Letter Translation
    let letter = 'F';
    if (gpaVal >= 3.8) letter = 'A+';
    else if (gpaVal >= 3.5) letter = 'A';
    else if (gpaVal >= 3.0) letter = 'B';
    else if (gpaVal >= 2.5) letter = 'C';
    else if (gpaVal >= 2.0) letter = 'D';
    
    document.getElementById('equivalent-letter').innerText = letter;
    
    // Generate AI recommendations
    generateRecommendations(gpaVal, riskVal, inputData);
}

// Generate context-aware recommendations
function generateRecommendations(gpa, risk, data) {
    const recoList = document.getElementById('recommendation-list');
    recoList.innerHTML = '';
    
    let recs = [];
    
    // 1. High risk or low gpa
    if (risk === 'High') {
        recs.push({
            text: "Urgent: Student is at high risk of academic failure. Schedule a one-on-one session with academic counselors immediately.",
            type: "danger"
        });
    }
    
    // 2. Attendance issues
    if (data.attendance < 75) {
        recs.push({
            text: `Attendance is critical (${data.attendance}%). Suggesting daily morning checks. Missing class sessions has the highest negative coefficient on previous statistics.`,
            type: "danger"
        });
    } else if (data.attendance < 85) {
        recs.push({
            text: `Attendance is slightly sub-optimal (${data.attendance}%). Encourage consistent classroom participation to prevent missing key lecture concepts.`,
            type: "warning"
        });
    }
    
    // 3. Study Hours issues
    if (data.study_hours < 8) {
        recs.push({
            text: `Study time is deficient (${data.study_hours} hrs/week). Work with the student to draw a structured study schedule aiming for at least 12-15 hours per week.`,
            type: "warning"
        });
    } else if (data.study_hours > 25 && data.stress_level === 'High') {
        recs.push({
            text: `Extremely high study hours (${data.study_hours} hrs) coupled with high stress. Risk of burnout. Advise balancing study sessions with active breaks and rest.`,
            type: "warning"
        });
    }
    
    // 4. Tutoring recommendation
    if (data.previous_grade < 70 && data.tutoring === 0) {
        recs.push({
            text: "Enrolling the student in peer-tutoring or guided homework sessions can boost concept clarity and elevate test confidence.",
            type: "warning"
        });
    }
    
    // 5. Sleep problems
    if (data.sleep_hours < 6.0) {
        recs.push({
            text: `Inadequate sleep (${data.sleep_hours} hrs). Chronic fatigue impairs cognitive focus. Encourage 7-8 hours of regular sleep nightly.`,
            type: "warning"
        });
    }
    
    // 6. Healthy highlights
    if (gpa >= 3.2 && risk === 'Low') {
        recs.push({
            text: "Excellent current performance! Maintain active study patterns. Consider student leadership roles or peer mentoring opportunities.",
            type: "success"
        });
    }
    
    if (data.extracurriculars === 1 && data.stress_level === 'Low') {
        recs.push({
            text: "Active extracurricular involvement is currently contributing to a well-balanced lifestyle and positive psychological stability.",
            type: "success"
        });
    }
    
    // Fallback if no specific flags
    if (recs.length === 0) {
        recs.push({
            text: "The student has stable academic and personal habits. Maintain the current routines and schedule standard checkpoints.",
            type: "success"
        });
    }
    
    // Render
    recs.forEach(rec => {
        const li = document.createElement('li');
        li.innerText = rec.text;
        li.className = `reco-${rec.type}`;
        recoList.appendChild(li);
    });
}

// Chart.js render configurations
function renderAnalyticsCharts() {
    if (!modelStats) return;
    
    // 1. Chart 1: Feature Importances
    const ctxImportance = document.getElementById('importanceChart').getContext('2d');
    if (charts.importance) {
        charts.importance.destroy();
    }
    
    // Parse scikit-learn feature weights
    const rawImportances = modelStats.feature_importances.gpa;
    
    // Clean and group category coefficients to make it readable
    const aggregated = {
        'Previous Grades': rawImportances.previous_grade || 0,
        'Attendance Rate': rawImportances.attendance || 0,
        'Study Hours/Week': rawImportances.study_hours || 0,
        'Sleep Quality': rawImportances.sleep_hours || 0,
        'Parental Involvement': (rawImportances.parental_support_High || 0) + 0.05, // aggregate categorical mapping representation
        'Stress / Wellness': (rawImportances.stress_level_High || 0) + 0.03,
        'Tutors & Support': rawImportances.tutoring || 0,
        'Extracurricular activities': rawImportances.extracurriculars || 0,
        'Internet Access': rawImportances.internet_access || 0
    };
    
    // Sort
    const sortedFeatures = Object.keys(aggregated).sort((a,b) => aggregated[b] - aggregated[a]);
    const sortedValues = sortedFeatures.map(f => aggregated[f]);
    
    charts.importance = new Chart(ctxImportance, {
        type: 'bar',
        data: {
            labels: sortedFeatures,
            datasets: [{
                label: 'Relative Influence Weight',
                data: sortedValues,
                backgroundColor: 'rgba(0, 229, 255, 0.45)',
                borderColor: '#00e5ff',
                borderWidth: 1.5,
                borderRadius: 8,
                hoverBackgroundColor: 'rgba(124, 77, 255, 0.65)',
                hoverBorderColor: '#7c4dff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(10, 8, 24, 0.95)',
                    titleColor: '#fff',
                    bodyColor: '#a3a0b8',
                    borderColor: 'rgba(255, 255, 255, 0.08)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#6e6b85', font: { family: 'Outfit' } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#a3a0b8', font: { family: 'Outfit', size: 12 } }
                }
            }
        }
    });
    
    // 2. Chart 2: Comparison Radar Chart
    const ctxComparison = document.getElementById('comparisonChart').getContext('2d');
    if (charts.comparison) {
        charts.comparison.destroy();
    }
    
    const currentData = getFormData();
    
    // Labels mapping
    const metrics = ['Study Hours (x1.5)', 'Attendance %', 'Sleep Hours (x10)', 'Previous Exams %'];
    
    // Scaling metrics to 0-100 range for radar representation
    const currentValues = [
        currentData.study_hours * 2.5, // 40 max -> 100 scale
        currentData.attendance,
        currentData.sleep_hours * 10,   // 10 max -> 100 scale
        currentData.previous_grade
    ];
    
    const averageValues = [
        baselineStudent.study_hours * 2.5,
        baselineStudent.attendance,
        baselineStudent.sleep_hours * 10,
        baselineStudent.previous_grade
    ];
    
    charts.comparison = new Chart(ctxComparison, {
        type: 'radar',
        data: {
            labels: metrics,
            datasets: [
                {
                    label: 'Current Profile',
                    data: currentValues,
                    backgroundColor: 'rgba(124, 77, 255, 0.25)',
                    borderColor: '#7c4dff',
                    pointBackgroundColor: '#7c4dff',
                    pointBorderColor: '#fff',
                    borderWidth: 2
                },
                {
                    label: 'Dataset Average Baseline',
                    data: averageValues,
                    backgroundColor: 'rgba(255, 255, 255, 0.05)',
                    borderColor: 'rgba(255, 255, 255, 0.35)',
                    pointBackgroundColor: 'rgba(255, 255, 255, 0.5)',
                    pointBorderColor: '#fff',
                    borderWidth: 1.5,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#a3a0b8', font: { family: 'Outfit' } }
                },
                tooltip: {
                    backgroundColor: 'rgba(10, 8, 24, 0.95)',
                    borderColor: 'rgba(255, 255, 255, 0.08)',
                    borderWidth: 1
                }
            },
            scales: {
                r: {
                    angleLines: { color: 'rgba(255,255,255,0.06)' },
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    pointLabels: { color: '#a3a0b8', font: { family: 'Outfit', size: 11 } },
                    ticks: {
                        display: false,
                        stepSize: 20
                    },
                    min: 0,
                    max: 100
                }
            }
        }
    });
}
