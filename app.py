import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Model loading logic with fallback dummy model to prevent crashes if file is missing
MODEL_PATH = "linear_model.pkl"

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Successfully loaded model from 'linear_model.pkl'.")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        model = None
else:
    print("Warning: 'linear_model.pkl' not found in root directory. Running in demonstration mode.")
    model = None

# Exact order of feature names embedded in the linear_model.pkl file
FEATURE_KEYS = [
    "number of bedrooms",
    "number of bathrooms",
    "living area",
    "lot area",
    "number of floors",
    "waterfront present",
    "number of views",
    "condition of the house",
    "grade of the house",
    "Area of the house(excluding basement)",
    "Area of the basement",
    "Built Year",
    "Renovation Year",
    "lot_area_renov",
    "Number of schools nearby",
    "Distance from the airport"
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real Estate Price Intelligence Dashboard</title>
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        :root {
            /* Colorful Professional Dark Theme Defaults */
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            --panel-bg: rgba(30, 41, 59, 0.7);
            --panel-border: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --input-border: rgba(255, 255, 255, 0.15);
            --card-glow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
            
            /* Color Accents */
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.4);
            --secondary: #ec4899;
            --accent-cyan: #06b6d4;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-purple: #8b5cf6;
        }

        [data-theme="light"] {
            --bg-gradient: linear-gradient(135deg, #f1f5f9 0%, #e0e7ff 50%, #f1f5f9 100%);
            --panel-bg: rgba(255, 255, 255, 0.85);
            --panel-border: rgba(226, 232, 240, 0.8);
            --text-main: #0f172a;
            --text-muted: #64748b;
            --input-bg: rgba(248, 250, 252, 0.9);
            --input-border: rgba(203, 213, 225, 0.8);
            --card-glow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
            
            --primary: #4f46e5;
            --primary-glow: rgba(79, 70, 229, 0.2);
            --secondary: #db2777;
            --accent-cyan: #0891b2;
            --accent-emerald: #059669;
            --accent-amber: #d97706;
            --accent-purple: #7c3aed;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--bg-gradient);
            background-attachment: fixed;
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem 1rem;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        /* Top Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            padding: 1.25rem 2rem;
            border-radius: 20px;
            border: 1px solid var(--panel-border);
            box-shadow: var(--card-glow);
        }

        .logo-area {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .logo-icon {
            width: 48px;
            height: 48px;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: #fff;
            box-shadow: 0 4px 15px var(--primary-glow);
        }

        .logo-text h1 {
            font-size: 1.4rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, var(--text-main), var(--text-muted));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .logo-text p {
            font-size: 0.825rem;
            color: var(--text-muted);
        }

        /* Theme Toggle Button */
        .theme-toggle-btn {
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            color: var(--text-main);
            padding: 0.6rem 1.2rem;
            border-radius: 30px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-weight: 600;
            font-size: 0.875rem;
        }

        .theme-toggle-btn:hover {
            border-color: var(--primary);
            color: var(--primary);
        }

        /* Dashboard Grid Layout */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Glassmorphism Card Style */
        .card {
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--panel-border);
            border-radius: 24px;
            padding: 1.75rem;
            box-shadow: var(--card-glow);
            margin-bottom: 2rem;
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--panel-border);
        }

        .card-header i {
            font-size: 1.25rem;
            color: var(--primary);
        }

        .card-header h2 {
            font-size: 1.15rem;
            font-weight: 700;
        }

        /* Form Inputs Grid */
        .inputs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .input-field-wrapper {
            position: relative;
        }

        .input-field-wrapper i {
            position: absolute;
            left: 1rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        .input-control {
            width: 100%;
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            padding: 0.75rem 1rem 0.75rem 2.5rem;
            border-radius: 12px;
            color: var(--text-main);
            font-family: inherit;
            font-size: 0.925rem;
            font-weight: 600;
            outline: none;
        }

        .input-control:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--primary-glow);
        }

        /* Predict Button */
        .submit-btn {
            width: 100%;
            margin-top: 1.5rem;
            padding: 1rem;
            border: none;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            font-family: inherit;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            box-shadow: 0 10px 20px var(--primary-glow);
            position: relative;
            overflow: hidden;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 24px var(--primary-glow);
        }

        .submit-btn:active {
            transform: translateY(0);
        }

        /* Valuation Result Display */
        .valuation-card {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(236, 72, 153, 0.15));
            border: 1px solid var(--primary-glow);
            text-align: center;
            padding: 2rem;
            position: relative;
            overflow: hidden;
        }

        .valuation-title {
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            font-weight: 700;
        }

        .valuation-price {
            font-family: 'JetBrains Mono', monospace;
            font-size: 3rem;
            font-weight: 800;
            margin: 0.5rem 0;
            background: linear-gradient(135deg, var(--accent-cyan), var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: pulseText 3s infinite alternate;
        }

        @keyframes pulseText {
            0% { filter: drop-shadow(0 0 2px rgba(99, 102, 241, 0.2)); }
            100% { filter: drop-shadow(0 0 15px rgba(236, 72, 153, 0.6)); }
        }

        .valuation-subtitle {
            font-size: 0.85rem;
            color: var(--accent-emerald);
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(16, 185, 129, 0.1);
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
        }

        /* Chart Wrappers */
        .chart-container {
            position: relative;
            height: 280px;
            width: 100%;
        }

        /* Stats Pills */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .stat-card {
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            padding: 1rem;
            border-radius: 16px;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .stat-icon {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }

        .stat-info h4 {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .stat-info p {
            font-size: 1.1rem;
            font-weight: 700;
        }

        /* Animated Loading Overlay */
        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

<div class="container">
    <!-- Header Navigation -->
    <header>
        <div class="logo-area">
            <div class="logo-icon">
                <i class="fa-solid fa-chart-line"></i>
            </div>
            <div class="logo-text">
                <h1>ValuaTech AI</h1>
                <p>Linear Regression Analytics Engine</p>
            </div>
        </div>
        <button class="theme-toggle-btn" onclick="toggleTheme()" id="themeBtn">
            <i class="fa-solid fa-sun"></i> Light Mode
        </button>
    </header>

    <!-- Main Content Layout -->
    <div class="dashboard-grid">
        
        <!-- Left Panel: Form Parameters -->
        <div class="left-panel">
            <div class="card">
                <div class="card-header">
                    <i class="fa-solid fa-sliders"></i>
                    <h2>Property Parameters</h2>
                </div>
                
                <form id="predictionForm">
                    <div class="inputs-grid">
                        
                        <div class="input-group">
                            <label><i class="fa-solid fa-bed"></i> Bedrooms</label>
                            <div class="input-field-wrapper">
                                <i class="fa-solid fa-hashtag"></i>
                                <input type="number" class="input-control" name="number of bedrooms" value="3" step="1" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label><i class="fa-solid fa-bath"></i> Bathrooms</label>
                            <div class="input-field-wrapper">
                                <i class="fa-solid fa-hashtag"></i>
                                <input type="number" class="input-control" name="number of bathrooms" value="2.25" step="0.25" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label><i class="fa-solid fa-ruler-combined"></i> Living Area (sqft)</label>
                            <div class="input-field-wrapper">
                                <i class="fa-solid fa-vector-square"></i>
                                <input type="number" class="input-control" name="living area" value="1800" step="10" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label><i class="fa-solid fa-chart-area"></i> Lot Area (sqft)</label>
                            <div class="input-field-wrapper">
                                <i class="fa-solid fa-border-all"></i>
                                <input type="number" class="input-control" name="lot area" value="5000" step="10" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label><i class="fa-solid fa-building"></i> Floors</label>
                            <div class="input-field-wrapper">
                                <i class="fa-solid fa-layer-group"></i>
                                <input type="number" class="input-control" name="number of floors" value="1.5" step="0.5" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label><i class="fa-solid fa-water"></i> Waterfront (0/1)</label>
                            <div class="input-field-wrapper">
                                <i class="fa-solid fa-toggle-on"></i>
                                <input type="number" class="input-control" name="waterfront present" value="0" min="0" max="1" step="1" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label><i class="fa-solid fa-eye"></i> Views Rating (0-4)</label>
                            <div class="input-field-wrapper">
                                <i class="fa-solid fa-star"></i>
                                <input type="number" class="input-control" name="number of views" value="0" min="0" max="4" step="1" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label><i class="fa-solid fa-wrench"></i> Condition (1-5)</label>
                            <div class="input-field-wrapper">
                                <i class="fa-solid fa-thumbs-up"></i>
                                <input type="number" class="input-control" name="condition of the house" value="3" min="1" max="5" step="1" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label><i class="fa-solid fa-award"></i> Grade (1-13)</label>
                            <div class="input-field-wrapper">
                                <i class="fa-solid fa-medal"></i>
                                <input type="number" class="input-control" name="grade of the house" value="7" min="1" max="13" step="1" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label><i class="fa-solid fa-house"></i> Area Excl. Basement</label>
                            <div class="input-field-wrapper">
                                <i class="fa-solid fa-expand"></i>
                                <input type="number" class="input-control" name="Area of the house(excluding basement)" value="1500" step="10" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label><i class="fa-solid fa-dungeon"></i> Basement Area</label>
                            <div class="input-field-wrapper">
                                <i class="fa-solid fa-square"></i>
                                <input type="number" class="input-control" name="Area of the basement" value="300" step="10" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label><i class="fa-solid fa-calendar"></i> Built Year</label>
                            <div class="input-field-wrapper">
                                <i class="fa-solid fa-hammer"></i>
                                <input type="number" class="input-control" name="Built Year" value="1995" step="1" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label><i class="fa-solid fa-paint-roller"></i> Renovation Year</label>
                            <div class="input-field-wrapper">
                                <i class="fa-solid fa-clock-rotate-left"></i>
                                <input type="number" class="input-control" name="Renovation Year" value="0" step="1" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label><i class="fa-solid fa-tree"></i> Renovated Lot Area</label>
                            <div class="input-field-wrapper">
                                <i class="fa-solid fa-map-location-dot"></i>
                                <input type="number" class="input-control" name="lot_area_renov" value="4500" step="10" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label><i class="fa-solid fa-school"></i> Schools Nearby</label>
                            <div class="input-field-wrapper">
                                <i class="fa-solid fa-graduation-cap"></i>
                                <input type="number" class="input-control" name="Number of schools nearby" value="3" step="1" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label><i class="fa-solid fa-plane"></i> Airport Distance</label>
                            <div class="input-field-wrapper">
                                <i class="fa-solid fa-plane-departure"></i>
                                <input type="number" class="input-control" name="Distance from the airport" value="12" step="1" required>
                            </div>
                        </div>

                    </div>

                    <button type="submit" class="submit-btn" id="submitBtn">
                        <div class="spinner" id="btnSpinner"></div>
                        <span id="btnText"><i class="fa-solid fa-bolt"></i> Calculate Market Estimate</span>
                    </button>
                </form>
            </div>
        </div>

        <!-- Right Panel: Valuation & Charts Output -->
        <div class="right-panel">
            
            <!-- Valuation Result -->
            <div class="card valuation-card">
                <div class="valuation-title">Estimated Property Value</div>
                <div class="valuation-price" id="predictedPrice">$0.00</div>
                <div class="valuation-subtitle">
                    <i class="fa-solid fa-circle-check"></i> ML Model Confidence Active
                </div>
            </div>

            <!-- Stats Overview -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-icon" style="background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan);">
                        <i class="fa-solid fa-chart-pie"></i>
                    </div>
                    <div class="stat-info">
                        <h4>Space Ratio</h4>
                        <p id="statRatio">0.36</p>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon" style="background: rgba(245, 158, 11, 0.15); color: var(--accent-amber);">
                        <i class="fa-solid fa-house-user"></i>
                    </div>
                    <div class="stat-info">
                        <h4>Quality Index</h4>
                        <p id="statQuality">7 / 13</p>
                    </div>
                </div>
            </div>

            <!-- Radar Breakdown Chart -->
            <div class="card">
                <div class="card-header">
                    <i class="fa-solid fa-chart-radar"></i>
                    <h2>Property Profile Radar</h2>
                </div>
                <div class="chart-container">
                    <canvas id="radarChart"></canvas>
                </div>
            </div>

            <!-- Bar Comparison Chart -->
            <div class="card">
                <div class="card-header">
                    <i class="fa-solid fa-chart-bar"></i>
                    <h2>Key Feature Distribution</h2>
                </div>
                <div class="chart-container">
                    <canvas id="barChart"></canvas>
                </div>
            </div>

        </div>
    </div>
</div>

<script>
    // Theme Switcher Logic
    function toggleTheme() {
        const htmlElement = document.documentElement;
        const themeBtn = document.getElementById('themeBtn');
        const currentTheme = htmlElement.getAttribute('data-theme');
        
        if (currentTheme === 'dark') {
            htmlElement.setAttribute('data-theme', 'light');
            themeBtn.innerHTML = '<i class="fa-solid fa-moon"></i> Dark Mode';
            updateChartColors(false);
        } else {
            htmlElement.setAttribute('data-theme', 'dark');
            themeBtn.innerHTML = '<i class="fa-solid fa-sun"></i> Light Mode';
            updateChartColors(true);
        }
    }

    // Chart.js Setup
    let radarChart, barChart;

    function initCharts() {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        const textColor = isDark ? '#94a3b8' : '#64748b';
        const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';

        // Radar Chart Definition
        const ctxRadar = document.getElementById('radarChart').getContext('2d');
        radarChart = new Chart(ctxRadar, {
            type: 'radar',
            data: {
                labels: ['Bedrooms', 'Bathrooms', 'Views', 'Condition', 'Grade', 'Schools'],
                datasets: [{
                    label: 'Selected Property',
                    data: [3, 2.25, 0, 3, 7, 3],
                    backgroundColor: 'rgba(99, 102, 241, 0.25)',
                    borderColor: '#6366f1',
                    pointBackgroundColor: '#ec4899',
                    pointBorderColor: '#fff',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: gridColor },
                        grid: { color: gridColor },
                        pointLabels: { color: textColor, font: { family: 'Plus Jakarta Sans', size: 11 } },
                        ticks: { backdropColor: 'transparent', color: textColor }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });

        // Bar Chart Definition
        const ctxBar = document.getElementById('barChart').getContext('2d');
        barChart = new Chart(ctxBar, {
            type: 'bar',
            data: {
                labels: ['Living Area', 'Excl. Basement', 'Basement'],
                datasets: [{
                    data: [1800, 1500, 300],
                    backgroundColor: [
                        'rgba(99, 102, 241, 0.85)',
                        'rgba(6, 182, 212, 0.85)',
                        'rgba(236, 72, 153, 0.85)'
                    ],
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { display: false }, ticks: { color: textColor } },
                    y: { grid: { color: gridColor }, ticks: { color: textColor } }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    function updateChartColors(isDark) {
        const textColor = isDark ? '#94a3b8' : '#64748b';
        const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';

        if (radarChart && barChart) {
            radarChart.options.scales.r.angleLines.color = gridColor;
            radarChart.options.scales.r.grid.color = gridColor;
            radarChart.options.scales.r.pointLabels.color = textColor;
            radarChart.options.scales.r.ticks.color = textColor;
            radarChart.update();

            barChart.options.scales.x.ticks.color = textColor;
            barChart.options.scales.y.grid.color = gridColor;
            barChart.options.scales.y.ticks.color = textColor;
            barChart.update();
        }
    }

    // Dynamic Form Submission & AJAX Request
    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const btnSpinner = document.getElementById('btnSpinner');
        const btnText = document.getElementById('btnText');
        
        btnSpinner.style.display = 'block';
        btnText.style.opacity = '0.7';

        const formData = new FormData(this);
        const dataObj = {};
        formData.forEach((value, key) => dataObj[key] = parseFloat(value));

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'json' },
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(dataObj)
            });

            const result = await response.json();

            if (result.success) {
                // Animate Counter Effect
                animateValue("predictedPrice", result.prediction);

                // Update Stats
                const living = dataObj["living area"] || 1;
                const lot = dataObj["lot area"] || 1;
                document.getElementById('statRatio').innerText = (living / lot).toFixed(2);
                document.getElementById('statQuality').innerText = `${dataObj["grade of the house"] || 0} / 13`;

                // Update Radar Chart
                radarChart.data.datasets[0].data = [
                    dataObj["number of bedrooms"],
                    dataObj["number of bathrooms"],
                    dataObj["number of views"],
                    dataObj["condition of the house"],
                    dataObj["grade of the house"],
                    dataObj["Number of schools nearby"]
                ];
                radarChart.update();

                // Update Bar Chart
                barChart.data.datasets[0].data = [
                    dataObj["living area"],
                    dataObj["Area of the house(excluding basement)"],
                    dataObj["Area of the basement"]
                ];
                barChart.update();
            }
        } catch (err) {
            console.error(err);
        } finally {
            btnSpinner.style.display = 'none';
            btnText.style.opacity = '1';
        }
    });

    function animateValue(id, finalVal) {
        const obj = document.getElementById(id);
        const startVal = 0;
        const duration = 1000;
        let startTimestamp = null;

        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const currentVal = Math.floor(progress * (finalVal - startVal) + startVal);
            
            obj.innerText = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                maximumFractionDigits: 0
            }).format(currentVal);

            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    // Initialize charts on page load
    window.onload = initCharts;
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Build features array matching exact sequence
        features = []
        for feature_name in FEATURE_KEYS:
            val = float(data.get(feature_name, 0.0))
            features.append(val)

        # Convert to 2D numpy array
        input_data = np.array([features])

        # Model Prediction
        if model is not None:
            prediction = model.predict(input_data)[0]
        else:
            # Fallback estimation logic if pickle file is not present
            prediction = sum(features) * 150.0

        return jsonify({
            "success": True,
            "prediction": float(prediction)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
