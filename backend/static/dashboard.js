// AlphaPulse Performance Dashboard JavaScript

// Dashboard state
let performanceChart;
let wsConnection;
let updateInterval;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing AlphaPulse Performance Dashboard...');
    initializeCharts();
    connectWebSocket();
    loadInitialData();
    startAutoRefresh();
});

function initializeCharts() {
    const ctx = document.getElementById('performance-chart');
    if (!ctx) {
        console.warn('Chart canvas not found');
        return;
    }
    
    performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Performance Score',
                data: [],
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.4,
                fill: true,
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 10,
                    ticks: {
                        stepSize: 2,
                        color: '#666'
                    },
                    grid: {
                        color: 'rgba(0,0,0,0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#666',
                        maxRotation: 45
                    },
                    grid: {
                        color: 'rgba(0,0,0,0.1)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#333',
                        font: {
                            size: 12,
                            weight: 'bold'
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#667eea',
                    borderWidth: 1
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
    
    console.log('✅ Performance chart initialized');
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    console.log('🔌 Connecting to WebSocket:', wsUrl);
    
    wsConnection = new WebSocket(wsUrl);
    
    wsConnection.onopen = function() {
        console.log('✅ WebSocket connected');
        showConnectionStatus('connected');
    };
    
    wsConnection.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        } catch (error) {
            console.error('❌ Error parsing WebSocket message:', error);
        }
    };
    
    wsConnection.onclose = function() {
        console.log('🔌 WebSocket disconnected, retrying...');
        showConnectionStatus('disconnected');
        setTimeout(connectWebSocket, 5000);
    };
    
    wsConnection.onerror = function(error) {
        console.error('❌ WebSocket error:', error);
        showConnectionStatus('error');
    };
}

function showConnectionStatus(status) {
    const statusEl = document.getElementById('connection-status');
    if (statusEl) {
        statusEl.textContent = status === 'connected' ? '🟢 Connected' : 
                              status === 'disconnected' ? '🟡 Disconnected' : '🔴 Error';
        statusEl.className = `connection-status ${status}`;
    }
}

async function loadInitialData() {
    try {
        console.log('📊 Loading initial dashboard data...');
        
        const [health, metrics, optStatus] = await Promise.all([
            fetch('/api/health').then(r => r.json()),
            fetch('/api/metrics').then(r => r.json()),
            fetch('/api/optimization-status').then(r => r.json())
        ]);
        
        updateDashboard({
            system_health: health,
            performance: metrics.performance,
            optimization: optStatus
        });
        
        console.log('✅ Initial data loaded successfully');
        
    } catch (error) {
        console.error('❌ Error loading initial data:', error);
        showError('Failed to load initial data. Please refresh the page.');
    }
}

function updateDashboard(data) {
    // Update system health
    if (data.system_health) {
        updateSystemHealth(data.system_health);
    }
    
    // Update performance metrics
    if (data.performance) {
        updatePerformanceMetrics(data.performance);
    }
    
    // Update optimization status
    if (data.optimization) {
        updateOptimizationStatus(data.optimization);
    }
    
    // Update chart
    if (data.performance && data.performance.overall_score) {
        updatePerformanceChart(data.performance.overall_score);
    }
    
    // Update timestamp
    updateLastUpdate();
}

function updateSystemHealth(health) {
    const overallScore = document.getElementById('overall-score');
    const dbHealth = document.getElementById('db-health');
    const perfScore = document.getElementById('perf-score');
    const optStatus = document.getElementById('opt-status');
    
    if (overallScore) overallScore.textContent = `${health.overall_score.toFixed(1)}%`;
    if (perfScore) perfScore.textContent = `${health.performance_score.toFixed(2)}/10`;
    if (optStatus) optStatus.textContent = health.optimization_status;
    
    if (dbHealth) {
        dbHealth.className = 'metric-value';
        dbHealth.innerHTML = `<span class="status-indicator status-${health.database_health}"></span>${health.database_health}`;
    }
    
    // Add visual feedback based on health score
    if (overallScore) {
        overallScore.className = `metric-value ${getHealthClass(health.overall_score)}`;
    }
}

function updatePerformanceMetrics(perf) {
    const totalPatterns = document.getElementById('total-patterns');
    const fastQueries = document.getElementById('fast-queries');
    const indexUsage = document.getElementById('index-usage');
    
    if (totalPatterns) totalPatterns.textContent = formatNumber(perf.total_patterns);
    if (fastQueries) fastQueries.textContent = perf.fast_queries || 0;
    
    if (indexUsage && perf.index_scans !== undefined && perf.seq_scans !== undefined) {
        const total = perf.index_scans + perf.seq_scans;
        const percentage = total > 0 ? Math.round((perf.index_scans / total) * 100) : 0;
        indexUsage.textContent = `${perf.index_scans}/${total} (${percentage}%)`;
        indexUsage.className = `metric-value ${percentage >= 80 ? 'success' : percentage >= 60 ? 'warning' : 'error'}`;
    }
}

function updateOptimizationStatus(opt) {
    const monitoringActive = document.getElementById('monitoring-active');
    const autoOptimization = document.getElementById('auto-optimization');
    const pendingRecommendations = document.getElementById('pending-recommendations');
    const activeAlerts = document.getElementById('active-alerts');
    
    if (monitoringActive) {
        monitoringActive.textContent = opt.monitoring_active ? '✅ Active' : '❌ Inactive';
        monitoringActive.className = `metric-value ${opt.monitoring_active ? 'success' : 'error'}`;
    }
    
    if (autoOptimization) {
        autoOptimization.textContent = opt.auto_optimization_enabled ? '✅ Enabled' : '❌ Disabled';
        autoOptimization.className = `metric-value ${opt.auto_optimization_enabled ? 'success' : 'warning'}`;
    }
    
    if (pendingRecommendations) {
        pendingRecommendations.textContent = opt.pending_recommendations || 0;
        pendingRecommendations.className = `metric-value ${(opt.pending_recommendations || 0) === 0 ? 'success' : 'warning'}`;
    }
    
    if (activeAlerts) {
        activeAlerts.textContent = opt.active_alerts || 0;
        activeAlerts.className = `metric-value ${(opt.active_alerts || 0) === 0 ? 'success' : 'error'}`;
    }
}

function updatePerformanceChart(score) {
    if (!performanceChart) return;
    
    const now = new Date();
    const timeLabel = now.toLocaleTimeString();
    
    performanceChart.data.labels.push(timeLabel);
    performanceChart.data.datasets[0].data.push(score);
    
    // Keep only last 20 data points
    if (performanceChart.data.labels.length > 20) {
        performanceChart.data.labels.shift();
        performanceChart.data.datasets[0].data.shift();
    }
    
    performanceChart.update('none');
}

function updateLastUpdate() {
    const lastUpdate = document.getElementById('last-update');
    if (lastUpdate) {
        lastUpdate.textContent = new Date().toLocaleTimeString();
    }
}

function getHealthClass(score) {
    if (score >= 90) return 'success';
    if (score >= 75) return 'info';
    if (score >= 60) return 'warning';
    return 'error';
}

function formatNumber(num) {
    if (num === null || num === undefined) return '--';
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function formatTime(timestamp) {
    if (!timestamp) return '--';
    try {
        const date = new Date(timestamp);
        return date.toLocaleTimeString();
    } catch (error) {
        return '--';
    }
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert error';
    errorDiv.textContent = message;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(errorDiv, container.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }
}

function startAutoRefresh() {
    // Fallback refresh every 30 seconds if WebSocket fails
    updateInterval = setInterval(() => {
        if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
            // WebSocket is working, no need for fallback
            return;
        }
        
        console.log('🔄 WebSocket not available, using fallback refresh');
        loadInitialData();
    }, 30000);
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    if (wsConnection) {
        wsConnection.close();
    }
});

// Export functions for debugging
window.dashboardDebug = {
    updateDashboard,
    loadInitialData,
    connectWebSocket,
    performanceChart
};
