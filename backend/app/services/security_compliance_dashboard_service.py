#!/usr/bin/env python3
"""
Security & Compliance Dashboard Service for AlphaPulse
Provides web interface for monitoring and controlling security operations
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.security_compliance import (
    get_encryption_manager,
    get_rbac_manager,
    get_audit_logger,
    get_threat_detector,
    Permission,
    SecurityLevel,
    ThreatLevel
)

logger = logging.getLogger(__name__)
security = HTTPBearer()

class SecurityComplianceDashboardService:
    """FastAPI service for security and compliance dashboard"""
    
    def __init__(self):
        self.app = FastAPI(
            title="AlphaPulse Security & Compliance",
            description="Enterprise security, RBAC, audit logging, and threat detection",
            version="1.0.0"
        )
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_security_dashboard():
            """Get the main security dashboard HTML"""
            return self._get_security_dashboard_html()
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
        
        @self.app.get("/api/security/status")
        async def get_security_status():
            """Get overall security status"""
            try:
                threat_detector = get_threat_detector()
                audit_logger = get_audit_logger()
                
                # Get threat summary
                threat_summary = threat_detector.get_threat_summary(24)
                
                # Get recent security events
                recent_security_events = audit_logger.get_security_events(limit=10)
                
                # Get recent audit logs
                recent_audit_logs = audit_logger.get_audit_logs(limit=10)
                
                return {
                    "threat_summary": threat_summary,
                    "recent_security_events": recent_security_events,
                    "recent_audit_logs": recent_audit_logs,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                logger.error(f"❌ Error getting security status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/security/threats")
        async def get_threats(time_window_hours: int = 24):
            """Get threat analysis for specified time window"""
            try:
                threat_detector = get_threat_detector()
                threat_summary = threat_detector.get_threat_summary(time_window_hours)
                return threat_summary
                
            except Exception as e:
                logger.error(f"❌ Error getting threats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/security/events")
        async def get_security_events(
            threat_level: Optional[str] = None,
            user_id: Optional[str] = None,
            limit: int = 100
        ):
            """Get filtered security events"""
            try:
                audit_logger = get_audit_logger()
                
                # Convert string to ThreatLevel enum if provided
                threat_level_enum = None
                if threat_level:
                    try:
                        threat_level_enum = ThreatLevel(threat_level)
                    except ValueError:
                        raise HTTPException(status_code=400, detail=f"Invalid threat level: {threat_level}")
                
                events = audit_logger.get_security_events(
                    threat_level=threat_level_enum,
                    user_id=user_id,
                    limit=limit
                )
                
                return {"security_events": events}
                
            except Exception as e:
                logger.error(f"❌ Error getting security events: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/audit/logs")
        async def get_audit_logs(
            user_id: Optional[str] = None,
            action: Optional[str] = None,
            resource: Optional[str] = None,
            limit: int = 100
        ):
            """Get filtered audit logs"""
            try:
                audit_logger = get_audit_logger()
                
                logs = audit_logger.get_audit_logs(
                    user_id=user_id,
                    action=action,
                    resource=resource,
                    limit=limit
                )
                
                return {"audit_logs": logs}
                
            except Exception as e:
                logger.error(f"❌ Error getting audit logs: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/users")
        async def get_users():
            """Get all users (admin only)"""
            try:
                rbac_manager = get_rbac_manager()
                
                # In a real implementation, verify admin permissions here
                users = []
                for user in rbac_manager.users.values():
                    user_dict = {
                        "user_id": user.user_id,
                        "username": user.username,
                        "email": user.email,
                        "roles": user.roles,
                        "is_active": user.is_active,
                        "created_at": user.created_at.isoformat() if user.created_at else None,
                        "last_login": user.last_login.isoformat() if user.last_login else None
                    }
                    users.append(user_dict)
                
                return {"users": users}
                
            except Exception as e:
                logger.error(f"❌ Error getting users: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/roles")
        async def get_roles():
            """Get all roles"""
            try:
                rbac_manager = get_rbac_manager()
                
                roles = []
                for role in rbac_manager.roles.values():
                    role_dict = {
                        "role_id": role.role_id,
                        "name": role.name,
                        "description": role.description,
                        "permissions": [p.value for p in role.permissions],
                        "security_level": role.security_level.value,
                        "is_active": role.is_active
                    }
                    roles.append(role_dict)
                
                return {"roles": roles}
                
            except Exception as e:
                logger.error(f"❌ Error getting roles: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/encryption/rotate-keys")
        async def rotate_encryption_keys():
            """Rotate encryption keys (admin only)"""
            try:
                encryption_manager = get_encryption_manager()
                success = encryption_manager.rotate_keys()
                
                if success:
                    return {"message": "Encryption keys rotated successfully", "success": True}
                else:
                    return {"message": "Key rotation not needed at this time", "success": False}
                    
            except Exception as e:
                logger.error(f"❌ Error rotating encryption keys: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/encryption/encrypt")
        async def encrypt_data(data: Dict[str, Any]):
            """Encrypt data"""
            try:
                encryption_manager = get_encryption_manager()
                
                text_data = data.get("data")
                security_level = SecurityLevel(data.get("security_level", "medium"))
                
                if not text_data:
                    raise HTTPException(status_code=400, detail="Data field is required")
                
                encrypted_package = encryption_manager.encrypt_data(text_data, security_level)
                return {"encrypted_package": encrypted_package}
                
            except Exception as e:
                logger.error(f"❌ Error encrypting data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/encryption/decrypt")
        async def decrypt_data(encrypted_package: Dict[str, Any]):
            """Decrypt data"""
            try:
                encryption_manager = get_encryption_manager()
                
                decrypted_data = encryption_manager.decrypt_data(encrypted_package)
                return {"decrypted_data": decrypted_data}
                
            except Exception as e:
                logger.error(f"❌ Error decrypting data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _get_security_dashboard_html(self) -> str:
        """Generate the security dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaPulse Security & Compliance</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: #333;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .security-item {
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            border-left: 5px solid #3498db;
            background: #f8f9fa;
            transition: all 0.3s ease;
        }
        
        .security-item.critical {
            border-left-color: #e74c3c;
            background: #fdf2f2;
        }
        
        .security-item.high {
            border-left-color: #f39c12;
            background: #fef9e7;
        }
        
        .security-item.medium {
            border-left-color: #f39c12;
            background: #fef9e7;
        }
        
        .security-item.low {
            border-left-color: #27ae60;
            background: #f0f9f0;
        }
        
        .security-item.info {
            border-left-color: #3498db;
            background: #f0f8ff;
        }
        
        .threat-level {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
            text-transform: uppercase;
        }
        
        .level-critical {
            background: #f8d7da;
            color: #721c24;
        }
        
        .level-high {
            background: #fff3cd;
            color: #856404;
        }
        
        .level-medium {
            background: #d1ecf1;
            color: #0c5460;
        }
        
        .level-low {
            background: #d4edda;
            color: #155724;
        }
        
        .level-info {
            background: #e2e3e5;
            color: #383d41;
        }
        
        .security-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 10px 0;
            font-size: 0.9rem;
        }
        
        .metric-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
        }
        
        .metric-label {
            color: #7f8c8d;
        }
        
        .metric-value {
            font-weight: 500;
            color: #2c3e50;
        }
        
        .action-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background 0.3s ease;
            margin-top: 10px;
        }
        
        .action-btn:hover {
            background: #2980b9;
        }
        
        .action-btn.danger {
            background: #e74c3c;
        }
        
        .action-btn.danger:hover {
            background: #c0392b;
        }
        
        .status-section {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .status-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            background: #f8f9fa;
            border-left: 4px solid #3498db;
        }
        
        .status-success {
            border-left-color: #27ae60;
            background: #f0f9f0;
        }
        
        .status-warning {
            border-left-color: #f39c12;
            background: #fef9e7;
        }
        
        .status-danger {
            border-left-color: #e74c3c;
            background: #fdf2f2;
        }
        
        .refresh-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.3s ease;
            margin-bottom: 20px;
        }
        
        .refresh-btn:hover {
            background: #2980b9;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #7f8c8d;
        }
        
        .error {
            background: #fdf2f2;
            color: #c0392b;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .encryption-section {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .encryption-input {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 5px 0;
        }
        
        .encryption-select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 5px 0;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔐 AlphaPulse Security & Compliance</h1>
            <p>Enterprise security, RBAC, audit logging, and threat detection for your trading bot</p>
        </div>
        
        <div style="text-align: center; margin-bottom: 20px;">
            <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh Dashboard</button>
            <span style="margin-left: 20px; color: white; opacity: 0.8;" id="lastUpdate"></span>
        </div>
        
        <div id="dashboardContent">
            <div class="loading">Loading security dashboard...</div>
        </div>
    </div>
    
    <script>
        let dashboardData = null;
        
        async function loadDashboard() {
            try {
                const response = await fetch('/api/security/status');
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                dashboardData = await response.json();
                updateDashboard();
                
            } catch (error) {
                console.error('Error loading dashboard:', error);
                document.getElementById('dashboardContent').innerHTML = 
                    `<div class="error">Error loading dashboard: ${error.message}</div>`;
            }
        }
        
        function updateDashboard() {
            if (!dashboardData) return;
            
            const content = document.getElementById('dashboardContent');
            
            // Update last update time
            const lastUpdate = new Date();
            document.getElementById('lastUpdate').textContent = 
                `Last updated: ${lastUpdate.toLocaleString()}`;
            
            // Build dashboard content
            let html = '';
            
            // Threat Summary Section
            const threatSummary = dashboardData.threat_summary || {};
            html += `
                <div class="status-section">
                    <h3 style="margin-bottom: 20px; color: #2c3e50;">🚨 Threat Analysis Summary (Last 24 Hours)</h3>
                    <div class="dashboard-grid">
                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">⚠️ Total Threats</span>
                            </div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: #e74c3c;">
                                ${threatSummary.total_threats || 0}
                            </div>
                            <small style="color: #7f8c8d;">Threats detected in the last 24 hours</small>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">🔍 Threat Types</span>
                            </div>
                            <div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50;">
                                ${Object.keys(threatSummary.threat_breakdown || {}).length}
                            </div>
                            <small style="color: #7f8c8d;">Different types of threats detected</small>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">📊 Time Window</span>
                            </div>
                            <div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50;">
                                ${threatSummary.time_window_hours || 24}h
                            </div>
                            <small style="color: #7f8c8d;">Analysis time window</small>
                        </div>
                    </div>
                </div>
            `;
            
            // Threat Breakdown
            const threatBreakdown = threatSummary.threat_breakdown || {};
            if (Object.keys(threatBreakdown).length > 0) {
                html += `
                    <div class="status-section">
                        <h3 style="margin-bottom: 20px; color: #2c3e50;">📈 Threat Breakdown</h3>
                        <div class="dashboard-grid">
                `;
                
                Object.entries(threatBreakdown).forEach(([threatType, count]) => {
                    const threatColor = threatType === 'brute_force' ? '#e74c3c' : 
                                       threatType === 'privilege_escalation' ? '#f39c12' : '#3498db';
                    
                    html += `
                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">${threatType.replace('_', ' ').toUpperCase()}</span>
                            </div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: ${threatColor};">
                                ${count}
                            </div>
                            <small style="color: #7f8c8d;">Occurrences detected</small>
                        </div>
                    `;
                });
                
                html += '</div></div>';
            }
            
            // Recent Security Events
            const recentSecurityEvents = dashboardData.recent_security_events || [];
            if (recentSecurityEvents.length > 0) {
                html += `
                    <div class="status-section">
                        <h3 style="margin-bottom: 20px; color: #2c3e50;">🔒 Recent Security Events</h3>
                `;
                
                recentSecurityEvents.forEach(event => {
                    const threatLevel = event.threat_level || 'info';
                    const levelClass = `level-${threatLevel}`;
                    const statusClass = event.success ? 'status-success' : 'status-danger';
                    const statusIcon = event.success ? '✅' : '❌';
                    
                    html += `
                        <div class="status-item ${statusClass}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong>${statusIcon} ${event.action}</strong> on ${event.resource}<br>
                                    <small>User: ${event.user_id || 'Unknown'} | IP: ${event.ip_address || 'Unknown'}</small>
                                </div>
                                <div class="threat-level ${levelClass}">${threatLevel.toUpperCase()}</div>
                            </div>
                        </div>
                    `;
                });
                
                html += '</div>';
            }
            
            // Encryption Tools Section
            html += `
                <div class="status-section">
                    <h3 style="margin-bottom: 20px; color: #2c3e50;">🔐 Encryption Tools</h3>
                    <div class="dashboard-grid">
                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">🔒 Encrypt Data</span>
                            </div>
                            <div class="encryption-section">
                                <input type="text" id="encryptInput" class="encryption-input" placeholder="Enter text to encrypt">
                                <select id="securityLevel" class="encryption-select">
                                    <option value="low">Low Security</option>
                                    <option value="medium" selected>Medium Security</option>
                                    <option value="high">High Security</option>
                                    <option value="critical">Critical Security</option>
                                </select>
                                <button class="action-btn" onclick="encryptData()">🔒 Encrypt</button>
                                <div id="encryptResult" style="margin-top: 10px; word-break: break-all;"></div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">🔓 Decrypt Data</span>
                            </div>
                            <div class="encryption-section">
                                <textarea id="decryptInput" class="encryption-input" placeholder="Paste encrypted package (JSON)" rows="4"></textarea>
                                <button class="action-btn" onclick="decryptData()">🔓 Decrypt</button>
                                <div id="decryptResult" style="margin-top: 10px; word-break: break-all;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            // Recent Audit Logs
            const recentAuditLogs = dashboardData.recent_audit_logs || [];
            if (recentAuditLogs.length > 0) {
                html += `
                    <div class="status-section">
                        <h3 style="margin-bottom: 20px; color: #2c3e50;">📝 Recent Audit Logs</h3>
                `;
                
                recentAuditLogs.forEach(log => {
                    const complianceTags = log.compliance_tags || [];
                    const tagsHtml = complianceTags.length > 0 ? 
                        `<small style="color: #7f8c8d;">Tags: ${complianceTags.join(', ')}</small>` : '';
                    
                    html += `
                        <div class="status-item">
                            <strong>${log.action}</strong> on ${log.resource}<br>
                            <small>User: ${log.user_id} | IP: ${log.ip_address} | Time: ${new Date(log.timestamp).toLocaleString()}</small>
                            ${tagsHtml}
                        </div>
                    `;
                });
                
                html += '</div>';
            }
            
            content.innerHTML = html;
        }
        
        async function encryptData() {
            try {
                const input = document.getElementById('encryptInput').value;
                const securityLevel = document.getElementById('securityLevel').value;
                
                if (!input.trim()) {
                    alert('Please enter text to encrypt');
                    return;
                }
                
                const response = await fetch('/api/encryption/encrypt', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        data: input,
                        security_level: securityLevel
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const result = await response.json();
                document.getElementById('encryptResult').innerHTML = 
                    `<strong>Encrypted:</strong><br><code>${JSON.stringify(result.encrypted_package, null, 2)}</code>`;
                
            } catch (error) {
                console.error('Error encrypting data:', error);
                document.getElementById('encryptResult').innerHTML = 
                    `<span style="color: #e74c3c;">❌ Encryption failed: ${error.message}</span>`;
            }
        }
        
        async function decryptData() {
            try {
                const input = document.getElementById('decryptInput').value;
                
                if (!input.trim()) {
                    alert('Please enter encrypted package to decrypt');
                    return;
                }
                
                let encryptedPackage;
                try {
                    encryptedPackage = JSON.parse(input);
                } catch (e) {
                    alert('Invalid JSON format');
                    return;
                }
                
                const response = await fetch('/api/encryption/decrypt', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(encryptedPackage)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const result = await response.json();
                document.getElementById('decryptResult').innerHTML = 
                    `<strong>Decrypted:</strong><br><code>${result.decrypted_data}</code>`;
                
            } catch (error) {
                console.error('Error decrypting data:', error);
                document.getElementById('decryptResult').innerHTML = 
                    `<span style="color: #e74c3c;">❌ Decryption failed: ${error.message}</span>`;
            }
        }
        
        function refreshDashboard() {
            loadDashboard();
        }
        
        // Auto-refresh every 30 seconds
        setInterval(loadDashboard, 30000);
        
        // Load dashboard on page load
        document.addEventListener('DOMContentLoaded', loadDashboard);
    </script>
</body>
</html>
        """

# Global service instance
_security_service = None

def get_security_service() -> SecurityComplianceDashboardService:
    """Get the global security dashboard service instance"""
    global _security_service
    if _security_service is None:
        _security_service = SecurityComplianceDashboardService()
    return _security_service

def get_security_app() -> FastAPI:
    """Get the FastAPI app for security and compliance"""
    service = get_security_service()
    return service.app
