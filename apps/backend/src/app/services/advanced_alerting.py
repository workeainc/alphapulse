import asyncio
import json
import logging
import time
import smtplib
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"

class NotificationChannel(Enum):
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SLACK = "slack"

@dataclass
class AlertRule:
    rule_id: str
    name: str
    severity: AlertSeverity
    conditions: Dict[str, Any]
    actions: List[str]
    enabled: bool

@dataclass
class Alert:
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    status: AlertStatus
    created_at: datetime
    escalation_level: int

class AdvancedAlertingService:
    """Advanced alerting service with notifications and escalation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Configuration
        self.email_config = config.get('email', {})
        self.sms_config = config.get('sms', {})
        self.webhook_config = config.get('webhook', {})
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'notifications_sent': 0,
            'escalations': 0
        }
        
        self._initialize_default_rules()
        logger.info("AdvancedAlertingService initialized")
    
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule('high_memory', 'High Memory Usage', AlertSeverity.WARNING,
                     {'metric': 'memory_usage', 'threshold': 85}, ['email'], True),
            AlertRule('high_cpu', 'High CPU Usage', AlertSeverity.WARNING,
                     {'metric': 'cpu_usage', 'threshold': 90}, ['email'], True),
            AlertRule('low_throughput', 'Low Throughput', AlertSeverity.ERROR,
                     {'metric': 'patterns_per_second', 'threshold': 100}, ['email', 'sms'], True)
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    async def start(self):
        """Start the alerting service"""
        logger.info("Starting advanced alerting service...")
        asyncio.create_task(self._monitor_alerts())
        logger.info("Advanced alerting service started")
    
    async def _monitor_alerts(self):
        """Monitor alerts and send notifications"""
        while True:
            try:
                for alert_id, alert in self.active_alerts.items():
                    if alert.status == AlertStatus.ACTIVE:
                        await self._send_notifications(alert)
                
                await asyncio.sleep(30)
            except Exception as e:
                logger.error("Error in alert monitor: %s", e)
                await asyncio.sleep(60)
    
    def create_alert(self, rule_id: str, message: str, details: Dict[str, Any] = None) -> str:
        """Create a new alert"""
        if rule_id not in self.alert_rules:
            return None
        
        rule = self.alert_rules[rule_id]
        if not rule.enabled:
            return None
        
        alert_id = f"alert_{int(time.time())}"
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule_id,
            severity=rule.severity,
            message=message,
            details=details or {},
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(timezone.utc),
            escalation_level=1
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.stats['total_alerts'] += 1
        
        logger.info("Created alert %s: %s", alert_id, message)
        return alert_id
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        rule = self.alert_rules[alert.rule_id]
        
        for action in rule.actions:
            try:
                if action == 'email':
                    await self._send_email(alert)
                elif action == 'sms':
                    await self._send_sms(alert)
                elif action == 'webhook':
                    await self._send_webhook(alert)
                
                self.stats['notifications_sent'] += 1
                
            except Exception as e:
                logger.error("Error sending %s notification: %s", action, e)
    
    async def _send_email(self, alert: Alert):
        """Send email notification"""
        if not self.email_config:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config.get('from_email', 'alerts@company.com')
            msg['To'] = ', '.join(self.email_config.get('to_emails', ['team@company.com']))
            msg['Subject'] = f"ALERT: {alert.severity.value.upper()} - {alert.message}"
            
            body = f"""
            Alert: {alert.message}
            Severity: {alert.severity.value}
            Time: {alert.created_at.isoformat()}
            Details: {json.dumps(alert.details, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.email_config.get('smtp_host', 'localhost'), 
                             self.email_config.get('smtp_port', 587)) as server:
                if self.email_config.get('use_tls', True):
                    server.starttls()
                
                if self.email_config.get('username') and self.email_config.get('password'):
                    server.login(self.email_config['username'], self.email_config['password'])
                
                server.send_message(msg)
            
            logger.info("Email notification sent for alert %s", alert.alert_id)
            
        except Exception as e:
            logger.error("Error sending email: %s", e)
    
    async def _send_sms(self, alert: Alert):
        """Send SMS notification"""
        if not self.sms_config:
            return
        
        try:
            message = f"ALERT: {alert.message}"
            url = self.sms_config.get('api_url')
            api_key = self.sms_config.get('api_key')
            phone_numbers = self.sms_config.get('phone_numbers', [])
            
            if url and api_key and phone_numbers:
                for phone in phone_numbers:
                    payload = {
                        'api_key': api_key,
                        'to': phone,
                        'message': message
                    }
                    
                    response = requests.post(url, json=payload, timeout=10)
                    if response.status_code == 200:
                        logger.info("SMS sent to %s", phone)
            
        except Exception as e:
            logger.error("Error sending SMS: %s", e)
    
    async def _send_webhook(self, alert: Alert):
        """Send webhook notification"""
        if not self.webhook_config:
            return
        
        try:
            payload = {
                'alert_id': alert.alert_id,
                'severity': alert.severity.value,
                'message': alert.message,
                'timestamp': alert.created_at.isoformat(),
                'details': alert.details
            }
            
            webhook_urls = self.webhook_config.get('urls', [])
            for url in webhook_urls:
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code in [200, 201, 202]:
                    logger.info("Webhook sent to %s", url)
            
        except Exception as e:
            logger.error("Error sending webhook: %s", e)
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [asdict(alert) for alert in self.active_alerts.values()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            'stats': self.stats,
            'active_alerts_count': len(self.active_alerts),
            'total_alerts_count': len(self.alert_history)
        }
    
    async def stop(self):
        """Stop the alerting service"""
        logger.info("Advanced alerting service stopped")
