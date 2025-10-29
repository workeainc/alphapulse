#!/usr/bin/env python3
"""
Security & Compliance Framework for AlphaPulse
Provides data encryption, RBAC, audit logging, and threat detection
"""

import asyncio
import logging
import hashlib
import hmac
import secrets
import time
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import base64
import jwt

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatLevel(Enum):
    """Threat levels for security events"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Permission(Enum):
    """System permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    AUDIT = "audit"
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"

@dataclass
class User:
    """User entity with roles and permissions"""
    user_id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[Permission]
    is_active: bool = True
    created_at: datetime = None
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None

@dataclass
class Role:
    """Role with associated permissions"""
    role_id: str
    name: str
    description: str
    permissions: List[Permission]
    security_level: SecurityLevel
    is_active: bool = True

@dataclass
class SecurityEvent:
    """Security event record"""
    event_id: str
    timestamp: datetime
    user_id: Optional[str]
    action: str
    resource: str
    threat_level: ThreatLevel
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True

@dataclass
class AuditLog:
    """Audit log entry"""
    log_id: str
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    session_id: str
    compliance_tags: List[str]

class EncryptionManager:
    """Manages data encryption and key management"""
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or self._generate_master_key()
        self.fernet = Fernet(self.master_key)
        self.key_rotation_interval = timedelta(days=90)
        self.last_rotation = datetime.now(timezone.utc)
        self.logger = logging.getLogger(__name__)
        
        # Generate RSA key pair for asymmetric encryption
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
    
    def _generate_master_key(self) -> bytes:
        """Generate a new master encryption key"""
        return Fernet.generate_key()
    
    def encrypt_data(self, data: Union[str, bytes], security_level: SecurityLevel = SecurityLevel.MEDIUM) -> Dict[str, Any]:
        """Encrypt data with appropriate security level"""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Encrypt the data
            encrypted_data = self.fernet.encrypt(data)
            
            # Add metadata
            result = {
                "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
                "security_level": security_level.value,
                "encryption_method": "Fernet",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "key_version": "1.0"
            }
            
            self.logger.info(f"✅ Data encrypted successfully (level: {security_level.value})")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_package: Dict[str, Any]) -> Union[str, bytes]:
        """Decrypt data from encrypted package"""
        try:
            encrypted_data = base64.b64decode(encrypted_package["encrypted_data"])
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            # Try to decode as string, return bytes if it fails
            try:
                return decrypted_data.decode('utf-8')
            except UnicodeDecodeError:
                return decrypted_data
                
        except Exception as e:
            self.logger.error(f"❌ Decryption failed: {e}")
            raise
    
    def encrypt_asymmetric(self, data: Union[str, bytes]) -> Dict[str, Any]:
        """Encrypt data using RSA public key"""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted_data = self.public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return {
                "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
                "encryption_method": "RSA-OAEP",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Asymmetric encryption failed: {e}")
            raise
    
    def decrypt_asymmetric(self, encrypted_package: Dict[str, Any]) -> Union[str, bytes]:
        """Decrypt data using RSA private key"""
        try:
            encrypted_data = base64.b64decode(encrypted_package["encrypted_data"])
            
            decrypted_data = self.private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            try:
                return decrypted_data.decode('utf-8')
            except UnicodeDecodeError:
                return decrypted_data
                
        except Exception as e:
            self.logger.error(f"❌ Asymmetric decryption failed: {e}")
            raise
    
    def rotate_keys(self) -> bool:
        """Rotate encryption keys"""
        try:
            if datetime.now(timezone.utc) - self.last_rotation < self.key_rotation_interval:
                return False
            
            # Generate new keys
            new_master_key = self._generate_master_key()
            new_fernet = Fernet(new_master_key)
            
            # Update keys
            self.master_key = new_master_key
            self.fernet = new_fernet
            self.last_rotation = datetime.now(timezone.utc)
            
            self.logger.info("✅ Encryption keys rotated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Key rotation failed: {e}")
            return False

class RoleBasedAccessControl:
    """Role-based access control system"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Initialize default roles
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize default system roles"""
        default_roles = [
            Role(
                role_id="admin",
                name="Administrator",
                description="Full system access",
                permissions=[p for p in Permission],
                security_level=SecurityLevel.CRITICAL
            ),
            Role(
                role_id="user",
                name="Standard User",
                description="Basic user access",
                permissions=[Permission.READ, Permission.WRITE],
                security_level=SecurityLevel.MEDIUM
            ),
            Role(
                role_id="viewer",
                name="Viewer",
                description="Read-only access",
                permissions=[Permission.READ],
                security_level=SecurityLevel.LOW
            ),
            Role(
                role_id="auditor",
                name="Auditor",
                description="Audit and compliance access",
                permissions=[Permission.READ, Permission.AUDIT],
                security_level=SecurityLevel.HIGH
            )
        ]
        
        for role in default_roles:
            self.roles[role.role_id] = role
    
    def create_user(self, username: str, email: str, roles: List[str], password_hash: str) -> str:
        """Create a new user"""
        with self.lock:
            user_id = self._generate_user_id()
            
            # Get permissions from roles
            permissions = []
            for role_id in roles:
                if role_id in self.roles:
                    permissions.extend(self.roles[role_id].permissions)
            
            # Remove duplicates
            permissions = list(set(permissions))
            
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                roles=roles,
                permissions=permissions,
                created_at=datetime.now(timezone.utc)
            )
            
            self.users[user_id] = user
            self.logger.info(f"✅ User created: {username} with roles: {roles}")
            
            return user_id
    
    def _generate_user_id(self) -> str:
        """Generate unique user ID"""
        return f"user_{int(time.time() * 1000)}_{secrets.token_hex(4)}"
    
    def authenticate_user(self, username: str, password_hash: str) -> Optional[str]:
        """Authenticate user and return session token"""
        with self.lock:
            # Find user by username
            user = None
            for u in self.users.values():
                if u.username == username and u.is_active:
                    user = u
                    break
            
            if not user:
                self.logger.warning(f"⚠️ Authentication failed: User not found - {username}")
                return None
            
            # Check if account is locked
            if user.locked_until and datetime.now(timezone.utc) < user.locked_until:
                self.logger.warning(f"⚠️ Account locked: {username}")
                return None
            
            # Verify password (in real implementation, use proper password verification)
            # For now, we'll assume the password_hash is correct
            
            # Reset failed attempts on successful login
            user.failed_attempts = 0
            user.last_login = datetime.now(timezone.utc)
            
            # Generate session token
            session_token = self._generate_session_token(user.user_id)
            
            self.logger.info(f"✅ User authenticated: {username}")
            return session_token
    
    def _generate_session_token(self, user_id: str) -> str:
        """Generate JWT session token"""
        payload = {
            "user_id": user_id,
            "exp": datetime.now(timezone.utc) + timedelta(hours=24),
            "iat": datetime.now(timezone.utc)
        }
        
        # In production, use a proper secret key
        secret = "your-secret-key-here"
        token = jwt.encode(payload, secret, algorithm="HS256")
        
        # Store session
        self.sessions[token] = {
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc)
        }
        
        return token
    
    def verify_permission(self, session_token: str, permission: Permission, resource: str) -> bool:
        """Verify if user has permission for resource"""
        try:
            # Decode session token
            secret = "your-secret-key-here"
            payload = jwt.decode(session_token, secret, algorithms=["HS256"])
            user_id = payload["user_id"]
            
            # Check if session exists and is valid
            if session_token not in self.sessions:
                return False
            
            session = self.sessions[session_token]
            if session["user_id"] != user_id:
                return False
            
            # Update last activity
            session["last_activity"] = datetime.now(timezone.utc)
            
            # Check user permissions
            if user_id not in self.users:
                return False
            
            user = self.users[user_id]
            if not user.is_active:
                return False
            
            return permission in user.permissions
            
        except Exception as e:
            self.logger.error(f"❌ Permission verification failed: {e}")
            return False
    
    def get_user_info(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get user information from session token"""
        try:
            secret = "your-secret-key-here"
            payload = jwt.decode(session_token, secret, algorithms=["HS256"])
            user_id = payload["user_id"]
            
            if user_id not in self.users:
                return None
            
            user = self.users[user_id]
            return asdict(user)
            
        except Exception as e:
            self.logger.error(f"❌ Get user info failed: {e}")
            return None

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, max_logs: int = 10000):
        self.audit_logs: List[AuditLog] = []
        self.security_events: List[SecurityEvent] = []
        self.max_logs = max_logs
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Compliance tags for different regulations
        self.compliance_tags = {
            "gdpr": ["data_protection", "privacy", "consent"],
            "sox": ["financial", "audit", "controls"],
            "pci": ["payment", "card", "security"],
            "hipaa": ["health", "privacy", "security"]
        }
    
    def log_audit_event(self, user_id: str, action: str, resource: str, details: Dict[str, Any], 
                        ip_address: str, user_agent: str, session_id: str, 
                        compliance_requirements: List[str] = None) -> str:
        """Log an audit event"""
        with self.lock:
            log_id = f"audit_{int(time.time() * 1000)}_{secrets.token_hex(4)}"
            
            # Determine compliance tags
            compliance_tags = []
            if compliance_requirements:
                for req in compliance_requirements:
                    if req.lower() in self.compliance_tags:
                        compliance_tags.extend(self.compliance_tags[req.lower()])
            
            audit_log = AuditLog(
                log_id=log_id,
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                action=action,
                resource=resource,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=session_id,
                compliance_tags=compliance_tags
            )
            
            self.audit_logs.append(audit_log)
            
            # Maintain log size
            if len(self.audit_logs) > self.max_logs:
                self.audit_logs = self.audit_logs[-self.max_logs:]
            
            self.logger.info(f"📝 Audit event logged: {action} on {resource} by {user_id}")
            return log_id
    
    def log_security_event(self, user_id: Optional[str], action: str, resource: str, 
                          threat_level: ThreatLevel, details: Dict[str, Any], 
                          ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                          success: bool = True) -> str:
        """Log a security event"""
        with self.lock:
            event_id = f"security_{int(time.time() * 1000)}_{secrets.token_hex(4)}"
            
            security_event = SecurityEvent(
                event_id=event_id,
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                action=action,
                resource=resource,
                threat_level=threat_level,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent,
                success=success
            )
            
            self.security_events.append(security_event)
            
            # Maintain event size
            if len(self.security_events) > self.max_logs:
                self.security_events = self.security_events[-self.max_logs:]
            
            # Log high threat events immediately
            if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                self.logger.warning(f"🚨 High threat security event: {action} on {resource} - {threat_level.value}")
            
            return event_id
    
    def get_audit_logs(self, user_id: Optional[str] = None, action: Optional[str] = None,
                       resource: Optional[str] = None, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get filtered audit logs"""
        with self.lock:
            filtered_logs = self.audit_logs
            
            if user_id:
                filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
            
            if action:
                filtered_logs = [log for log in filtered_logs if log.action == action]
            
            if resource:
                filtered_logs = [log for log in filtered_logs if log.resource == resource]
            
            if start_time:
                filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
            
            if end_time:
                filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]
            
            # Sort by timestamp (newest first)
            filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
            
            return [asdict(log) for log in filtered_logs[:limit]]
    
    def get_security_events(self, threat_level: Optional[ThreatLevel] = None,
                           user_id: Optional[str] = None, start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get filtered security events"""
        with self.lock:
            filtered_events = self.security_events
            
            if threat_level:
                filtered_events = [event for event in filtered_events if event.threat_level == threat_level]
            
            if user_id:
                filtered_events = [event for event in filtered_events if event.user_id == user_id]
            
            if start_time:
                filtered_events = [event for event in filtered_events if event.timestamp >= start_time]
            
            if end_time:
                filtered_events = [event for event in filtered_events if event.timestamp <= end_time]
            
            # Sort by timestamp (newest first)
            filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
            
            return [asdict(event) for event in filtered_events[:limit]]

class ThreatDetector:
    """Threat detection and analysis system"""
    
    def __init__(self):
        self.threat_patterns: Dict[str, Dict[str, Any]] = {}
        self.anomaly_thresholds: Dict[str, float] = {}
        self.suspicious_activities: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Initialize threat patterns
        self._initialize_threat_patterns()
    
    def _initialize_threat_patterns(self):
        """Initialize common threat patterns"""
        self.threat_patterns = {
            "brute_force": {
                "description": "Multiple failed login attempts",
                "threshold": 5,
                "time_window": 300,  # 5 minutes
                "threat_level": ThreatLevel.HIGH
            },
            "data_exfiltration": {
                "description": "Unusual data access patterns",
                "threshold": 1000,
                "time_window": 3600,  # 1 hour
                "threat_level": ThreatLevel.CRITICAL
            },
            "privilege_escalation": {
                "description": "Unauthorized permission access",
                "threshold": 1,
                "time_window": 86400,  # 24 hours
                "threat_level": ThreatLevel.CRITICAL
            },
            "anomalous_timing": {
                "description": "Unusual operation timing",
                "threshold": 0.8,  # 80% deviation
                "time_window": 3600,
                "threat_level": ThreatLevel.MEDIUM
            }
        }
    
    def analyze_security_event(self, event: SecurityEvent) -> Optional[Dict[str, Any]]:
        """Analyze security event for threats"""
        try:
            threats_detected = []
            
            # Check for brute force attempts
            if event.action == "login_failed":
                brute_force_threat = self._check_brute_force(event)
                if brute_force_threat:
                    threats_detected.append(brute_force_threat)
            
            # Check for privilege escalation
            if event.action == "permission_denied":
                privilege_threat = self._check_privilege_escalation(event)
                if privilege_threat:
                    threats_detected.append(privilege_threat)
            
            # Check for data exfiltration
            if event.action == "data_access":
                exfiltration_threat = self._check_data_exfiltration(event)
                if exfiltration_threat:
                    threats_detected.append(exfiltration_threat)
            
            if threats_detected:
                threat_summary = {
                    "event_id": event.event_id,
                    "threats": threats_detected,
                    "timestamp": event.timestamp,
                    "user_id": event.user_id,
                    "ip_address": event.ip_address
                }
                
                with self.lock:
                    self.suspicious_activities.append(threat_summary)
                
                self.logger.warning(f"🚨 Threats detected in event {event.event_id}: {len(threats_detected)} threats")
                return threat_summary
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Threat analysis failed: {e}")
            return None
    
    def _check_brute_force(self, event: SecurityEvent) -> Optional[Dict[str, Any]]:
        """Check for brute force attack patterns"""
        # This would typically check against a database of recent events
        # For now, we'll return a simple check
        if event.details.get("failed_attempts", 0) >= 5:
            return {
                "type": "brute_force",
                "description": "Multiple failed login attempts detected",
                "threat_level": ThreatLevel.HIGH,
                "recommendation": "Lock account and investigate IP address"
            }
        return None
    
    def _check_privilege_escalation(self, event: SecurityEvent) -> Optional[Dict[str, Any]]:
        """Check for privilege escalation attempts"""
        if event.details.get("requested_permission") in [Permission.ADMIN, Permission.ENCRYPT]:
            return {
                "type": "privilege_escalation",
                "description": "Unauthorized access to sensitive permissions",
                "threat_level": ThreatLevel.CRITICAL,
                "recommendation": "Immediate account suspension and investigation"
            }
        return None
    
    def _check_data_exfiltration(self, event: SecurityEvent) -> Optional[Dict[str, Any]]:
        """Check for data exfiltration patterns"""
        data_volume = event.details.get("data_volume", 0)
        if data_volume > 1000:  # More than 1000 records
            return {
                "type": "data_exfiltration",
                "description": "Unusual large data access detected",
                "threat_level": ThreatLevel.CRITICAL,
                "recommendation": "Review access patterns and suspend if suspicious"
            }
        return None
    
    def get_threat_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get threat summary for specified time window"""
        with self.lock:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
            
            recent_threats = [
                activity for activity in self.suspicious_activities
                if activity["timestamp"] >= cutoff_time
            ]
            
            threat_counts = {}
            for activity in recent_threats:
                for threat in activity["threats"]:
                    threat_type = threat["type"]
                    threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
            
            return {
                "time_window_hours": time_window_hours,
                "total_threats": len(recent_threats),
                "threat_breakdown": threat_counts,
                "recent_activities": recent_threats[-10:]  # Last 10 activities
            }

# Global instances
_encryption_manager = None
_rbac_manager = None
_audit_logger = None
_threat_detector = None

def get_encryption_manager() -> EncryptionManager:
    """Get the global encryption manager instance"""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager

def get_rbac_manager() -> RoleBasedAccessControl:
    """Get the global RBAC manager instance"""
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RoleBasedAccessControl()
    return _rbac_manager

def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger

def get_threat_detector() -> ThreatDetector:
    """Get the global threat detector instance"""
    global _threat_detector
    if _threat_detector is None:
        _threat_detector = ThreatDetector()
    return _threat_detector

# Security decorators
def require_permission(permission: Permission, resource: str = "default"):
    """Decorator to require specific permission"""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # Extract session token from kwargs or request context
            session_token = kwargs.get('session_token')
            if not session_token:
                raise PermissionError("Session token required")
            
            rbac = get_rbac_manager()
            if not rbac.verify_permission(session_token, permission, resource):
                raise PermissionError(f"Permission {permission.value} required for {resource}")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def audit_action(action: str, resource: str, compliance_requirements: List[str] = None):
    """Decorator to audit function calls"""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # Extract audit information from kwargs
            user_id = kwargs.get('user_id', 'system')
            ip_address = kwargs.get('ip_address', 'unknown')
            user_agent = kwargs.get('user_agent', 'unknown')
            session_id = kwargs.get('session_id', 'unknown')
            
            # Log the action
            audit_logger = get_audit_logger()
            audit_logger.log_audit_event(
                user_id=user_id,
                action=action,
                resource=resource,
                details={"function": func.__name__, "args": str(args), "kwargs": str(kwargs)},
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=session_id,
                compliance_requirements=compliance_requirements
            )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
