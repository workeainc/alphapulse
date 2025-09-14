"""
Security Manager for AlphaPlus
Handles audit logging, access control, secrets management, and security monitoring
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
import json
import ipaddress
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

class SecurityManager:
    """Comprehensive security management for AlphaPlus"""
    
    def __init__(self, async_engine: AsyncEngine):
        self.async_engine = async_engine
        self.logger = logger
        
        # Security configuration
        self.audit_logging_enabled = True
        self.access_control_enabled = True
        self.secrets_management_enabled = True
        self.security_monitoring_enabled = True
        
        # Security thresholds
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 15
        self.session_timeout_minutes = 30
        self.alert_threshold = 10
        
        # Security state
        self.failed_attempts = {}  # user_id -> count
        self.locked_users = {}     # user_id -> lock_until
        self.active_sessions = {}  # session_id -> session_data
    
    async def initialize(self):
        """Initialize the security manager"""
        try:
            # Validate database tables exist
            await self._validate_security_tables()
            
            # Initialize default security policies
            await self._initialize_default_policies()
            
            # Start security monitoring
            if self.security_monitoring_enabled:
                await self._start_security_monitoring()
            
            self.logger.info("✅ Security manager initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize security manager: {e}")
            return False
    
    async def _validate_security_tables(self):
        """Validate that security tables exist"""
        try:
            async with self.async_engine.begin() as conn:
                # Check if security tables exist
                result = await conn.execute(text("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_name IN (
                        'security_audit_logs', 'security_access_control', 
                        'security_secrets_metadata', 'security_events', 'security_policies'
                    )
                """))
                tables = [row[0] for row in result.fetchall()]
                
                if len(tables) != 5:
                    raise RuntimeError(f"Missing security tables. Found: {tables}")
                
                self.logger.info("✅ Security tables validated")
                
        except Exception as e:
            self.logger.error(f"❌ Security tables validation failed: {e}")
            raise
    
    async def _initialize_default_policies(self):
        """Initialize default security policies"""
        try:
            async with self.async_engine.begin() as conn:
                # Check if default policies exist
                result = await conn.execute(text("""
                    SELECT COUNT(*) FROM security_policies 
                    WHERE policy_name LIKE 'default_%'
                """))
                policy_count = result.scalar()
                
                if policy_count == 0:
                    self.logger.info("Creating default security policies...")
                    # Default policies are created in the migration
                
                self.logger.info("✅ Default security policies initialized")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize default policies: {e}")
    
    async def _start_security_monitoring(self):
        """Start security monitoring tasks"""
        try:
            # Start background monitoring tasks
            asyncio.create_task(self._monitor_security_events())
            asyncio.create_task(self._monitor_failed_attempts())
            asyncio.create_task(self._monitor_secret_rotation())
            
            self.logger.info("✅ Security monitoring started")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start security monitoring: {e}")
    
    async def log_audit_event(self, user_id: str, session_id: str, action_type: str,
                            resource_type: str = None, resource_id: str = None,
                            action_details: dict = None, ip_address: str = None,
                            user_agent: str = None, request_method: str = None,
                            request_path: str = None, response_status: int = None,
                            execution_time_ms: int = None, success: bool = True,
                            error_message: str = None, metadata: dict = None) -> int:
        """Log security audit event"""
        try:
            if not self.audit_logging_enabled:
                return 0
            
            # Validate IP address
            validated_ip = None
            if ip_address:
                try:
                    validated_ip = str(ipaddress.ip_address(ip_address))
                except ValueError:
                    self.logger.warning(f"Invalid IP address: {ip_address}")
            
            # Prepare action details
            action_details_json = json.dumps(action_details) if action_details else None
            metadata_json = json.dumps(metadata) if metadata else None
            
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("""
                    SELECT log_security_audit(
                        :user_id, :session_id, :action_type, :resource_type, :resource_id,
                        :action_details, :ip_address, :user_agent, :request_method, :request_path,
                        :response_status, :execution_time_ms, :success, :error_message, :metadata
                    )
                """), {
                    'user_id': user_id,
                    'session_id': session_id,
                    'action_type': action_type,
                    'resource_type': resource_type,
                    'resource_id': resource_id,
                    'action_details': action_details_json,
                    'ip_address': validated_ip,
                    'user_agent': user_agent,
                    'request_method': request_method,
                    'request_path': request_path,
                    'response_status': response_status,
                    'execution_time_ms': execution_time_ms,
                    'success': success,
                    'error_message': error_message,
                    'metadata': metadata_json
                })
                
                audit_id = result.scalar()
                
                # Log failed attempts for access control
                if not success and action_type in ['login', 'authentication']:
                    await self._record_failed_attempt(user_id, ip_address)
                
                self.logger.debug(f"✅ Audit event logged: {audit_id}")
                return audit_id
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log audit event: {e}")
            return 0
    
    async def check_permission(self, user_id: str, permission: str,
                             resource_type: str = None, resource_id: str = None) -> bool:
        """Check if user has specific permission"""
        try:
            if not self.access_control_enabled:
                return True
            
            # Check if user is locked
            if await self._is_user_locked(user_id):
                return False
            
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("""
                    SELECT check_user_permission(:user_id, :permission, :resource_type, :resource_id)
                """), {
                    'user_id': user_id,
                    'permission': permission,
                    'resource_type': resource_type,
                    'resource_id': resource_id
                })
                
                has_permission = result.scalar()
                return bool(has_permission)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to check permission: {e}")
            return False
    
    async def log_security_event(self, event_type: str, severity: str, source: str,
                               user_id: str = None, session_id: str = None,
                               event_details: dict = None, ip_address: str = None,
                               user_agent: str = None, threat_level: str = None) -> int:
        """Log security event"""
        try:
            if not self.security_monitoring_enabled:
                return 0
            
            # Validate IP address
            validated_ip = None
            if ip_address:
                try:
                    validated_ip = str(ipaddress.ip_address(ip_address))
                except ValueError:
                    self.logger.warning(f"Invalid IP address: {ip_address}")
            
            # Prepare event details
            event_details_json = json.dumps(event_details) if event_details else None
            
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("""
                    SELECT log_security_event(
                        :event_type, :severity, :source, :user_id, :session_id,
                        :event_details, :ip_address, :user_agent, :threat_level
                    )
                """), {
                    'event_type': event_type,
                    'severity': severity,
                    'source': source,
                    'user_id': user_id,
                    'session_id': session_id,
                    'event_details': event_details_json,
                    'ip_address': validated_ip,
                    'user_agent': user_agent,
                    'threat_level': threat_level
                })
                
                event_id = result.scalar()
                
                # Check if event requires immediate attention
                if severity in ['high', 'critical']:
                    await self._handle_high_severity_event(event_type, severity, user_id, ip_address)
                
                self.logger.info(f"✅ Security event logged: {event_id} ({severity})")
                return event_id
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log security event: {e}")
            return 0
    
    async def rotate_secret(self, secret_name: str, new_version: str) -> bool:
        """Rotate a secret"""
        try:
            if not self.secrets_management_enabled:
                return False
            
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("""
                    SELECT rotate_secret(:secret_name, :new_version)
                """), {
                    'secret_name': secret_name,
                    'new_version': new_version
                })
                
                success = result.scalar()
                
                if success:
                    self.logger.info(f"✅ Secret rotated: {secret_name} -> {new_version}")
                else:
                    self.logger.warning(f"❌ Failed to rotate secret: {secret_name}")
                
                return bool(success)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to rotate secret: {e}")
            return False
    
    async def get_statistics(self, days_back: int = 30) -> dict:
        """Get security statistics"""
        try:
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("""
                    SELECT get_security_statistics(:days_back)
                """), {
                    'days_back': days_back
                })
                
                stats = result.scalar()
                return json.loads(stats) if stats else {}
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get security statistics: {e}")
            return {}
    
    async def _record_failed_attempt(self, user_id: str, ip_address: str = None):
        """Record a failed authentication attempt"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Increment failed attempts count
            if user_id not in self.failed_attempts:
                self.failed_attempts[user_id] = {'count': 0, 'first_attempt': current_time}
            
            self.failed_attempts[user_id]['count'] += 1
            self.failed_attempts[user_id]['last_attempt'] = current_time
            self.failed_attempts[user_id]['ip_address'] = ip_address
            
            # Check if user should be locked
            if self.failed_attempts[user_id]['count'] >= self.max_failed_attempts:
                lock_until = current_time + timedelta(minutes=self.lockout_duration_minutes)
                self.locked_users[user_id] = lock_until
                
                # Log security event
                await self.log_security_event(
                    event_type='account_locked',
                    severity='medium',
                    source='access_control',
                    user_id=user_id,
                    ip_address=ip_address,
                    event_details={
                        'failed_attempts': self.failed_attempts[user_id]['count'],
                        'lock_until': lock_until.isoformat(),
                        'reason': 'max_failed_attempts_exceeded'
                    },
                    threat_level='medium'
                )
                
                self.logger.warning(f"🔒 User {user_id} locked until {lock_until}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to record failed attempt: {e}")
    
    async def _is_user_locked(self, user_id: str) -> bool:
        """Check if user is currently locked"""
        try:
            if user_id not in self.locked_users:
                return False
            
            lock_until = self.locked_users[user_id]
            current_time = datetime.now(timezone.utc)
            
            if current_time > lock_until:
                # Lock expired, remove from locked users
                del self.locked_users[user_id]
                if user_id in self.failed_attempts:
                    del self.failed_attempts[user_id]
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to check user lock status: {e}")
            return False
    
    async def _handle_high_severity_event(self, event_type: str, severity: str,
                                        user_id: str = None, ip_address: str = None):
        """Handle high severity security events"""
        try:
            # Log additional details
            self.logger.warning(f"🚨 HIGH SEVERITY EVENT: {event_type} ({severity})")
            
            # Could implement additional actions here:
            # - Send immediate alerts
            # - Block IP addresses
            # - Trigger incident response
            # - Notify security team
            
        except Exception as e:
            self.logger.error(f"❌ Failed to handle high severity event: {e}")
    
    async def _monitor_security_events(self):
        """Monitor security events for patterns and threats"""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                # Get recent security events
                stats = await self.get_statistics(days_back=1)
                
                # Check for unusual patterns
                if stats.get('security_events', {}).get('high_severity', 0) > self.alert_threshold:
                    await self.log_security_event(
                        event_type='high_severity_threshold_exceeded',
                        severity='medium',
                        source='security_monitoring',
                        event_details={'threshold': self.alert_threshold, 'actual': stats['security_events']['high_severity']},
                        threat_level='medium'
                    )
                
        except Exception as e:
            self.logger.error(f"❌ Security event monitoring failed: {e}")
    
    async def _monitor_failed_attempts(self):
        """Monitor failed authentication attempts"""
        try:
            while True:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                current_time = datetime.now(timezone.utc)
                
                # Clean up expired locks and failed attempts
                for user_id in list(self.locked_users.keys()):
                    if current_time > self.locked_users[user_id]:
                        del self.locked_users[user_id]
                        if user_id in self.failed_attempts:
                            del self.failed_attempts[user_id]
                
                # Clean up old failed attempts (older than 1 hour)
                for user_id in list(self.failed_attempts.keys()):
                    if 'last_attempt' in self.failed_attempts[user_id]:
                        if current_time - self.failed_attempts[user_id]['last_attempt'] > timedelta(hours=1):
                            del self.failed_attempts[user_id]
                
        except Exception as e:
            self.logger.error(f"❌ Failed attempts monitoring failed: {e}")
    
    async def _monitor_secret_rotation(self):
        """Monitor secret rotation schedules"""
        try:
            while True:
                await asyncio.sleep(3600)  # Check every hour
                
                async with self.async_engine.begin() as conn:
                    result = await conn.execute(text("""
                        SELECT secret_name, next_rotation_at 
                        FROM security_secrets_metadata 
                        WHERE is_active = true AND next_rotation_at < NOW() + INTERVAL '7 days'
                    """))
                    
                    secrets_due = result.fetchall()
                    
                    for secret_name, next_rotation in secrets_due:
                        days_until_rotation = (next_rotation - datetime.now(timezone.utc)).days
                        
                        if days_until_rotation <= 0:
                            await self.log_security_event(
                                event_type='secret_rotation_overdue',
                                severity='medium',
                                source='secrets_management',
                                event_details={'secret_name': secret_name, 'days_overdue': abs(days_until_rotation)},
                                threat_level='low'
                            )
                        elif days_until_rotation <= 3:
                            await self.log_security_event(
                                event_type='secret_rotation_due_soon',
                                severity='low',
                                source='secrets_management',
                                event_details={'secret_name': secret_name, 'days_until_rotation': days_until_rotation},
                                threat_level='low'
                            )
                
        except Exception as e:
            self.logger.error(f"❌ Secret rotation monitoring failed: {e}")
    
    async def get_user_security_status(self, user_id: str) -> dict:
        """Get comprehensive security status for a user"""
        try:
            status = {
                'user_id': user_id,
                'is_locked': await self._is_user_locked(user_id),
                'failed_attempts': self.failed_attempts.get(user_id, {}).get('count', 0),
                'lock_until': self.locked_users.get(user_id),
                'has_active_session': user_id in [session.get('user_id') for session in self.active_sessions.values()]
            }
            
            # Get user permissions
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("""
                    SELECT role_name, permissions, expires_at 
                    FROM security_access_control 
                    WHERE user_id = :user_id AND is_active = true
                    ORDER BY created_at DESC
                """), {'user_id': user_id})
                
                access_control = result.fetchone()
                if access_control:
                    status['role'] = access_control.role_name
                    status['permissions'] = json.loads(access_control.permissions)
                    status['access_expires'] = access_control.expires_at.isoformat() if access_control.expires_at else None
            
            return status
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get user security status: {e}")
            return {'user_id': user_id, 'error': str(e)}
    
    async def unlock_user(self, user_id: str, reason: str = None) -> bool:
        """Manually unlock a user"""
        try:
            if user_id in self.locked_users:
                del self.locked_users[user_id]
            
            if user_id in self.failed_attempts:
                del self.failed_attempts[user_id]
            
            # Log the unlock event
            await self.log_security_event(
                event_type='user_unlocked',
                severity='low',
                source='manual_action',
                user_id=user_id,
                event_details={'reason': reason or 'manual_unlock'},
                threat_level='low'
            )
            
            self.logger.info(f"✅ User {user_id} unlocked manually")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to unlock user: {e}")
            return False
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get overall security system status"""
        try:
            status = {
                'system_status': 'active',
                'audit_logging_enabled': self.audit_logging_enabled,
                'access_control_enabled': self.access_control_enabled,
                'secrets_management_enabled': self.secrets_management_enabled,
                'security_monitoring_enabled': self.security_monitoring_enabled,
                'locked_users_count': len(self.locked_users),
                'active_sessions_count': len(self.active_sessions),
                'failed_attempts_count': len(self.failed_attempts),
                'last_security_check': datetime.now(timezone.utc).isoformat()
            }
            
            # Get recent security events count
            try:
                async with self.async_engine.begin() as conn:
                    result = await conn.execute(text("""
                        SELECT COUNT(*) as event_count,
                               COUNT(CASE WHEN severity = 'high' THEN 1 END) as high_severity_count,
                               COUNT(CASE WHEN created_at > NOW() - INTERVAL '1 hour' THEN 1 END) as recent_events
                        FROM security_audit_log
                        WHERE created_at > NOW() - INTERVAL '24 hours'
                    """))
                    
                    event_stats = result.fetchone()
                    if event_stats:
                        status['events_24h'] = event_stats.event_count
                        status['high_severity_events_24h'] = event_stats.high_severity_count
                        status['events_last_hour'] = event_stats.recent_events
            except Exception as e:
                self.logger.warning(f"Could not get security event stats: {e}")
                status['events_24h'] = 0
                status['high_severity_events_24h'] = 0
                status['events_last_hour'] = 0
            
            return status
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get security status: {e}")
            return {
                'system_status': 'error',
                'error': str(e),
                'last_security_check': datetime.now(timezone.utc).isoformat()
            }