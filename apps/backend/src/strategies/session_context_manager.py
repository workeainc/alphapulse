"""
Session Context Manager for AlphaPulse
Manages trading sessions, kill zones, and time-based filtering for ICT concepts
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, time, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
import pytz

logger = logging.getLogger(__name__)

class TradingSession(Enum):
    """Trading session enumeration"""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    LONDON_NY_OVERLAP = "london_ny_overlap"
    UNKNOWN = "unknown"

class KillZoneType(Enum):
    """Kill zone types (high probability trading windows)"""
    LONDON_KILLZONE = "london_killzone"
    NEW_YORK_KILLZONE = "new_york_killzone"
    ASIAN_RANGE = "asian_range"
    SILVER_BULLET_AM = "silver_bullet_am"
    SILVER_BULLET_PM = "silver_bullet_pm"
    GOLD_BULLET = "gold_bullet"
    NONE = "none"

@dataclass
class SessionInfo:
    """Trading session information"""
    session: TradingSession
    session_start: datetime
    session_end: datetime
    kill_zone: KillZoneType
    is_high_probability: bool
    probability_multiplier: float
    metadata: Dict[str, Any]

@dataclass
class SessionContext:
    """Complete session context for a given timestamp"""
    timestamp: datetime
    active_session: TradingSession
    active_kill_zone: KillZoneType
    is_high_probability_time: bool
    probability_multiplier: float
    session_info: SessionInfo
    hours_into_session: float
    minutes_to_session_end: float

class SessionContextManager:
    """
    Manages trading sessions and time-based contexts for ICT concepts
    
    This manager handles:
    - Session identification (Asian, London, New York)
    - Kill zone detection (high probability trading windows)
    - Macro time detection (Silver/Gold bullets)
    - Signal weighting based on time
    """
    
    def __init__(self, reference_timezone: str = "US/Eastern"):
        """
        Initialize Session Context Manager
        
        Args:
            reference_timezone: Reference timezone for ICT times (default: US/Eastern)
        """
        self.reference_tz = pytz.timezone(reference_timezone)
        
        # Session definitions (in EST)
        self.session_times = {
            TradingSession.ASIAN: {
                'start': time(20, 0),  # 8:00 PM EST
                'end': time(0, 0),     # 12:00 AM EST (midnight)
                'probability': 0.6,
                'description': 'Asian session - lower volatility'
            },
            TradingSession.LONDON: {
                'start': time(2, 0),   # 2:00 AM EST
                'end': time(12, 0),    # 12:00 PM EST
                'probability': 0.9,
                'description': 'London session - high volatility'
            },
            TradingSession.NEW_YORK: {
                'start': time(8, 0),   # 8:00 AM EST
                'end': time(17, 0),    # 5:00 PM EST
                'probability': 0.95,
                'description': 'New York session - highest volatility'
            },
            TradingSession.LONDON_NY_OVERLAP: {
                'start': time(8, 0),   # 8:00 AM EST
                'end': time(12, 0),    # 12:00 PM EST
                'probability': 1.0,
                'description': 'London/NY overlap - highest probability'
            }
        }
        
        # Kill Zone definitions (ICT high-probability windows)
        self.kill_zones = {
            KillZoneType.LONDON_KILLZONE: {
                'start': time(2, 0),   # 2:00 AM EST
                'end': time(5, 0),     # 5:00 AM EST
                'probability': 0.9,
                'description': 'London Kill Zone - high probability moves'
            },
            KillZoneType.NEW_YORK_KILLZONE: {
                'start': time(8, 0),   # 8:00 AM EST
                'end': time(11, 0),    # 11:00 AM EST
                'probability': 0.95,
                'description': 'New York Kill Zone - highest probability'
            },
            KillZoneType.ASIAN_RANGE: {
                'start': time(20, 0),  # 8:00 PM EST
                'end': time(0, 0),     # 12:00 AM EST
                'probability': 0.5,
                'description': 'Asian range - liquidity building'
            },
            KillZoneType.SILVER_BULLET_AM: {
                'start': time(9, 50),  # 9:50 AM EST
                'end': time(10, 10),   # 10:10 AM EST
                'probability': 1.0,
                'description': 'Silver Bullet AM - macro time precision'
            },
            KillZoneType.SILVER_BULLET_PM: {
                'start': time(15, 0),  # 3:00 PM EST
                'end': time(15, 20),   # 3:20 PM EST
                'probability': 1.0,
                'description': 'Silver Bullet PM - macro time precision'
            },
            KillZoneType.GOLD_BULLET: {
                'start': time(10, 0),  # 10:00 AM EST
                'end': time(11, 0),    # 11:00 AM EST
                'probability': 0.95,
                'description': 'Gold Bullet - extended high-probability window'
            }
        }
        
        # Performance tracking
        self.stats = {
            'contexts_generated': 0,
            'kill_zones_detected': 0,
            'high_prob_signals_weighted': 0,
            'low_prob_signals_filtered': 0
        }
        
        logger.info(f"🕐 Session Context Manager initialized (timezone: {reference_timezone})")
    
    def get_session_context(self, timestamp: Optional[datetime] = None) -> SessionContext:
        """
        Get complete session context for a given timestamp
        
        Args:
            timestamp: Timestamp to analyze (default: now)
            
        Returns:
            SessionContext with all session information
        """
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)
            
            # Convert to reference timezone
            if timestamp.tzinfo is None:
                timestamp = pytz.utc.localize(timestamp)
            
            reference_time = timestamp.astimezone(self.reference_tz)
            current_time = reference_time.time()
            
            # Identify active session
            active_session = self._identify_session(current_time)
            
            # Identify active kill zone
            active_kill_zone = self._identify_kill_zone(current_time)
            
            # Calculate probability multiplier
            is_high_prob = active_kill_zone != KillZoneType.NONE
            probability_multiplier = self._calculate_probability_multiplier(
                active_session, active_kill_zone
            )
            
            # Get session info
            session_info = self._get_session_info(active_session, reference_time)
            
            # Calculate time metrics
            hours_into_session = self._calculate_hours_into_session(
                current_time, session_info
            )
            minutes_to_end = self._calculate_minutes_to_session_end(
                current_time, session_info
            )
            
            # Update stats
            self.stats['contexts_generated'] += 1
            if is_high_prob:
                self.stats['kill_zones_detected'] += 1
            
            context = SessionContext(
                timestamp=reference_time,
                active_session=active_session,
                active_kill_zone=active_kill_zone,
                is_high_probability_time=is_high_prob,
                probability_multiplier=probability_multiplier,
                session_info=session_info,
                hours_into_session=hours_into_session,
                minutes_to_session_end=minutes_to_end
            )
            
            logger.debug(
                f"Session context: {active_session.value}, "
                f"Kill zone: {active_kill_zone.value}, "
                f"Multiplier: {probability_multiplier:.2f}"
            )
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Error getting session context: {e}")
            return self._get_default_context(timestamp)
    
    def _identify_session(self, current_time: time) -> TradingSession:
        """Identify active trading session"""
        try:
            # Check London/NY overlap first (highest priority)
            if self._is_time_in_range(
                current_time,
                self.session_times[TradingSession.LONDON_NY_OVERLAP]['start'],
                self.session_times[TradingSession.LONDON_NY_OVERLAP]['end']
            ):
                return TradingSession.LONDON_NY_OVERLAP
            
            # Check other sessions
            for session, times in self.session_times.items():
                if session == TradingSession.LONDON_NY_OVERLAP:
                    continue
                
                if self._is_time_in_range(current_time, times['start'], times['end']):
                    return session
            
            return TradingSession.UNKNOWN
            
        except Exception as e:
            logger.error(f"Error identifying session: {e}")
            return TradingSession.UNKNOWN
    
    def _identify_kill_zone(self, current_time: time) -> KillZoneType:
        """Identify active kill zone"""
        try:
            # Check macro times first (highest priority)
            if self._is_time_in_range(
                current_time,
                self.kill_zones[KillZoneType.SILVER_BULLET_AM]['start'],
                self.kill_zones[KillZoneType.SILVER_BULLET_AM]['end']
            ):
                return KillZoneType.SILVER_BULLET_AM
            
            if self._is_time_in_range(
                current_time,
                self.kill_zones[KillZoneType.SILVER_BULLET_PM]['start'],
                self.kill_zones[KillZoneType.SILVER_BULLET_PM]['end']
            ):
                return KillZoneType.SILVER_BULLET_PM
            
            # Check other kill zones
            for kz, times in self.kill_zones.items():
                if kz in [KillZoneType.SILVER_BULLET_AM, KillZoneType.SILVER_BULLET_PM]:
                    continue
                
                if self._is_time_in_range(current_time, times['start'], times['end']):
                    return kz
            
            return KillZoneType.NONE
            
        except Exception as e:
            logger.error(f"Error identifying kill zone: {e}")
            return KillZoneType.NONE
    
    def _is_time_in_range(self, current: time, start: time, end: time) -> bool:
        """Check if time is within range (handles midnight crossing)"""
        try:
            if start < end:
                return start <= current < end
            else:  # Crosses midnight
                return current >= start or current < end
        except Exception:
            return False
    
    def _calculate_probability_multiplier(
        self, 
        session: TradingSession, 
        kill_zone: KillZoneType
    ) -> float:
        """Calculate probability multiplier based on session and kill zone"""
        try:
            multiplier = 1.0
            
            # Session multiplier
            if session in self.session_times:
                session_prob = self.session_times[session]['probability']
                multiplier *= session_prob
            
            # Kill zone multiplier (additive bonus)
            if kill_zone != KillZoneType.NONE and kill_zone in self.kill_zones:
                kz_prob = self.kill_zones[kill_zone]['probability']
                multiplier *= kz_prob
            
            return min(1.5, multiplier)  # Cap at 1.5x
            
        except Exception as e:
            logger.error(f"Error calculating probability multiplier: {e}")
            return 1.0
    
    def _get_session_info(
        self, 
        session: TradingSession, 
        reference_time: datetime
    ) -> SessionInfo:
        """Get detailed session information"""
        try:
            if session not in self.session_times:
                return SessionInfo(
                    session=session,
                    session_start=reference_time,
                    session_end=reference_time,
                    kill_zone=KillZoneType.NONE,
                    is_high_probability=False,
                    probability_multiplier=1.0,
                    metadata={}
                )
            
            session_data = self.session_times[session]
            
            # Calculate session start/end as datetime
            session_start = reference_time.replace(
                hour=session_data['start'].hour,
                minute=session_data['start'].minute,
                second=0,
                microsecond=0
            )
            
            session_end = reference_time.replace(
                hour=session_data['end'].hour,
                minute=session_data['end'].minute,
                second=0,
                microsecond=0
            )
            
            # Handle midnight crossing
            if session_data['start'] > session_data['end']:
                if reference_time.time() < session_data['end']:
                    session_start -= timedelta(days=1)
                else:
                    session_end += timedelta(days=1)
            
            return SessionInfo(
                session=session,
                session_start=session_start,
                session_end=session_end,
                kill_zone=self._identify_kill_zone(reference_time.time()),
                is_high_probability=session_data['probability'] >= 0.8,
                probability_multiplier=session_data['probability'],
                metadata={
                    'description': session_data['description'],
                    'base_probability': session_data['probability']
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return SessionInfo(
                session=session,
                session_start=reference_time,
                session_end=reference_time,
                kill_zone=KillZoneType.NONE,
                is_high_probability=False,
                probability_multiplier=1.0,
                metadata={}
            )
    
    def _calculate_hours_into_session(
        self, 
        current_time: time, 
        session_info: SessionInfo
    ) -> float:
        """Calculate hours into current session"""
        try:
            session_start = session_info.session_start
            current_dt = session_info.session_start.replace(
                hour=current_time.hour,
                minute=current_time.minute
            )
            
            if current_dt < session_start:
                current_dt += timedelta(days=1)
            
            delta = current_dt - session_start
            return delta.total_seconds() / 3600
            
        except Exception:
            return 0.0
    
    def _calculate_minutes_to_session_end(
        self, 
        current_time: time, 
        session_info: SessionInfo
    ) -> float:
        """Calculate minutes to session end"""
        try:
            session_end = session_info.session_end
            current_dt = session_info.session_start.replace(
                hour=current_time.hour,
                minute=current_time.minute
            )
            
            if current_dt > session_end:
                return 0.0
            
            delta = session_end - current_dt
            return delta.total_seconds() / 60
            
        except Exception:
            return 0.0
    
    def apply_session_filter(
        self, 
        signal_confidence: float, 
        timestamp: Optional[datetime] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Apply session-based filtering to signal confidence
        
        Args:
            signal_confidence: Original signal confidence (0-1)
            timestamp: Timestamp for context (default: now)
            
        Returns:
            Tuple of (adjusted_confidence, filter_metadata)
        """
        try:
            context = self.get_session_context(timestamp)
            
            # Apply probability multiplier
            adjusted_confidence = signal_confidence * context.probability_multiplier
            adjusted_confidence = min(1.0, adjusted_confidence)
            
            # Update stats
            if context.is_high_probability_time:
                self.stats['high_prob_signals_weighted'] += 1
            elif context.probability_multiplier < 0.8:
                self.stats['low_prob_signals_filtered'] += 1
            
            metadata = {
                'original_confidence': signal_confidence,
                'adjusted_confidence': adjusted_confidence,
                'probability_multiplier': context.probability_multiplier,
                'active_session': context.active_session.value,
                'active_kill_zone': context.active_kill_zone.value,
                'is_high_probability': context.is_high_probability_time,
                'filter_reason': self._get_filter_reason(context)
            }
            
            logger.debug(
                f"Session filter applied: {signal_confidence:.3f} → "
                f"{adjusted_confidence:.3f} ({context.active_session.value})"
            )
            
            return adjusted_confidence, metadata
            
        except Exception as e:
            logger.error(f"Error applying session filter: {e}")
            return signal_confidence, {'error': str(e)}
    
    def _get_filter_reason(self, context: SessionContext) -> str:
        """Get human-readable filter reason"""
        if context.active_kill_zone != KillZoneType.NONE:
            return f"Enhanced by {context.active_kill_zone.value}"
        elif context.is_high_probability_time:
            return f"High probability {context.active_session.value}"
        else:
            return f"Standard {context.active_session.value}"
    
    def _get_default_context(self, timestamp: Optional[datetime] = None) -> SessionContext:
        """Get default context when errors occur"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        return SessionContext(
            timestamp=timestamp,
            active_session=TradingSession.UNKNOWN,
            active_kill_zone=KillZoneType.NONE,
            is_high_probability_time=False,
            probability_multiplier=1.0,
            session_info=SessionInfo(
                session=TradingSession.UNKNOWN,
                session_start=timestamp,
                session_end=timestamp,
                kill_zone=KillZoneType.NONE,
                is_high_probability=False,
                probability_multiplier=1.0,
                metadata={}
            ),
            hours_into_session=0.0,
            minutes_to_session_end=0.0
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'stats': self.stats,
            'session_definitions': {
                session.value: {
                    'start': times['start'].strftime('%H:%M'),
                    'end': times['end'].strftime('%H:%M'),
                    'probability': times['probability'],
                    'description': times['description']
                }
                for session, times in self.session_times.items()
            },
            'kill_zone_definitions': {
                kz.value: {
                    'start': times['start'].strftime('%H:%M'),
                    'end': times['end'].strftime('%H:%M'),
                    'probability': times['probability'],
                    'description': times['description']
                }
                for kz, times in self.kill_zones.items()
            }
        }

