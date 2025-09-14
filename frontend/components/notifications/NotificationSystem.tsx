/**
 * Sophisticated Notification System
 * Real-time notifications for trading signals and system events
 * Phase 5: Advanced Features & Performance Optimization
 */

import React, { useState, useEffect, useCallback } from 'react';
import { createContext, useContext, ReactNode } from 'react';
import { toast, Toast } from 'react-hot-toast';
import { 
  Bell, 
  X, 
  CheckCircle, 
  AlertTriangle, 
  Info, 
  Zap,
  Target,
  TrendingUp,
  TrendingDown,
  Activity,
  Shield,
  Settings,
  Volume2,
  VolumeX
} from 'lucide-react';

// Notification Types
export type NotificationType = 'signal' | 'confidence' | 'execution' | 'error' | 'info' | 'warning';

export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  timestamp: Date;
  pair?: string;
  confidence?: number;
  signalDirection?: 'long' | 'short';
  isRead?: boolean;
  isPersistent?: boolean;
  actions?: NotificationAction[];
}

export interface NotificationAction {
  label: string;
  action: () => void;
  variant?: 'primary' | 'secondary' | 'danger';
}

// Notification Context
interface NotificationContextType {
  notifications: Notification[];
  unreadCount: number;
  isEnabled: boolean;
  soundEnabled: boolean;
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void;
  markAsRead: (id: string) => void;
  markAllAsRead: () => void;
  removeNotification: (id: string) => void;
  clearAll: () => void;
  toggleEnabled: () => void;
  toggleSound: () => void;
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

// Notification Provider
export const NotificationProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [isEnabled, setIsEnabled] = useState(true);
  const [soundEnabled, setSoundEnabled] = useState(true);

  const unreadCount = notifications.filter(n => !n.isRead).length;

  const addNotification = useCallback((notification: Omit<Notification, 'id' | 'timestamp'>) => {
    if (!isEnabled) return;

    const newNotification: Notification = {
      ...notification,
      id: `notification_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      isRead: false,
    };

    setNotifications(prev => [newNotification, ...prev.slice(0, 49)]); // Keep only last 50

    // Show toast notification
    const toastOptions = {
      duration: notification.isPersistent ? Infinity : 5000,
      position: 'top-right' as const,
      style: {
        background: getNotificationColor(notification.type),
        color: '#ffffff',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: '8px',
        padding: '16px',
        maxWidth: '400px',
      },
    };

    toast.custom((t: Toast) => (
      <NotificationToast
        notification={newNotification}
        toast={t}
        onAction={(action) => {
          action.action();
          toast.dismiss(t.id);
        }}
      />
    ), toastOptions);

    // Play notification sound
    if (soundEnabled) {
      playNotificationSound(notification.type);
    }
  }, [isEnabled, soundEnabled]);

  const markAsRead = useCallback((id: string) => {
    setNotifications(prev => 
      prev.map(n => n.id === id ? { ...n, isRead: true } : n)
    );
  }, []);

  const markAllAsRead = useCallback(() => {
    setNotifications(prev => 
      prev.map(n => ({ ...n, isRead: true }))
    );
  }, []);

  const removeNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);

  const clearAll = useCallback(() => {
    setNotifications([]);
  }, []);

  const toggleEnabled = useCallback(() => {
    setIsEnabled(prev => !prev);
  }, []);

  const toggleSound = useCallback(() => {
    setSoundEnabled(prev => !prev);
  }, []);

  return (
    <NotificationContext.Provider value={{
      notifications,
      unreadCount,
      isEnabled,
      soundEnabled,
      addNotification,
      markAsRead,
      markAllAsRead,
      removeNotification,
      clearAll,
      toggleEnabled,
      toggleSound,
    }}>
      {children}
    </NotificationContext.Provider>
  );
};

// Hook to use notifications
export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider');
  }
  return context;
};

// Notification Toast Component
const NotificationToast: React.FC<{
  notification: Notification;
  toast: Toast;
  onAction: (action: NotificationAction) => void;
}> = ({ notification, toast, onAction }) => {
  const { markAsRead, removeNotification } = useNotifications();

  useEffect(() => {
    markAsRead(notification.id);
  }, [notification.id, markAsRead]);

  const getIcon = () => {
    switch (notification.type) {
      case 'signal':
        return notification.signalDirection === 'long' ? 
          <TrendingUp className="h-5 w-5 text-green-400" /> : 
          <TrendingDown className="h-5 w-5 text-red-400" />;
      case 'confidence':
        return <Target className="h-5 w-5 text-blue-400" />;
      case 'execution':
        return <CheckCircle className="h-5 w-5 text-green-400" />;
      case 'error':
        return <AlertTriangle className="h-5 w-5 text-red-400" />;
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-yellow-400" />;
      default:
        return <Info className="h-5 w-5 text-blue-400" />;
    }
  };

  return (
    <div className="flex items-start space-x-3 p-4 bg-gray-900 border border-gray-700 rounded-lg shadow-lg">
      <div className="flex-shrink-0">
        {getIcon()}
      </div>
      
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between">
          <h4 className="text-white font-semibold text-sm">
            {notification.title}
          </h4>
          <button
            onClick={() => toast.dismiss(toast.id)}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        
        <p className="text-gray-300 text-sm mt-1">
          {notification.message}
        </p>
        
        {notification.pair && (
          <div className="flex items-center space-x-2 mt-2">
            <span className="text-xs text-gray-400">Pair:</span>
            <span className="text-xs text-blue-400 font-medium">{notification.pair}</span>
            {notification.confidence && (
              <>
                <span className="text-xs text-gray-400">•</span>
                <span className="text-xs text-green-400 font-medium">
                  {(notification.confidence * 100).toFixed(1)}%
                </span>
              </>
            )}
          </div>
        )}
        
        {notification.actions && notification.actions.length > 0 && (
          <div className="flex space-x-2 mt-3">
            {notification.actions.map((action, index) => (
              <button
                key={index}
                onClick={() => onAction(action)}
                className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
                  action.variant === 'primary' ? 'bg-blue-600 hover:bg-blue-700 text-white' :
                  action.variant === 'danger' ? 'bg-red-600 hover:bg-red-700 text-white' :
                  'bg-gray-700 hover:bg-gray-600 text-gray-300'
                }`}
              >
                {action.label}
              </button>
            ))}
          </div>
        )}
        
        <div className="text-xs text-gray-500 mt-2">
          {notification.timestamp.toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
};

// Notification Bell Component
export const NotificationBell: React.FC<{
  className?: string;
}> = ({ className = '' }) => {
  const { unreadCount, isEnabled, soundEnabled, toggleEnabled, toggleSound } = useNotifications();
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className={`relative ${className}`}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="relative p-2 text-gray-400 hover:text-white transition-colors"
      >
        <Bell className="h-5 w-5" />
        {unreadCount > 0 && (
          <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
            {unreadCount > 99 ? '99+' : unreadCount}
          </span>
        )}
      </button>
      
      {isOpen && (
        <NotificationPanel onClose={() => setIsOpen(false)} />
      )}
    </div>
  );
};

// Notification Panel Component
const NotificationPanel: React.FC<{
  onClose: () => void;
}> = ({ onClose }) => {
  const { 
    notifications, 
    unreadCount, 
    isEnabled, 
    soundEnabled,
    markAsRead, 
    markAllAsRead, 
    removeNotification, 
    clearAll,
    toggleEnabled,
    toggleSound
  } = useNotifications();

  return (
    <div className="absolute right-0 top-12 w-96 bg-gray-900 border border-gray-700 rounded-lg shadow-xl z-50">
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <h3 className="text-white font-semibold">Notifications</h3>
          <div className="flex items-center space-x-2">
            <button
              onClick={toggleSound}
              className={`p-1 rounded transition-colors ${
                soundEnabled ? 'text-green-400 hover:text-green-300' : 'text-gray-400 hover:text-gray-300'
              }`}
              title={soundEnabled ? 'Disable sound' : 'Enable sound'}
            >
              {soundEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
            </button>
            <button
              onClick={toggleEnabled}
              className={`p-1 rounded transition-colors ${
                isEnabled ? 'text-green-400 hover:text-green-300' : 'text-gray-400 hover:text-gray-300'
              }`}
              title={isEnabled ? 'Disable notifications' : 'Enable notifications'}
            >
              <Bell className="h-4 w-4" />
            </button>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
        
        {unreadCount > 0 && (
          <div className="flex items-center justify-between mt-2">
            <span className="text-sm text-gray-400">
              {unreadCount} unread notification{unreadCount !== 1 ? 's' : ''}
            </span>
            <button
              onClick={markAllAsRead}
              className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
            >
              Mark all as read
            </button>
          </div>
        )}
      </div>
      
      <div className="max-h-96 overflow-y-auto">
        {notifications.length === 0 ? (
          <div className="p-8 text-center text-gray-400">
            <Bell className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p>No notifications yet</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-700">
            {notifications.map((notification) => (
              <NotificationItem
                key={notification.id}
                notification={notification}
                onMarkAsRead={() => markAsRead(notification.id)}
                onRemove={() => removeNotification(notification.id)}
              />
            ))}
          </div>
        )}
      </div>
      
      {notifications.length > 0 && (
        <div className="p-4 border-t border-gray-700">
          <button
            onClick={clearAll}
            className="w-full text-sm text-red-400 hover:text-red-300 transition-colors"
          >
            Clear all notifications
          </button>
        </div>
      )}
    </div>
  );
};

// Notification Item Component
const NotificationItem: React.FC<{
  notification: Notification;
  onMarkAsRead: () => void;
  onRemove: () => void;
}> = ({ notification, onMarkAsRead, onRemove }) => {
  const getIcon = () => {
    switch (notification.type) {
      case 'signal':
        return notification.signalDirection === 'long' ? 
          <TrendingUp className="h-4 w-4 text-green-400" /> : 
          <TrendingDown className="h-4 w-4 text-red-400" />;
      case 'confidence':
        return <Target className="h-4 w-4 text-blue-400" />;
      case 'execution':
        return <CheckCircle className="h-4 w-4 text-green-400" />;
      case 'error':
        return <AlertTriangle className="h-4 w-4 text-red-400" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-yellow-400" />;
      default:
        return <Info className="h-4 w-4 text-blue-400" />;
    }
  };

  return (
    <div className={`p-4 hover:bg-gray-800/50 transition-colors ${
      !notification.isRead ? 'bg-blue-500/5 border-l-2 border-l-blue-500' : ''
    }`}>
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0 mt-0.5">
          {getIcon()}
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <h4 className="text-white font-medium text-sm">
              {notification.title}
            </h4>
            <div className="flex items-center space-x-1">
              {!notification.isRead && (
                <button
                  onClick={onMarkAsRead}
                  className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                >
                  Mark read
                </button>
              )}
              <button
                onClick={onRemove}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          </div>
          
          <p className="text-gray-300 text-sm mt-1">
            {notification.message}
          </p>
          
          {notification.pair && (
            <div className="flex items-center space-x-2 mt-2">
              <span className="text-xs text-gray-400">Pair:</span>
              <span className="text-xs text-blue-400 font-medium">{notification.pair}</span>
              {notification.confidence && (
                <>
                  <span className="text-xs text-gray-400">•</span>
                  <span className="text-xs text-green-400 font-medium">
                    {(notification.confidence * 100).toFixed(1)}%
                  </span>
                </>
              )}
            </div>
          )}
          
          <div className="text-xs text-gray-500 mt-2">
            {notification.timestamp.toLocaleString()}
          </div>
        </div>
      </div>
    </div>
  );
};

// Helper functions
function getNotificationColor(type: NotificationType): string {
  switch (type) {
    case 'signal':
      return '#1f2937';
    case 'confidence':
      return '#1e3a8a';
    case 'execution':
      return '#166534';
    case 'error':
      return '#991b1b';
    case 'warning':
      return '#92400e';
    default:
      return '#374151';
  }
}

function playNotificationSound(type: NotificationType): void {
  // This would play different sounds based on notification type
  // For now, we'll use a simple beep
  try {
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.setValueAtTime(
      type === 'signal' ? 800 : type === 'confidence' ? 600 : 400,
      audioContext.currentTime
    );
    
    gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.5);
  } catch (error) {
    console.warn('Could not play notification sound:', error);
  }
}

export default NotificationProvider;
