import { create } from 'zustand';

export interface Notification {
  id: string;
  type: 'signal' | 'tp' | 'sl' | 'system' | 'alert';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  data?: any;
}

interface NotificationStore {
  notifications: Notification[];
  soundEnabled: boolean;
  browserEnabled: boolean;
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => void;
  markAsRead: (id: string) => void;
  markAllAsRead: () => void;
  clearNotification: (id: string) => void;
  clearAll: () => void;
  toggleSound: () => void;
  toggleBrowser: () => void;
}

export const useNotificationStore = create<NotificationStore>((set) => ({
  notifications: [],
  soundEnabled: true,
  browserEnabled: true,

  addNotification: (notification) => {
    const newNotification: Notification = {
      ...notification,
      id: `${Date.now()}-${Math.random()}`,
      timestamp: new Date(),
      read: false,
    };

    set((state) => ({
      notifications: [newNotification, ...state.notifications].slice(0, 50), // Keep last 50
    }));

    // Play sound if enabled
    if (notification.type === 'signal') {
      // Sound logic here
    }

    // Show browser notification if enabled
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification(newNotification.title, {
        body: newNotification.message,
        icon: '/favicon.ico',
      });
    }
  },

  markAsRead: (id) =>
    set((state) => ({
      notifications: state.notifications.map((n) =>
        n.id === id ? { ...n, read: true } : n
      ),
    })),

  markAllAsRead: () =>
    set((state) => ({
      notifications: state.notifications.map((n) => ({ ...n, read: true })),
    })),

  clearNotification: (id) =>
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
    })),

  clearAll: () => set({ notifications: [] }),

  toggleSound: () => set((state) => ({ soundEnabled: !state.soundEnabled })),

  toggleBrowser: () => set((state) => ({ browserEnabled: !state.browserEnabled })),
}));

