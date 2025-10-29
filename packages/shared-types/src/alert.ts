/**
 * Alert Types
 * Types for notification and alert system
 */

export type AlertType = 'new_signal' | 'price_target' | 'stop_loss' | 'system' | 'critical';
export type AlertPriority = 'low' | 'medium' | 'high' | 'critical';
export type DeliveryMethod = 'email' | 'telegram' | 'discord' | 'webhook' | 'in_app';

export interface Alert {
  alert_id: string;
  signal_id?: string;
  user_id?: string;
  alert_type: AlertType;
  delivery_method: DeliveryMethod;
  priority?: AlertPriority;
  sent_at: string;
  delivered: boolean;
  delivery_error?: string;
  read_at?: string;
  message?: string;
  metadata?: Record<string, any>;
  created_at: string;
}

export interface AlertHistory extends Alert {
  id: number;
}

