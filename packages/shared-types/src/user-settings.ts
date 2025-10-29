/**
 * User Settings Types
 * Types for user preferences and notification configuration
 */

export type RiskTolerance = 'low' | 'medium' | 'high';
export type AlertFrequency = 'immediate' | 'hourly' | 'daily';
export type DeliveryMethod = 'email' | 'telegram' | 'discord' | 'webhook' | 'in_app';

export interface NotificationPreferences {
  email?: {
    enabled: boolean;
    address: string;
  };
  telegram?: {
    enabled: boolean;
    chat_id: string;
    bot_token: string;
  };
  discord?: {
    enabled: boolean;
    webhook_url: string;
  };
  webhook?: {
    enabled: boolean;
    url: string;
    headers?: Record<string, string>;
  };
}

export interface UserSettings {
  id: number;
  user_id: string;
  email?: string;
  notification_preferences?: NotificationPreferences;
  preferred_symbols?: string[];
  min_confidence_threshold: number;
  preferred_timeframes?: string[];
  risk_tolerance: RiskTolerance;
  alert_high_confidence_only: boolean;
  alert_frequency: AlertFrequency;
  quiet_hours_start?: number;
  quiet_hours_end?: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

