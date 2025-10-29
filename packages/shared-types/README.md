# @alphapulse/shared-types

Shared TypeScript type definitions for AlphaPulse monorepo.

## Purpose

Provides type-safe interfaces for communication between backend and frontend, ensuring data consistency across the application.

## Usage

```typescript
import {
  Signal,
  SignalRecommendation,
  UserSettings,
  Alert,
  AlertHistory
} from '@alphapulse/shared-types';
```

## Types Included

### Signal Types
- `Signal` - Trading signal with full metadata
- `SignalDirection` - 'long' | 'short'
- `SignalSource` - 'pattern' | 'ml_ensemble' | 'hybrid' | 'manual'
- `SignalFilters` - Query filters for signals
- `SignalResponse` - Paginated signal response

### Recommendation Types
- `SignalRecommendation` - Trading recommendation for users
- `RecommendationStatus` - 'pending' | 'user_executed' | 'expired' | 'cancelled'
- `RecommendationSide` - 'long' | 'short'
- `RecommendationFilters` - Query filters
- `RecommendationResponse` - Paginated response

### User Settings Types
- `UserSettings` - User preferences and notification config
- `RiskTolerance` - 'low' | 'medium' | 'high'
- `AlertFrequency` - 'immediate' | 'hourly' | 'daily'
- `NotificationPreferences` - Email, Telegram, Discord, Webhook

### Alert Types
- `Alert` - Alert/notification structure
- `AlertHistory` - Alert history with delivery status
- `AlertType` - 'new_signal' | 'price_target' | 'stop_loss' | 'system' | 'critical'
- `AlertPriority` - 'low' | 'medium' | 'high' | 'critical'
- `DeliveryMethod` - 'email' | 'telegram' | 'discord' | 'webhook' | 'in_app'

## Benefits

✅ Single source of truth for data structures  
✅ Type safety across frontend/backend  
✅ IntelliSense support in IDE  
✅ Prevents API contract mismatches  
✅ Self-documenting code  

## Version

1.0.0

