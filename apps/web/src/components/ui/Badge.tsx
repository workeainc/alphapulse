import * as React from 'react';
import { cn } from '@/lib/utils/cn';

interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info';
}

export const Badge = React.forwardRef<HTMLDivElement, BadgeProps>(
  ({ className, variant = 'default', ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold',
          {
            'bg-gray-700 text-gray-200': variant === 'default',
            'bg-green-500/20 text-green-400': variant === 'success',
            'bg-yellow-500/20 text-yellow-400': variant === 'warning',
            'bg-red-500/20 text-red-400': variant === 'danger',
            'bg-blue-500/20 text-blue-400': variant === 'info',
          },
          className
        )}
        {...props}
      />
    );
  }
);

Badge.displayName = 'Badge';

