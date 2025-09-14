import React from 'react';
import { CheckCircle, AlertTriangle, XCircle, Activity } from 'lucide-react';

interface StatusCardProps {
  title: string;
  value: string | number;
  status: 'success' | 'warning' | 'error' | 'info';
  subtitle?: string;
  icon?: React.ReactNode;
  className?: string;
}

const statusConfig = {
  success: {
    icon: CheckCircle,
    color: 'text-green-600',
    bgColor: 'bg-green-50',
    borderColor: 'border-green-200',
  },
  warning: {
    icon: AlertTriangle,
    color: 'text-yellow-600',
    bgColor: 'bg-yellow-50',
    borderColor: 'border-yellow-200',
  },
  error: {
    icon: XCircle,
    color: 'text-red-600',
    bgColor: 'bg-red-50',
    borderColor: 'border-red-200',
  },
  info: {
    icon: Activity,
    color: 'text-blue-600',
    bgColor: 'bg-blue-50',
    borderColor: 'border-blue-200',
  },
};

export const StatusCard: React.FC<StatusCardProps> = ({
  title,
  value,
  status,
  subtitle,
  icon,
  className = '',
}) => {
  const config = statusConfig[status];
  const IconComponent = config.icon;

  return (
    <div className={`bg-white rounded-lg border ${config.borderColor} p-4 ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-2">
            {icon || <IconComponent className={`w-5 h-5 ${config.color}`} />}
            <h3 className="text-sm font-medium text-gray-900">{title}</h3>
          </div>
          <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
          {subtitle && (
            <p className="text-sm text-gray-500 mt-1">{subtitle}</p>
          )}
        </div>
      </div>
    </div>
  );
};
