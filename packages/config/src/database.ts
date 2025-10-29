/**
 * Database Configuration
 * Shared database configuration constants
 */

export const DATABASE_CONFIG = {
  defaultPort: 55433,
  defaultHost: 'localhost',
  defaultDatabase: 'alphapulse',
  defaultUser: 'alpha_emon',
  poolSize: 20,
  maxOverflow: 10,
  poolTimeout: 30,
} as const;

export function buildDatabaseUrl(
  user: string = DATABASE_CONFIG.defaultUser,
  password: string,
  host: string = DATABASE_CONFIG.defaultHost,
  port: number = DATABASE_CONFIG.defaultPort,
  database: string = DATABASE_CONFIG.defaultDatabase
): string {
  // URL-encode the password to handle special characters
  const encodedPassword = encodeURIComponent(password);
  return `postgresql://${user}:${encodedPassword}@${host}:${port}/${database}`;
}

