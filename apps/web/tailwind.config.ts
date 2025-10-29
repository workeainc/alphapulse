import type { Config } from 'tailwindcss';

const config: Config = {
  darkMode: ['class'],
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Base Colors (Dark Mode)
        background: {
          primary: '#0B0E11',
          secondary: '#141820',
          tertiary: '#1E2329',
        },
        // Signal Colors
        signal: {
          long: '#10B981',
          short: '#EF4444',
          neutral: '#6B7280',
        },
        // Confidence Levels
        confidence: {
          'very-high': '#34D399',
          high: '#10B981',
          medium: '#FBBF24',
          low: '#F87171',
        },
        // Status Indicators
        status: {
          active: '#3B82F6',
          warning: '#F59E0B',
          error: '#DC2626',
          success: '#059669',
        },
        // SDE Heads
        heads: {
          technical: '#8B5CF6',
          sentiment: '#EC4899',
          volume: '#14B8A6',
          rules: '#F59E0B',
          ict: '#06B6D4',
          wyckoff: '#10B981',
          harmonic: '#EAB308',
          structure: '#3B82F6',
          crypto: '#A855F7',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Roboto Mono', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-slow': 'bounce 2s infinite',
      },
    },
  },
  plugins: [],
};

export default config;

