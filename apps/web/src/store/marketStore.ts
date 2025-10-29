import { create } from 'zustand';
import type { MarketStatus } from '@/types';

interface MarketStore {
  status: MarketStatus | null;
  selectedSymbol: string;
  selectedTimeframe: string;
  watchlist: string[];
  setStatus: (status: MarketStatus) => void;
  setSelectedSymbol: (symbol: string) => void;
  setSelectedTimeframe: (timeframe: string) => void;
  addToWatchlist: (symbol: string) => void;
  removeFromWatchlist: (symbol: string) => void;
}

export const useMarketStore = create<MarketStore>((set) => ({
  status: null,
  selectedSymbol: 'BTCUSDT',
  selectedTimeframe: '1h',
  watchlist: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],

  setStatus: (status) => set({ status }),

  setSelectedSymbol: (selectedSymbol) => set({ selectedSymbol }),

  setSelectedTimeframe: (selectedTimeframe) => set({ selectedTimeframe }),

  addToWatchlist: (symbol) =>
    set((state) => ({
      watchlist: state.watchlist.includes(symbol)
        ? state.watchlist
        : [...state.watchlist, symbol],
    })),

  removeFromWatchlist: (symbol) =>
    set((state) => ({
      watchlist: state.watchlist.filter((s) => s !== symbol),
    })),
}));

