import { create } from 'zustand';
import type { Signal } from '@/types';

interface SignalStore {
  signals: Signal[];
  selectedSignal: Signal | null;
  filterConfidence: number;
  filterSymbol: string | null;
  setSignals: (signals: Signal[]) => void;
  addSignal: (signal: Signal) => void;
  selectSignal: (signal: Signal | null) => void;
  setFilterConfidence: (confidence: number) => void;
  setFilterSymbol: (symbol: string | null) => void;
  getFilteredSignals: () => Signal[];
}

export const useSignalStore = create<SignalStore>((set, get) => ({
  signals: [],
  selectedSignal: null,
  filterConfidence: 0.6,
  filterSymbol: null,

  setSignals: (signals) => set({ signals }),

  addSignal: (signal) =>
    set((state) => ({
      signals: [signal, ...state.signals].slice(0, 100), // Keep last 100
    })),

  selectSignal: (signal) => set({ selectedSignal: signal }),

  setFilterConfidence: (filterConfidence) => set({ filterConfidence }),

  setFilterSymbol: (filterSymbol) => set({ filterSymbol }),

  getFilteredSignals: () => {
    const { signals, filterConfidence, filterSymbol } = get();
    return signals.filter((signal) => {
      const meetsConfidence = signal.confidence >= filterConfidence;
      const meetsSymbol = !filterSymbol || signal.symbol === filterSymbol;
      return meetsConfidence && meetsSymbol;
    });
  },
}));

