import { create } from "zustand";
import { StoreState } from "../types";

export const useAppStore = create<StoreState>((set) => ({
  // Default States
  hasCameraPermission: null,
  cameraActive: false,
  capturedImage: null,
  facingMode: "environment",
  loading: false,
  error: null,
  narration: null,
  qaHistory: [],
  accessibility: {
    highContrast: false,
    largeText: false,
    speechRate: 1.0,
    autoplaySpeech: true,
  },
  isSpeaking: false,
  isListening: false,

  // Action Implementations
  setCameraPermission: (permission) => set({ hasCameraPermission: permission }),
  setCameraActive: (active) => set({ cameraActive: active }),
  setCapturedImage: (image) => set({ capturedImage: image }),
  setFacingMode: (facingMode) => set({ facingMode }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setNarration: (narration) => set({ narration }),
  addQAPair: (qa) => set((state) => ({ qaHistory: [...state.qaHistory, qa] })),
  clearQAPair: () => set({ qaHistory: [] }),
  updateAccessibility: (settings) =>
    set((state) => ({
      accessibility: { ...state.accessibility, ...settings },
    })),
  setIsSpeaking: (speaking) => set({ isSpeaking: speaking }),
  setIsListening: (listening) => set({ isListening: listening }),
}));
