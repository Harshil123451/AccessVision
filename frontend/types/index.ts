export interface DetectionItem {
  box: number[];
  label: string;
  confidence: number;
}

export interface ReasoningResult {
  success: boolean;
  question: string;
  intent: string;
  target_object: string | null;
  answer: string;
  grounded_by: string;
  detections: DetectionItem[];
  metrics: {
    inference_ms: number;
  };
}

export interface QAPair {
  question: string;
  answer: string;
  timestamp: number;
}

export interface AccessibilitySettings {
  highContrast: boolean;
  largeText: boolean;
  speechRate: number; // multiplier e.g. 1.0, 1.2, 1.5, 0.8
  autoplaySpeech: boolean;
}

export interface StoreState {
  // Camera State
  hasCameraPermission: boolean | null;
  cameraActive: boolean;
  capturedImage: string | null; // base64 representation of captured frame
  
  // API & Processing State
  loading: boolean;
  error: string | null;
  narration: string | null;
  
  // Q&A State
  qaHistory: QAPair[];
  
  // Accessibility & Speech State
  accessibility: AccessibilitySettings;
  isSpeaking: boolean;
  isListening: boolean;
  
  // Actions
  setCameraPermission: (permission: boolean | null) => void;
  setCameraActive: (active: boolean) => void;
  setCapturedImage: (image: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setNarration: (narration: string | null) => void;
  addQAPair: (qa: QAPair) => void;
  clearQAPair: () => void;
  updateAccessibility: (settings: Partial<AccessibilitySettings>) => void;
  setIsSpeaking: (speaking: boolean) => void;
  setIsListening: (listening: boolean) => void;
}
