import { useAppStore } from "../store/app-store";

export class SpeechService {
  private static synth: SpeechSynthesis | null =
    typeof window !== "undefined" ? window.speechSynthesis : null;
  private static currentUtterance: SpeechSynthesisUtterance | null = null;

  public static speak(text: string, onEndCallback?: () => void) {
    if (!this.synth) {
      console.warn("Speech Synthesis is not supported in this browser.");
      if (onEndCallback) onEndCallback();
      return;
    }

    // Cancel current speech to prevent queuing overlap
    this.cancel();

    // Fetch user rate from accessibility state
    const rate = useAppStore.getState().accessibility.speechRate;
    
    const utterance = new SpeechSynthesisUtterance(text);
    this.currentUtterance = utterance;
    utterance.rate = rate;

    utterance.onstart = () => {
      useAppStore.getState().setIsSpeaking(true);
    };

    utterance.onend = () => {
      useAppStore.getState().setIsSpeaking(false);
      this.currentUtterance = null;
      if (onEndCallback) onEndCallback();
    };

    utterance.onerror = (event) => {
      console.error("Speech Synthesis error:", event);
      useAppStore.getState().setIsSpeaking(false);
      this.currentUtterance = null;
      if (onEndCallback) onEndCallback();
    };

    this.synth.speak(utterance);
  }

  public static pause() {
    if (this.synth && this.synth.speaking && !this.synth.paused) {
      this.synth.pause();
    }
  }

  public static resume() {
    if (this.synth && this.synth.paused) {
      this.synth.resume();
    }
  }

  public static cancel() {
    if (this.synth) {
      this.synth.cancel();
      useAppStore.getState().setIsSpeaking(false);
      this.currentUtterance = null;
    }
  }

  public static isSpeaking(): boolean {
    return this.synth ? this.synth.speaking : false;
  }
}

export class SpeechRecognitionService {
  private static recognition: any = null;

  public static init(
    onResult: (text: string) => void,
    onEnd: () => void,
    onError: (err: string) => void
  ): boolean {
    if (typeof window === "undefined") return false;

    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      console.warn("Speech Recognition (microphone input) is not supported in this browser.");
      return false;
    }

    try {
      const rec = new SpeechRecognition();
      rec.continuous = false;
      rec.interimResults = false;
      rec.lang = "en-US";

      rec.onstart = () => {
        useAppStore.getState().setIsListening(true);
      };

      rec.onresult = (event: any) => {
        const text = event.results[0][0].transcript;
        onResult(text);
      };

      rec.onerror = (event: any) => {
        console.error("Speech Recognition error:", event.error);
        onError(event.error);
        useAppStore.getState().setIsListening(false);
      };

      rec.onend = () => {
        useAppStore.getState().setIsListening(false);
        onEnd();
      };

      this.recognition = rec;
      return true;
    } catch (e) {
      console.error("Initialization of SpeechRecognition failed", e);
      return false;
    }
  }

  public static start() {
    if (this.recognition) {
      try {
        this.recognition.start();
      } catch (e) {
        console.error("Failed to start speech recognition", e);
      }
    }
  }

  public static stop() {
    if (this.recognition) {
      try {
        this.recognition.stop();
      } catch (e) {
        console.error("Failed to stop speech recognition", e);
      }
    }
  }
}
