"use client";

import React, { useEffect } from "react";
import { useAppStore } from "../../store/app-store";
import { SpeechService } from "../../services/speech";
import { Volume2, Play, Pause, Square, Sparkles, Loader } from "lucide-react";

export default function NarrationBox() {
  const {
    narration,
    loading,
    error,
    isSpeaking,
    setIsSpeaking,
    accessibility,
    updateAccessibility,
  } = useAppStore();

  const { largeText, highContrast, speechRate, autoplaySpeech } = accessibility;

  // Speak when new narration is generated (if autoplay enabled)
  useEffect(() => {
    if (narration && autoplaySpeech) {
      SpeechService.speak(narration);
    }
    return () => {
      SpeechService.cancel();
    };
  }, [narration, autoplaySpeech]);

  // Handle Play / Replay
  const handlePlay = () => {
    if (narration) {
      SpeechService.speak(narration);
    }
  };

  // Handle Pause
  const handlePause = () => {
    SpeechService.pause();
    setIsSpeaking(false);
  };

  // Handle Resume / Play if paused or stopped
  const handleResume = () => {
    if (window.speechSynthesis?.paused) {
      SpeechService.resume();
      setIsSpeaking(true);
    } else {
      handlePlay();
    }
  };

  // Handle Stop
  const handleStop = () => {
    SpeechService.cancel();
  };

  // Handle speed changes
  const changeSpeechRate = (e: React.ChangeEvent<HTMLInputElement>) => {
    const rate = parseFloat(e.target.value);
    updateAccessibility({ speechRate: rate });
    
    // Briefly say test word if speaking changed to let the user calibrate
    SpeechService.cancel();
    setTimeout(() => {
      if (narration) {
        SpeechService.speak(narration);
      }
    }, 100);
  };

  return (
    <div
      className={`p-6 rounded-3xl transition-all shadow-xl flex flex-col gap-6 ${
        highContrast
          ? "bg-black border-4 border-white text-white"
          : "bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 text-neutral-900 dark:text-neutral-50"
      }`}
    >
      {/* Header section */}
      <div className="flex items-center gap-3">
        <div className={`p-3 rounded-2xl ${highContrast ? "bg-white text-black" : "bg-emerald-100 dark:bg-emerald-950 text-emerald-600"}`}>
          {loading ? (
            <Loader className="w-6 h-6 animate-spin" />
          ) : (
            <Sparkles className="w-6 h-6" />
          )}
        </div>
        <div>
          <h2 className={`font-bold tracking-wide ${largeText ? "text-2xl" : "text-lg"}`}>
            Scene Narration
          </h2>
          <p className="text-xs text-neutral-400">AI-generated environmental description</p>
        </div>
      </div>

      {/* Narrative Output Display */}
      <div
        className={`min-h-[100px] flex items-center justify-center p-5 rounded-2xl ${
          highContrast
            ? "bg-neutral-950 border border-white"
            : "bg-neutral-50 dark:bg-neutral-950 border border-neutral-100 dark:border-neutral-900"
        }`}
        aria-live="polite"
        role="region"
      >
        {loading && (
          <div className="flex flex-col items-center gap-3 py-4">
            <Loader className="w-8 h-8 animate-spin text-emerald-500" />
            <p className="text-neutral-400 font-medium">Analyzing environment...</p>
          </div>
        )}

        {!loading && error && (
          <p className="text-rose-500 font-bold text-center py-2" role="alert">
            {error}
          </p>
        )}

        {!loading && !error && !narration && (
          <p className="text-neutral-400 text-center italic py-2">
            No environment narration yet. Point camera and tap &ldquo;Analyze Scene&rdquo; below.
          </p>
        )}

        {!loading && !error && narration && (
          <p className={`w-full font-medium leading-relaxed ${largeText ? "text-2xl md:text-3xl" : "text-lg md:text-xl"}`}>
            {narration}
          </p>
        )}
      </div>

      {/* TTS Audio Controls */}
      {narration && !loading && !error && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center justify-between gap-4 border-t border-neutral-100 dark:border-neutral-800 pt-4">
            {/* Playback Buttons */}
            <div className="flex items-center gap-3">
              {isSpeaking ? (
                <button
                  onClick={handlePause}
                  className="px-5 py-4 bg-amber-500 hover:bg-amber-400 text-white rounded-2xl font-bold flex items-center gap-2 transition shadow-md active:scale-95 touch-manipulation min-w-[100px]"
                  aria-label="Pause spoken narration"
                >
                  <Pause className="w-5 h-5" />
                  Pause
                </button>
              ) : (
                <button
                  onClick={handleResume}
                  className="px-5 py-4 bg-emerald-600 hover:bg-emerald-500 text-white rounded-2xl font-bold flex items-center gap-2 transition shadow-md active:scale-95 touch-manipulation min-w-[100px]"
                  aria-label="Speak narration aloud"
                >
                  <Volume2 className="w-5 h-5" />
                  Read
                </button>
              )}
              <button
                onClick={handleStop}
                className="p-4 bg-rose-600 hover:bg-rose-500 text-white rounded-2xl transition shadow-md active:scale-95 touch-manipulation"
                aria-label="Stop speaking"
              >
                <Square className="w-5 h-5 fill-current" />
              </button>
            </div>

            {/* Autoplay toggle */}
            <label className="flex items-center gap-3 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={autoplaySpeech}
                onChange={(e) => updateAccessibility({ autoplaySpeech: e.target.checked })}
                className="w-6 h-6 rounded-md border-neutral-300 text-emerald-600 focus:ring-emerald-500"
                aria-label="Autoplay narration aloud upon generation"
              />
              <span className={`font-semibold ${largeText ? "text-lg" : "text-sm"}`}>Autoplay Audio</span>
            </label>
          </div>

          {/* Speech speed slider */}
          <div className="flex flex-col gap-2 bg-neutral-50 dark:bg-neutral-950 p-4 rounded-2xl border border-neutral-100 dark:border-neutral-900">
            <div className="flex justify-between items-center">
              <span className={`font-bold ${largeText ? "text-lg" : "text-sm"}`}>Speaking Speed</span>
              <span className="text-xs text-neutral-400 bg-neutral-200 dark:bg-neutral-800 px-2.5 py-1 rounded-full font-mono font-bold">
                {speechRate.toFixed(1)}x
              </span>
            </div>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.1"
              value={speechRate}
              onChange={changeSpeechRate}
              className="w-full h-3 bg-neutral-200 dark:bg-neutral-850 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              aria-label={`Speech rate: current speed ${speechRate.toFixed(1)} times normal`}
            />
          </div>
        </div>
      )}
    </div>
  );
}
