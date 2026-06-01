"use client";

import React, { useRef, useEffect } from "react";
import { useAppStore } from "../store/app-store";
import { queryReasoningPipeline, compressImage } from "../services/api";
import { SpeechService } from "../services/speech";
import CameraPreview from "../components/camera/CameraPreview";
import NarrationBox from "../components/narration/NarrationBox";
import QABox from "../components/controls/QABox";
import { Camera, Eye, Trash2, Accessibility, RefreshCw, Sun, Moon } from "lucide-react";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  
  // Register Service Worker on mount for PWA offline capabilities
  useEffect(() => {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker
        .register("/sw.js")
        .then((reg) => console.log("Service Worker registered successfully with scope:", reg.scope))
        .catch((err) => console.error("Service Worker registration failed:", err));
    }
  }, []);
  
  const {
    capturedImage,
    setCapturedImage,
    loading,
    setLoading,
    error,
    setError,
    setNarration,
    clearQAPair,
    setCameraActive,
    accessibility,
    updateAccessibility,
  } = useAppStore();

  const { highContrast, largeText } = accessibility;

  // Triggers image frame capture from HTML5 video element
  const captureFrame = (): string | null => {
    const video = videoRef.current;
    if (!video) return null;
    
    try {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      
      const ctx = canvas.getContext("2d");
      if (!ctx) return null;
      
      // Draw video frame onto canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL("image/jpeg", 0.95);
    } catch (e) {
      console.error("Frame capture failed:", e);
      return null;
    }
  };

  // Coordinates the full "Analyze Scene" process
  const handleAnalyzeScene = async () => {
    if (loading) return;
    
    // Stop any ongoing speech narration
    SpeechService.cancel();
    setError(null);
    setNarration(null);
    clearQAPair();

    const dataUrl = captureFrame();
    if (!dataUrl) {
      setError("Failed to capture image from camera feed. Make sure camera is running.");
      return;
    }

    setCapturedImage(dataUrl);
    setLoading(true);

    try {
      // Compress frame to optimized JPEG blob (clamped to max 800px dimension)
      const compressedBlob = await compressImage(dataUrl, 800, 0.8);
      
      // Call visual question router query endpoint
      const result = await queryReasoningPipeline(compressedBlob, "Describe what you see.");
      
      setNarration(result.answer);
      
      // Haptic feedback (supported on some mobile browsers)
      if (typeof navigator !== "undefined" && navigator.vibrate) {
        navigator.vibrate([100, 50, 100]);
      }
    } catch (err: any) {
      console.error("Scene analysis failed:", err);
      setError(err.message || "Unable to complete scene analysis. Please try again.");
      setCapturedImage(null); // Reset captured image state on failure
    } finally {
      setLoading(false);
    }
  };

  // Re-initializes session to clean state
  const handleResetSession = () => {
    SpeechService.cancel();
    setCapturedImage(null);
    setNarration(null);
    clearQAPair();
    setError(null);
    setCameraActive(true);
  };

  return (
    <main
      className={`min-h-screen w-full flex flex-col items-center p-4 md:p-8 transition-colors duration-300 ${
        highContrast
          ? "bg-black text-white"
          : "bg-neutral-50 dark:bg-neutral-950 text-neutral-900 dark:text-neutral-50"
      }`}
    >
      {/* Container wrapper limiting width for mobile-first layout */}
      <div className="w-full max-w-lg flex flex-col gap-6">
        
        {/* Header Bar */}
        <header className="flex justify-between items-center py-2">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-emerald-600 text-white rounded-xl">
              <Eye className="w-6 h-6" />
            </div>
            <h1 className={`font-extrabold tracking-wide ${largeText ? "text-3xl" : "text-xl"}`}>
              AccessVision
            </h1>
          </div>
          
          {/* Accessibility Settings Toolbar */}
          <div className="flex items-center gap-2">
            {/* High Contrast Toggle */}
            <button
              onClick={() => updateAccessibility({ highContrast: !highContrast })}
              className={`p-3 rounded-xl transition-all border active:scale-95 touch-manipulation ${
                highContrast
                  ? "bg-white text-black border-white"
                  : "bg-white dark:bg-neutral-900 text-neutral-600 dark:text-neutral-300 border-neutral-200 dark:border-neutral-800"
              }`}
              aria-label={highContrast ? "Deactivate high contrast mode" : "Activate high contrast mode"}
              title="Toggle High Contrast"
            >
              <Accessibility className="w-5 h-5" />
            </button>

            {/* Font Magnifier Toggle */}
            <button
              onClick={() => updateAccessibility({ largeText: !largeText })}
              className={`p-3 rounded-xl transition-all border active:scale-95 touch-manipulation font-bold font-mono ${
                largeText
                  ? "bg-white text-black border-white"
                  : "bg-white dark:bg-neutral-900 text-neutral-600 dark:text-neutral-300 border-neutral-200 dark:border-neutral-800"
              }`}
              aria-label={largeText ? "Deactivate extra large text" : "Activate extra large text"}
              title="Toggle Text Size"
            >
              <span className="text-sm">A+</span>
            </button>
          </div>
        </header>

        {/* Live Camera View and Captured Frame Overlay */}
        <section className="relative w-full aspect-[4/3] rounded-3xl overflow-hidden shadow-lg" aria-label="Camera Frame Panel">
          <CameraPreview videoRef={videoRef} />
          
          {/* Static Captured Frame Overlay (renders on top of live camera when analyzed) */}
          {capturedImage && !loading && (
            <div className="absolute inset-0 z-0 bg-black">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={capturedImage}
                alt="Captured visual context frame being analyzed"
                className="w-full h-full object-cover"
              />
              {/* Reset layout button on top of image */}
              <button
                onClick={handleResetSession}
                className="absolute bottom-4 right-4 px-4 py-3 bg-neutral-900/90 text-white rounded-full font-bold flex items-center gap-2 border border-neutral-700 hover:bg-neutral-800 transition active:scale-95 touch-manipulation"
                aria-label="Clear current image and restart camera stream"
              >
                <Trash2 className="w-5 h-5" />
                Reset Camera
              </button>
            </div>
          )}
        </section>

        {/* Primary Action Trigger: Giant Touch Target Button */}
        <section className="w-full">
          <button
            onClick={handleAnalyzeScene}
            disabled={loading}
            className={`w-full py-6 md:py-8 rounded-3xl font-extrabold flex items-center justify-center gap-4 transition-all shadow-xl active:scale-95 touch-manipulation ${
              loading 
                ? "bg-neutral-400 dark:bg-neutral-800 text-neutral-200 dark:text-neutral-500 cursor-not-allowed" 
                : highContrast
                  ? "bg-white text-black border-4 border-white hover:bg-neutral-200"
                  : "bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white"
            } ${largeText ? "text-3xl" : "text-xl md:text-2xl"}`}
            aria-label={loading ? "Analyzing environment..." : "Analyze Scene"}
            aria-live="polite"
          >
            {loading ? (
              <RefreshCw className="w-8 h-8 animate-spin" />
            ) : (
              <Camera className="w-8 h-8" />
            )}
            {loading ? "Analyzing Scene..." : "Analyze Scene"}
          </button>
        </section>

        {/* Narrative Description Output Section */}
        <section aria-label="Narration Outputs">
          <NarrationBox />
        </section>

        {/* Follow-up Visual Q&A Section */}
        <section aria-label="Visual Questions Console">
          <QABox />
        </section>

        {/* Accessibility Reset Button */}
        {capturedImage && (
          <footer className="w-full flex justify-center py-4">
            <button
              onClick={handleResetSession}
              className={`flex items-center gap-2 font-bold px-6 py-4 rounded-2xl transition border active:scale-95 touch-manipulation ${
                highContrast
                  ? "bg-black border-white text-white hover:bg-neutral-850"
                  : "bg-neutral-100 hover:bg-neutral-250 dark:bg-neutral-900 dark:hover:bg-neutral-800 text-neutral-600 dark:text-neutral-400 border-neutral-200 dark:border-neutral-800"
              } ${largeText ? "text-lg" : "text-sm"}`}
              aria-label="Reset all session states"
            >
              <Trash2 className="w-5 h-5" />
              Reset Platform
            </button>
          </footer>
        )}
      </div>
    </main>
  );
}
