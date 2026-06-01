"use client";

import React, { useRef, useState, useEffect } from "react";
import { Camera, StopCircle, Trash2, ShieldAlert, CheckCircle, RefreshCw } from "lucide-react";

export default function CameraTestPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isSecure, setIsSecure] = useState<boolean | null>(null);
  const [hasMediaDevices, setHasMediaDevices] = useState<boolean | null>(null);

  // Helper to log both to browser console and to on-screen debug panel
  const log = (message: string) => {
    console.log(`[CameraDebug] ${message}`);
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev, `[${timestamp}] ${message}`]);
  };

  // Perform basic checks on load
  useEffect(() => {
    if (typeof window !== "undefined") {
      setIsSecure(window.isSecureContext);
      setHasMediaDevices(!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia));
      log("Environment diagnostics checked.");
      log(`window.isSecureContext: ${window.isSecureContext}`);
      log(`navigator.mediaDevices support: ${!!navigator.mediaDevices}`);
      log(`Browser UserAgent: ${navigator.userAgent}`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleStartCamera = async () => {
    log("User interaction detected: initiating camera start...");
    setError(null);

    // Stop any existing stream first
    if (stream) {
      log("Stopping prior active stream...");
      stream.getTracks().forEach((track) => track.stop());
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const errMsg = "Camera API (getUserMedia) is not supported on this browser/origin.";
      log(`ERROR: ${errMsg}`);
      setError(errMsg);
      return;
    }

    try {
      // 1. Request rear camera (facingMode: "environment")
      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: "environment",
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      };

      log(`Requesting camera permission with constraints: ${JSON.stringify(constraints)}`);
      
      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      log("Permission granted. Camera stream received.");
      setStream(mediaStream);

      // 2. Attach stream to HTML video preview element
      if (videoRef.current) {
        log("Attaching stream to video element srcObject...");
        videoRef.current.srcObject = mediaStream;

        // 3. Play stream
        log("Invoking videoRef.current.play()...");
        try {
          await videoRef.current.play();
          log("Video play() resolved - preview streaming active.");
        } catch (playErr: any) {
          log(`WARNING: Video play() failed: ${playErr.name} - ${playErr.message}`);
          log("Autoplay policies or browser restrictions may require additional user gesture.");
          setError("Stream was blocked from playing automatically. Tap video viewport to resume.");
        }
      } else {
        log("ERROR: Video DOM element is not ready.");
        setError("Preview window reference failed. Please refresh.");
      }
    } catch (err: any) {
      log(`ERROR: getUserMedia failed with ${err.name} - ${err.message}`);
      
      // Fallback: request any camera
      log("Attempting fallback constraint: { video: true } (Any camera)...");
      try {
        const fallbackStream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false
        });
        log("Fallback permission granted. Stream received.");
        setStream(fallbackStream);
        if (videoRef.current) {
          videoRef.current.srcObject = fallbackStream;
          await videoRef.current.play();
          log("Fallback stream playback active.");
        }
      } catch (fallbackErr: any) {
        log(`ERROR: Fallback constraint failed with ${fallbackErr.name} - ${fallbackErr.message}`);
        
        let message = "Failed to access mobile camera.";
        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
          message = "Camera access denied. Please allow camera permissions in your browser address bar.";
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
          message = "No camera hardware located on this device.";
        } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
          message = "Camera is currently locked by another application, process, or tab.";
        }
        setError(message);
      }
    }
  };

  const handleStopCamera = () => {
    log("User interaction: stopping camera...");
    if (stream) {
      stream.getTracks().forEach((track) => {
        track.stop();
        log(`Stopped track: ${track.label}`);
      });
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    log("Camera stopped.");
  };

  const clearLogs = () => {
    setLogs([]);
  };

  return (
    <main className="min-h-screen w-full bg-neutral-950 text-white flex flex-col p-4 md:p-8 font-sans">
      <div className="w-full max-w-md mx-auto flex flex-col gap-6 flex-1">
        
        {/* Header Diagnostics */}
        <header className="flex flex-col gap-2">
          <h1 className="text-2xl font-extrabold tracking-wide flex items-center gap-2">
            <Camera className="w-6 h-6 text-emerald-500" />
            Camera Diagnostics
          </h1>
          <div className="grid grid-cols-2 gap-2 text-xs font-semibold">
            <div className={`p-2 rounded-lg flex items-center gap-1.5 ${isSecure ? "bg-emerald-950/40 text-emerald-400 border border-emerald-800" : "bg-rose-950/40 text-rose-400 border border-rose-800"}`}>
              {isSecure ? <CheckCircle className="w-3.5 h-3.5" /> : <ShieldAlert className="w-3.5 h-3.5" />}
              <span>{isSecure ? "Secure Origin" : "Insecure Origin"}</span>
            </div>
            <div className={`p-2 rounded-lg flex items-center gap-1.5 ${hasMediaDevices ? "bg-emerald-950/40 text-emerald-400 border border-emerald-800" : "bg-rose-950/40 text-rose-400 border border-rose-800"}`}>
              {hasMediaDevices ? <CheckCircle className="w-3.5 h-3.5" /> : <ShieldAlert className="w-3.5 h-3.5" />}
              <span>{hasMediaDevices ? "MediaDevices Ready" : "API Blocked"}</span>
            </div>
          </div>
        </header>

        {/* Viewfinder Preview */}
        <section className="relative w-full aspect-[4/3] rounded-3xl overflow-hidden border-4 border-neutral-800 bg-neutral-900 flex items-center justify-center">
          {stream ? (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
              aria-label="Camera test stream preview"
            />
          ) : (
            <div className="text-center p-6 flex flex-col items-center gap-2 text-neutral-500">
              <Camera className="w-12 h-12 mb-2" />
              <p className="font-bold">Camera Stopped</p>
              <p className="text-xs max-w-[200px]">Tap &ldquo;Start Camera&rdquo; below to trigger permission prompt.</p>
            </div>
          )}
          
          {error && (
            <div className="absolute inset-x-4 bottom-4 p-3 bg-rose-900/90 text-rose-100 rounded-xl border border-rose-700 text-xs font-bold flex items-center gap-2 animate-bounce">
              <ShieldAlert className="w-5 h-5 shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </section>

        {/* Controls */}
        <section className="flex gap-4">
          <button
            onClick={handleStartCamera}
            className="flex-1 py-5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-2xl font-extrabold text-lg shadow-lg active:scale-95 transition flex items-center justify-center gap-2.5 touch-manipulation"
          >
            <Camera className="w-5 h-5" />
            Start Camera
          </button>
          {stream && (
            <button
              onClick={handleStopCamera}
              className="px-5 bg-rose-600 hover:bg-rose-500 text-white rounded-2xl transition active:scale-95 flex items-center justify-center touch-manipulation"
              aria-label="Stop camera stream"
            >
              <StopCircle className="w-6 h-6" />
            </button>
          )}
        </section>

        {/* Debug Console Logs */}
        <section className="flex-1 flex flex-col min-h-[150px] bg-neutral-900 border border-neutral-800 rounded-2xl overflow-hidden">
          <div className="px-4 py-2 border-b border-neutral-800 flex justify-between items-center text-xs font-bold text-neutral-400">
            <span>LIVE LOGGER CONSOLE</span>
            <button onClick={clearLogs} className="flex items-center gap-1 hover:text-white transition">
              <Trash2 className="w-3.5 h-3.5" />
              Clear
            </button>
          </div>
          <div className="flex-1 p-3 overflow-y-auto font-mono text-[10px] md:text-xs leading-relaxed flex flex-col gap-1.5 select-text">
            {logs.length === 0 ? (
              <span className="text-neutral-600 italic">Logs will appear here once you tap Start Camera.</span>
            ) : (
              logs.map((logItem, idx) => (
                <div key={idx} className={logItem.includes("ERROR") ? "text-rose-400 font-bold" : logItem.includes("WARNING") ? "text-amber-400" : "text-neutral-300"}>
                  {logItem}
                </div>
              ))
            )}
          </div>
        </section>

      </div>
    </main>
  );
}
