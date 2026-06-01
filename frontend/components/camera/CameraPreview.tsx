"use client";

import React, { useEffect, useState } from "react";
import { useAppStore } from "../../store/app-store";
import { Camera, RefreshCw, AlertCircle } from "lucide-react";

interface CameraPreviewProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

export default function CameraPreview({ videoRef }: CameraPreviewProps) {
  const {
    hasCameraPermission,
    cameraActive,
    setCameraPermission,
    setCameraActive,
    error,
    setError,
  } = useAppStore();

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [facingMode, setFacingMode] = useState<"user" | "environment">("environment");
  const [isInitializing, setIsInitializing] = useState(false);

  // Starts the camera stream using facingMode
  const startCamera = async (mode: "user" | "environment") => {
    setIsInitializing(true);
    setError(null);

    // Stop current stream if active
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    try {
      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: { ideal: mode },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      };

      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(mediaStream);
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      
      setCameraPermission(true);
      setCameraActive(true);
    } catch (err: any) {
      console.error("Camera connection failed:", err);
      
      // If we request environment (rear camera) and fail, try to fall back to any video device
      if (mode === "environment") {
        console.log("Rear camera failed. Retrying with default camera source...");
        try {
          const mediaStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
          });
          setStream(mediaStream);
          if (videoRef.current) {
            videoRef.current.srcObject = mediaStream;
          }
          setCameraPermission(true);
          setCameraActive(true);
          setIsInitializing(false);
          return;
        } catch (fallbackErr) {
          console.error("Fallback camera connection failed:", fallbackErr);
        }
      }

      setCameraPermission(false);
      setCameraActive(false);
      
      let message = "Camera permission was denied. Please update settings to use AccessVision.";
      if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        message = "No camera devices could be located on this machine.";
      }
      setError(message);
    } finally {
      setIsInitializing(false);
    }
  };

  // Trigger stream management when active state changes
  useEffect(() => {
    if (cameraActive) {
      startCamera(facingMode);
    } else {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        setStream(null);
      }
    }
    
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraActive, facingMode]);

  // Toggle front vs back camera
  const toggleFacingMode = (e: React.MouseEvent) => {
    e.stopPropagation();
    const nextMode = facingMode === "environment" ? "user" : "environment";
    setFacingMode(nextMode);
    
    // Announce to screen readers
    const utterance = new SpeechSynthesisUtterance(
      `Switched to ${nextMode === "environment" ? "rear" : "selfie"} camera`
    );
    window.speechSynthesis?.speak(utterance);
  };

  return (
    <div className="relative w-full h-full bg-neutral-950 flex items-center justify-center overflow-hidden rounded-3xl border-4 border-neutral-800 shadow-2xl min-h-[300px]">
      {/* Active Streaming Video */}
      {cameraActive && !error && (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover transform scale-x-100"
          aria-label="Live camera preview feed"
        />
      )}

      {/* Loading overlay */}
      {isInitializing && (
        <div className="absolute inset-0 bg-neutral-900/90 flex flex-col items-center justify-center text-white z-10">
          <RefreshCw className="w-12 h-12 animate-spin mb-4 text-emerald-400" />
          <p className="text-xl font-bold tracking-wide" aria-live="polite">Initializing camera...</p>
        </div>
      )}

      {/* Permission Denied / Error view */}
      {(!cameraActive || error) && !isInitializing && (
        <div className="absolute inset-0 bg-neutral-900 flex flex-col items-center justify-center p-6 text-center text-white z-10">
          <AlertCircle className="w-16 h-16 text-rose-500 mb-4" />
          <h2 className="text-2xl font-bold mb-2">Camera Inactive</h2>
          <p className="text-neutral-400 mb-6 max-w-sm text-sm md:text-base">{error || "Please activate camera feed."}</p>
          <button
            onClick={() => startCamera(facingMode)}
            className="px-6 py-4 bg-emerald-600 hover:bg-emerald-500 text-white rounded-full font-bold text-lg flex items-center gap-3 transition shadow-lg active:scale-95 touch-manipulation"
            aria-label="Request camera access"
          >
            <Camera className="w-6 h-6" />
            Start Camera
          </button>
        </div>
      )}

      {/* Camera Swap Toggle Overlay */}
      {cameraActive && !error && !isInitializing && (
        <button
          onClick={toggleFacingMode}
          className="absolute top-4 right-4 p-4 bg-neutral-900/80 hover:bg-neutral-800/90 text-white rounded-full transition shadow-lg border border-neutral-700 active:scale-90 touch-manipulation"
          aria-label="Switch camera source"
          title="Switch Camera"
        >
          <RefreshCw className="w-6 h-6" />
        </button>
      )}
    </div>
  );
}
