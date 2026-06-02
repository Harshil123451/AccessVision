import axios from "axios";
import { ReasoningResult } from "../types";

const API_TIMEOUT = 15000; // 15 seconds timeout to allow for heavy model inference workloads

const apiClient = axios.create({
  baseURL: "", // Proxied to local FastAPI backend in next.config rewrites
  timeout: API_TIMEOUT,
  headers: {
    "X-API-Key": "development-secret-key",
  },
});

/**
 * Converts a base64 DataURL string into a standard Blob representation
 */
export function dataURItoBlob(dataURI: string): Blob {
  const parts = dataURI.split(",");
  const byteString = atob(parts[1]);
  const mimeString = parts[0].split(":")[1].split(";")[0];
  
  const ab = new ArrayBuffer(byteString.length);
  const ia = new Uint8Array(ab);
  for (let i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i);
  }
  return new Blob([ab], { type: mimeString });
}

/**
 * Compresses an image data URI using an HTML5 Canvas to output an optimized JPEG blob.
 * Drastically reduces packet payload sizes for faster network transfer on mobile devices.
 */
export async function compressImage(
  dataUrl: string,
  maxDimension: number = 800,
  quality: number = 0.8
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = dataUrl;
    img.onload = () => {
      const canvas = document.createElement("canvas");
      let width = img.width;
      let height = img.height;

      // Maintain aspect ratio while clamping larger dimension
      if (width > height) {
        if (width > maxDimension) {
          height = Math.round((height * maxDimension) / width);
          width = maxDimension;
        }
      } else {
        if (height > maxDimension) {
          width = Math.round((width * maxDimension) / height);
          height = maxDimension;
        }
      }

      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Canvas context creation failed."));
        return;
      }

      // Draw and export compressed JPEG blob
      ctx.drawImage(img, 0, 0, width, height);
      canvas.toBlob(
        (blob) => {
          if (blob) {
            resolve(blob);
          } else {
            reject(new Error("Failed to export compressed canvas blob."));
          }
        },
        "image/jpeg",
        quality
      );
    };
    img.onerror = () => {
      reject(new Error("Failed to load source image for rendering."));
    };
  });
}

/**
 * Submits the captured image frame and query string to the backend reasoning pipeline.
 * Incorporates active network retries with exponential backoffs.
 */
export async function queryReasoningPipeline(
  imageBlob: Blob,
  question: string,
  isMirrored: boolean = false,
  retries: number = 2
): Promise<ReasoningResult> {
  const formData = new FormData();
  formData.append("file", imageBlob, "frame.jpg");
  formData.append("question", question);
  formData.append("is_mirrored", isMirrored ? "true" : "false");

  for (let attempt = 1; attempt <= retries + 1; attempt++) {
    try {
      const response = await apiClient.post<ReasoningResult>(
        "/api/v1/reason/query",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      return response.data;
    } catch (error: any) {
      const isNetworkError = !error.response;
      const isTimeout = error.code === "ECONNABORTED";

      // Retry if network error or timeout occurred and we have remaining attempts
      if (attempt <= retries && (isNetworkError || isTimeout)) {
        console.warn(`API call attempt ${attempt} failed. Retrying in ${attempt * 500}ms...`);
        await new Promise((r) => setTimeout(r, attempt * 500));
        continue;
      }

      // Format clean, accessible errors
      let displayMessage = "An unexpected error occurred. Please check and try again.";
      if (isTimeout) {
        displayMessage = "The AI server is taking too long to process. Please try again.";
      } else if (isNetworkError) {
        displayMessage = "Connection failed. Please ensure the backend server is running.";
      } else if (error.response?.data?.detail) {
        displayMessage = `Server alert: ${error.response.data.detail}`;
      }
      throw new Error(displayMessage);
    }
  }
  throw new Error("Unable to contact the reasoning server. Retry count exceeded.");
}
