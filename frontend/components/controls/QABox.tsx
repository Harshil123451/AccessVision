"use client";

import React, { useState, useEffect, useRef } from "react";
import { useAppStore } from "../../store/app-store";
import { SpeechService, SpeechRecognitionService } from "../../services/speech";
import { queryReasoningPipeline, dataURItoBlob, compressImage } from "../../services/api";
import { Mic, MicOff, Send, MessageSquare, RefreshCw, Volume2 } from "lucide-react";

export default function QABox() {
  const {
    capturedImage,
    qaHistory,
    addQAPair,
    loading: appLoading,
    setLoading,
    error,
    setError,
    isListening,
    setIsListening,
    accessibility,
    facingMode,
  } = useAppStore();

  const { largeText, highContrast, autoplaySpeech } = accessibility;

  const [question, setQuestion] = useState("");
  const [qaLoading, setQaLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll QA thread to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [qaHistory, qaLoading]);

  // Initialize Speech Recognition on mount
  useEffect(() => {
    const isSupported = SpeechRecognitionService.init(
      (text) => {
        setQuestion(text);
        // Optional: auto-submit when voice is finished
        handleSubmitQuestion(text);
      },
      () => {
        setIsListening(false);
      },
      (err) => {
        console.error("Speech Recognition error:", err);
        setError("Voice listening failed or timed out.");
        setIsListening(false);
      }
    );

    if (!isSupported) {
      console.log("Speech recognition is not supported in this browser.");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Handle visual QA submission
  const handleSubmitQuestion = async (customText?: string) => {
    const activeQuestion = customText || question;
    if (!activeQuestion.trim() || !capturedImage || qaLoading || appLoading) return;

    setQuestion("");
    setQaLoading(true);
    setError(null);
    setLoading(true);

    try {
      // 1. Convert base64 data to blob
      const rawBlob = dataURItoBlob(capturedImage);
      
      // 2. Compress image using Canvas
      const compressedBlob = await compressImage(capturedImage, 640, 0.75);

      // 3. Dispatch to API
      const result = await queryReasoningPipeline(compressedBlob, activeQuestion, facingMode === "user");

      // 4. Save to history
      const qaItem = {
        question: activeQuestion,
        answer: result.answer,
        timestamp: Date.now(),
      };
      addQAPair(qaItem);

      // 5. Speak answer aloud automatically if enabled
      if (autoplaySpeech) {
        SpeechService.speak(result.answer);
      }
    } catch (err: any) {
      console.error("QA Query failed:", err);
      setError(err.message || "Could not answer question. Please try again.");
    } finally {
      setQaLoading(false);
      setLoading(false);
    }
  };

  // Toggle voice dictation
  const toggleListening = (e: React.MouseEvent) => {
    e.preventDefault();
    if (isListening) {
      SpeechRecognitionService.stop();
      setIsListening(false);
    } else {
      setError(null);
      SpeechRecognitionService.start();
    }
  };

  const handleSpeakAnswer = (text: string) => {
    SpeechService.speak(text);
  };

  // If no image has been analyzed, display a prompt
  if (!capturedImage) {
    return null;
  }

  return (
    <div
      className={`p-6 rounded-3xl transition-all shadow-xl flex flex-col gap-6 ${
        highContrast
          ? "bg-black border-4 border-white text-white"
          : "bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 text-neutral-900 dark:text-neutral-50"
      }`}
    >
      {/* Title */}
      <div className="flex items-center gap-3">
        <div className={`p-3 rounded-2xl ${highContrast ? "bg-white text-black" : "bg-blue-100 dark:bg-blue-950 text-blue-600"}`}>
          <MessageSquare className="w-6 h-6" />
        </div>
        <div>
          <h2 className={`font-bold tracking-wide ${largeText ? "text-2xl" : "text-lg"}`}>
            Ask Visual Questions
          </h2>
          <p className="text-xs text-neutral-400">Inquire about details inside the captured frame</p>
        </div>
      </div>

      {/* QA Thread History */}
      <div
        className={`flex flex-col gap-4 overflow-y-auto max-h-[300px] p-4 rounded-2xl ${
          highContrast
            ? "bg-neutral-950 border border-white"
            : "bg-neutral-50 dark:bg-neutral-950 border border-neutral-100 dark:border-neutral-900"
        }`}
        role="log"
        aria-label="Questions and answers log"
      >
        {qaHistory.length === 0 && !qaLoading && (
          <p className="text-neutral-400 text-center italic py-4 text-sm md:text-base">
            No questions asked yet. Type or tap the microphone to speak a query.
          </p>
        )}

        {/* Render thread elements */}
        {qaHistory.map((item, idx) => (
          <div key={idx} className="flex flex-col gap-2.5">
            {/* User Question */}
            <div className="flex justify-end">
              <div className={`max-w-[85%] rounded-2xl px-4 py-3 font-semibold ${
                highContrast ? "bg-white text-black border border-white" : "bg-neutral-200 dark:bg-neutral-800 text-neutral-900 dark:text-neutral-50"
              } ${largeText ? "text-xl" : "text-sm md:text-base"}`}>
                <span className="sr-only">You asked: </span>
                {item.question}
              </div>
            </div>

            {/* AI Response */}
            <div className="flex justify-start">
              <div className={`max-w-[85%] rounded-2xl px-4 py-3 flex flex-col gap-2 ${
                highContrast ? "bg-black border border-white text-emerald-400" : "bg-emerald-50 dark:bg-emerald-950/40 text-emerald-800 dark:text-emerald-300"
              } ${largeText ? "text-xl" : "text-sm md:text-base"}`}>
                <div>
                  <span className="sr-only">AI answered: </span>
                  {item.answer}
                </div>
                {/* Audio playback option for this step */}
                <button
                  onClick={() => handleSpeakAnswer(item.answer)}
                  className="self-start mt-1 p-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl flex items-center gap-1.5 transition active:scale-90 text-xs font-bold touch-manipulation"
                  aria-label="Listen to answer again"
                >
                  <Volume2 className="w-4 h-4" />
                  Listen
                </button>
              </div>
            </div>
          </div>
        ))}

        {/* Loading Spinner during request */}
        {qaLoading && (
          <div className="flex justify-start items-center gap-2 text-neutral-400 font-medium">
            <RefreshCw className="w-5 h-5 animate-spin text-emerald-500" />
            <span className={largeText ? "text-lg" : "text-sm"}>Thinking...</span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input controls */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSubmitQuestion();
        }}
        className="flex items-center gap-3"
      >
        {/* Voice Input Button */}
        <button
          onClick={toggleListening}
          className={`p-4 rounded-2xl transition border active:scale-95 touch-manipulation shrink-0 ${
            isListening
              ? "bg-rose-600 hover:bg-rose-500 text-white border-rose-500 animate-pulse"
              : highContrast
              ? "bg-black border-white text-white hover:bg-neutral-800"
              : "bg-neutral-100 hover:bg-neutral-200 dark:bg-neutral-800 dark:hover:bg-neutral-700 text-neutral-700 dark:text-neutral-300 border-neutral-200 dark:border-neutral-750"
          }`}
          aria-label={isListening ? "Stop listening to voice question" : "Record voice question"}
          title="Toggle microphone dictation"
        >
          {isListening ? <MicOff className="w-6 h-6 animate-bounce" /> : <Mic className="w-6 h-6" />}
        </button>

        {/* Text Input */}
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={qaLoading || appLoading}
          placeholder={isListening ? "Listening..." : "Ask: 'What color is the car?'"}
          className={`flex-1 p-4 rounded-2xl focus:outline-none focus:ring-2 focus:ring-emerald-500 transition ${
            highContrast
              ? "bg-neutral-950 border border-white text-white focus:ring-white"
              : "bg-neutral-50 focus:bg-white dark:bg-neutral-950 dark:focus:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 text-neutral-950 dark:text-white"
          } ${largeText ? "text-xl placeholder:text-neutral-500" : "text-sm md:text-base placeholder:text-neutral-400"}`}
          aria-label="Type follow-up question"
        />

        {/* Submit Button */}
        <button
          type="submit"
          disabled={!question.trim() || qaLoading || appLoading}
          className="p-4 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:hover:bg-emerald-600 text-white rounded-2xl transition active:scale-95 touch-manipulation shrink-0"
          aria-label="Send question"
        >
          <Send className="w-6 h-6" />
        </button>
      </form>
    </div>
  );
}
