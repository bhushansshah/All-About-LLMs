import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export type GenerationMetrics = {
  tokensPerSec: number;
  requestsPerSec: number;
  averageLatency: number;
  lastRequestLatency: number;
  lastRequestTokens: number;
};

export type GenerationOptions = {
  maxTokens: number;
  temperature: number;
};

export type StreamState = {
  isStreaming: boolean;
  output: string;
  metrics: GenerationMetrics | null;
  error: string | null;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const defaultMetrics: GenerationMetrics = {
  tokensPerSec: 0,
  requestsPerSec: 0,
  averageLatency: 0,
  lastRequestLatency: 0,
  lastRequestTokens: 0,
};

export const useLLMStream = () => {
  const [state, setState] = useState<StreamState>({
    isStreaming: false,
    output: "",
    metrics: null,
    error: null,
  });
  const pollingRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const updateMetrics = useCallback((metrics: GenerationMetrics) => {
    setState((prev: StreamState) => ({ ...prev, metrics }));
  }, []);

  const pollMetrics = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/metrics`);
      if (!res.ok) {
        throw new Error(`Metrics fetch failed: ${res.status}`);
      }
      const payload = await res.json();
      updateMetrics({
        tokensPerSec: payload.tokens_per_sec ?? payload.tokensPerSec ?? 0,
        requestsPerSec: payload.requests_per_sec ?? payload.requestsPerSec ?? 0,
        averageLatency: payload.average_latency ?? payload.averageLatency ?? 0,
        lastRequestLatency:
          payload.last_request_latency ?? payload.lastRequestLatency ?? 0,
        lastRequestTokens:
          payload.last_request_tokens ?? payload.lastRequestTokens ?? 0,
      });
    } catch (error) {
      console.warn("Failed to poll metrics", error);
    }
  }, [updateMetrics]);

  useEffect(() => {
    pollMetrics();
    pollingRef.current = window.setInterval(pollMetrics, 5000);
    return () => {
      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
      }
    };
  }, [pollMetrics]);

  const resetState = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setState((prev: StreamState) => ({
      isStreaming: false,
      output: "",
      metrics: prev.metrics ?? defaultMetrics,
      error: null,
    }));
  }, []);

  const stream = useCallback(
    async (prompt: string, options: GenerationOptions) => {
      if (!prompt.trim()) {
        setState((prev: StreamState) => ({ ...prev, error: "Prompt cannot be empty" }));
        return;
      }

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setState((prev: StreamState) => ({
        ...prev,
        isStreaming: true,
        output: "",
        error: null,
      }));

      try {
        const response = await fetch(`${API_BASE}/generate`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            prompt,
            max_tokens: options.maxTokens,
            temperature: options.temperature,
          }),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Generation failed with status ${response.status}`);
        }

        if (!response.body) {
          throw new Error("Streaming is not supported in this browser");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let shouldStop = false;
        const textChunks: string[] = [];

        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          let boundary = buffer.indexOf("\n\n");

          while (boundary !== -1) {
            const rawChunk = buffer.slice(0, boundary).trim();
            buffer = buffer.slice(boundary + 2);
            boundary = buffer.indexOf("\n\n");

            if (!rawChunk.startsWith("data:")) {
              continue;
            }

            const payloadText = rawChunk.replace(/^data:\s*/, "");
            if (payloadText === "[DONE]") {
              reader.cancel().catch(() => undefined);
              shouldStop = true;
              break;
            }

            try {
              const payload = JSON.parse(payloadText);
              if (payload.type === "token") {
                textChunks.push(payload.content);
                setState((prev: StreamState) => ({
                  ...prev,
                  output: textChunks.join(""),
                }));
              }
              if (payload.type === "metrics") {
                updateMetrics({
                  tokensPerSec: payload.metrics.tokens_per_sec ?? payload.metrics.tokensPerSec ?? 0,
                  requestsPerSec:
                    payload.metrics.requests_per_sec ?? payload.metrics.requestsPerSec ?? 0,
                  averageLatency:
                    payload.metrics.average_latency ?? payload.metrics.averageLatency ?? 0,
                  lastRequestLatency:
                    payload.metrics.last_request_latency ?? payload.metrics.lastRequestLatency ?? 0,
                  lastRequestTokens:
                    payload.metrics.last_request_tokens ?? payload.metrics.lastRequestTokens ?? 0,
                });
              }
              if (payload.type === "error") {
                throw new Error(payload.message ?? "Unknown generation error");
              }
            } catch (error) {
              console.warn("Failed to parse payload", payloadText, error);
            }
          }

          if (shouldStop) {
            break;
          }
        }
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        const message = error instanceof Error ? error.message : "Unexpected error";
        setState((prev: StreamState) => ({ ...prev, error: message }));
      } finally {
        setState((prev: StreamState) => ({ ...prev, isStreaming: false }));
        abortRef.current = null;
      }
    },
    [updateMetrics]
  );

  return useMemo(
    () => ({
      ...state,
      stream,
      resetState,
      metrics: state.metrics ?? defaultMetrics,
    }),
    [state, stream, resetState]
  );
};
