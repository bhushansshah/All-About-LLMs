import { ChangeEvent, FormEvent, useMemo, useState } from "react";
import type { GenerationOptions } from "../hooks/useLLMStream";

export type PromptFormProps = {
  onSubmit: (prompt: string, options: GenerationOptions) => void;
  onReset: () => void;
  isStreaming: boolean;
};

export const PromptForm = ({ onSubmit, onReset, isStreaming }: PromptFormProps) => {
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(256);
  const [temperature, setTemperature] = useState(0.7);
  const [useOptimizations, setUseOptimizations] = useState(false);

  const promptChars = useMemo(() => prompt.trim().length, [prompt]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onSubmit(prompt, { maxTokens, temperature, useOptimizations });
  };

  const handleReset = () => {
    setPrompt("");
    onReset();
  };

  return (
    <form onSubmit={handleSubmit} className="bg-surface-muted/60 rounded-2xl border border-surface-border p-6 shadow-2xl shadow-primary/10 backdrop-blur">
      <div className="flex flex-col gap-4">
        <label className="text-sm uppercase tracking-widest text-primary-foreground/70">Prompt</label>
        <textarea
          className="h-40 resize-none rounded-xl border border-surface-border bg-surface/80 px-4 py-3 text-base text-white placeholder:text-white/40 focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
          placeholder="Ask the model anything..."
          value={prompt}
          onChange={(event: ChangeEvent<HTMLTextAreaElement>) => setPrompt(event.target.value)}
          disabled={isStreaming}
        />
        <div className="flex flex-wrap items-center justify-between text-sm text-white/60">
          <span>{promptChars} characters</span>
          <button
            type="button"
            onClick={handleReset}
            className="rounded-full border border-transparent px-3 py-1 text-sm text-white/70 transition hover:border-white/30 hover:bg-surface-border/40"
            disabled={isStreaming && !prompt.length}
          >
            Clear
          </button>
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="flex flex-col gap-3 rounded-xl bg-surface/60 p-4">
            <label className="text-sm font-medium text-white/70">Max tokens</label>
            <input
              type="range"
              min={32}
              max={1024}
              step={32}
              value={maxTokens}
              onChange={(event: ChangeEvent<HTMLInputElement>) =>
                setMaxTokens(parseInt(event.target.value, 10))
              }
              disabled={isStreaming}
              className="accent-primary"
            />
            <span className="text-lg font-semibold text-white">{maxTokens}</span>
          </div>
          <div className="flex flex-col gap-3 rounded-xl bg-surface/60 p-4">
            <label className="text-sm font-medium text-white/70">Temperature</label>
            <input
              type="range"
              min={0}
              max={2}
              step={0.1}
              value={temperature}
              onChange={(event: ChangeEvent<HTMLInputElement>) =>
                setTemperature(parseFloat(event.target.value))
              }
              disabled={isStreaming}
              className="accent-primary"
            />
            <span className="text-lg font-semibold text-white">{temperature.toFixed(1)}</span>
          </div>
        </div>
        <div className="flex items-center gap-3 rounded-xl bg-surface/60 p-4">
          <input
            type="checkbox"
            id="optimizations"
            checked={useOptimizations}
            onChange={(e) => setUseOptimizations(e.target.checked)}
            disabled={isStreaming}
            className="h-5 w-5 accent-primary"
          />
          <label htmlFor="optimizations" className="text-sm font-medium text-white/90">
            Enable Version 2 Optimizations (Batching + KV Cache)
          </label>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="submit"
            disabled={isStreaming}
            className="flex-1 rounded-xl bg-primary px-6 py-3 text-base font-semibold text-primary-foreground shadow-lg shadow-primary/40 transition hover:shadow-primary/60 disabled:cursor-not-allowed disabled:bg-primary/60"
          >
            {isStreaming ? "Generating..." : "Generate"}
          </button>
          <button
            type="button"
            onClick={onReset}
            disabled={!isStreaming}
            className="rounded-xl border border-primary/30 px-4 py-3 text-base font-medium text-white/70 transition hover:border-primary hover:text-white disabled:cursor-not-allowed disabled:opacity-30"
          >
            Stop
          </button>
        </div>
      </div>
    </form>
  );
};
