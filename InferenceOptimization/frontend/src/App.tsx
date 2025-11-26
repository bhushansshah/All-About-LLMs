import { PromptForm } from "./components/PromptForm";
import { TokenStream } from "./components/TokenStream";
import { MetricsPanel } from "./components/MetricsPanel";
import { useLLMStream } from "./hooks/useLLMStream";

const App = () => {
  const { stream, resetState, isStreaming, output, metrics, error } = useLLMStream();

  return (
    <div className="gradient-background min-h-screen w-full overflow-x-hidden pb-16">
      <div className="mx-auto flex max-w-6xl flex-col gap-8 px-6 pt-12">
        <header className="flex flex-col gap-4">
          <span className="text-xs font-semibold uppercase tracking-[0.4em] text-primary-foreground/70">
            Version 1 · Baseline
          </span>
          <h1 className="text-4xl font-bold text-white md:text-5xl">
            LLM Inference Optimization Demo
          </h1>
          <p className="max-w-2xl text-lg text-white/70">
            Explore baseline latency, throughput, and per-request metrics while streaming token-by-token output from a local llama.cpp runtime.
          </p>
        </header>

        {error && (
          <div className="rounded-2xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
            {error}
          </div>
        )}

        <MetricsPanel metrics={metrics} isStreaming={isStreaming} />

        <main className="grid gap-8 lg:grid-cols-2">
          <PromptForm onSubmit={stream} onReset={resetState} isStreaming={isStreaming} />
          <TokenStream output={output} isStreaming={isStreaming} />
        </main>
      </div>
    </div>
  );
};

export default App;
