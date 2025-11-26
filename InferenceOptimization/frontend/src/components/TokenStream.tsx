import { useEffect, useRef } from "react";

export type TokenStreamProps = {
  output: string;
  isStreaming: boolean;
};

export const TokenStream = ({ output, isStreaming }: TokenStreamProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [output]);

  return (
    <section className="relative flex h-full flex-col rounded-3xl border border-surface-border bg-surface/80 p-6 shadow-2xl shadow-primary/10 backdrop-blur">
      <header className="flex items-center justify-between pb-4">
        <div>
          <h2 className="text-xl font-semibold text-white">Token Stream</h2>
          <p className="text-sm text-white/60">Live output streamed from the backend service.</p>
        </div>
        <span className="rounded-full border border-primary/40 px-4 py-1 text-xs font-medium uppercase tracking-[0.3em] text-primary-foreground/80">
          {isStreaming ? "Streaming" : "Idle"}
        </span>
      </header>
      <div
        ref={containerRef}
        className="mt-2 flex-1 overflow-y-auto rounded-2xl border border-surface-border/60 bg-surface-muted/60 px-5 py-4 text-base leading-relaxed text-white/90"
      >
        {output ? (
          <pre className="whitespace-pre-wrap text-white/90">{output}</pre>
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-white/50">
            <span className="text-4xl">💡</span>
            <p>
              Submit a prompt to start streaming tokens. Metrics update in real time as tokens arrive.
            </p>
          </div>
        )}
      </div>
    </section>
  );
};
