import type { GenerationMetrics } from "../hooks/useLLMStream";

export type MetricsPanelProps = {
  metrics: GenerationMetrics;
  isStreaming: boolean;
};

const formatNumber = (value: number, digits = 2) => value.toFixed(digits);

export const MetricsPanel = ({ metrics, isStreaming }: MetricsPanelProps) => {
  const cards = [
    {
      label: "Tokens / sec",
      value: formatNumber(metrics.tokensPerSec),
      description: "Rolling average throughput over the last minute.",
    },
    {
      label: "Requests / sec",
      value: formatNumber(metrics.requestsPerSec),
      description: "Baseline request handling rate.",
    },
    {
      label: "Avg latency",
      value: `${formatNumber(metrics.averageLatency, 3)} s`,
      description: "Average round-trip latency (60s window).",
    },
    {
      label: "Last latency",
      value: `${formatNumber(metrics.lastRequestLatency, 3)} s`,
      description: `Tokens: ${metrics.lastRequestTokens}`,
    },
    {
      label: "Avg Batch Size",
      value: formatNumber(metrics.averageBatchSize ?? 0, 1),
      description: "Average concurrent requests (Version 2).",
    },
  ];

  return (
    <aside className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      {cards.map((card) => (
        <article
          key={card.label}
          className="rounded-2xl border border-surface-border bg-surface/70 p-5 shadow-xl shadow-primary/10 transition hover:-translate-y-1 hover:shadow-primary/30"
        >
          <header className="flex items-center justify-between pb-2">
            <span className="text-sm font-semibold uppercase tracking-[0.2em] text-primary-foreground/80">
              {card.label}
            </span>
            <span className={`h-2 w-2 rounded-full ${isStreaming ? "bg-primary" : "bg-white/30"}`} />
          </header>
          <p className="text-3xl font-bold text-white">{card.value}</p>
          <p className="mt-2 text-sm text-white/60">{card.description}</p>
        </article>
      ))}
    </aside>
  );
};
