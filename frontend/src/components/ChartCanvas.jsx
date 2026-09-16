import { useEffect, useRef } from "react";
import Chart from "chart.js/auto";

// A chart only counts as renderable if at least one dataset has at least
// one finite numeric value. This is what stops the "axes render, nothing
// plots" failure mode: instead of handing Chart.js an all-null/all-NaN
// dataset (which it happily renders as an empty 0-1 axis), we detect that
// up front and show an explicit empty state instead.
export function hasRenderableChartData(chart) {
  if (!chart) return false;
  const datasets = Array.isArray(chart.datasets) ? chart.datasets : [];
  return datasets.some(
    (dataset) =>
      Array.isArray(dataset.data) &&
      dataset.data.some((value) => value !== null && value !== undefined && Number.isFinite(Number(value)))
  );
}

const PALETTE = ["#ffb347", "#5eead4", "#7c9cff", "#ff8c8c", "#c9a6ff", "#8fe38f"];

export default function ChartCanvas({ type = "line", labels = [], datasets = [] }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const chartType =
  type === "multi_line"
    ? "line"
    : type === "multi_bar"
      ? "bar"
      : type;

  const chart = { type: chartType, labels, datasets };
  const renderable = hasRenderableChartData(chart);
  const isCircular = chartType === "pie" || chartType === "doughnut";

  useEffect(() => {
    if (!renderable || !canvasRef.current) return undefined;

    // Always destroy any previous instance before creating a new one --
    // this is what stops stale state from a previous query bleeding into
    // the next chart, and what lets a new query reliably update the chart.
    const canvas = canvasRef.current;

// Chart.js may still own this canvas even when React's ref no longer
// contains the old instance. Destroy whichever instance Chart.js has
// registered before creating the replacement.
const existingChart = Chart.getChart(canvas);
if (existingChart) {
  existingChart.destroy();
}

chartRef.current = null;

    const styledDatasets = datasets.map((dataset, i) => {
      const color = PALETTE[i % PALETTE.length];
      // Invalid points become `null` (a real gap) rather than being
      // silently dropped, which would shift later points onto the wrong
      // label. Chart.js skips nulls when spanGaps is enabled.
      const cleanData = Array.isArray(dataset.data)
        ? dataset.data.map((v) => {
            const n = Number(v);
            return v === null || v === undefined || !Number.isFinite(n) ? null : n;
          })
        : [];
      const base = {
        label: dataset.label || `Series ${i + 1}`,
        data: cleanData,
        borderColor: color,
        backgroundColor: isCircular ? PALETTE : `${color}33`,
        borderWidth: 2,
        spanGaps: true,
      };
    if (chartType === "line") { 
        base.tension = 0.35;
        base.pointRadius = cleanData.length > 40 ? 0 : 3;
        base.fill = true;
      }
      return base;
    });

    const instance = new Chart(canvas, {
      type: chartType,
      data: { labels, datasets: styledDatasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        interaction: { mode: "index", intersect: false },
        // No hardcoded min/max here -- Chart.js auto-scales from the real
        // (now-validated) data, which is what prevents the flat 0-1 axis.
        scales: isCircular
          ? undefined
          : {
              x: { grid: { color: "rgba(231,236,245,0.06)" }, ticks: { color: "#8d97ac" } },
              y: { grid: { color: "rgba(231,236,245,0.06)" }, ticks: { color: "#8d97ac" } },
            },
        plugins: {
          legend: {
            display: datasets.length > 1 || isCircular,
            labels: { color: "#8d97ac" },
          },
        },
      },
        });

    chartRef.current = instance;

    return () => {
      const registeredChart = Chart.getChart(canvas);

      if (registeredChart === instance) {
        instance.destroy();
      }

      if (chartRef.current === instance) {
        chartRef.current = null;
      }
    };
    // Re-run whenever the actual chart content changes -- not just when
    // the labels/datasets array *references* change, so a new response
    // with equivalent-looking-but-different data still updates the chart.
    // eslint-disable-next-line react-hooks/exhaustive-deps
 }, [renderable, chartType, JSON.stringify(labels), JSON.stringify(datasets)]);

  if (!renderable) {
    return (
      <div className="chart-empty">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M4 19V5M4 19h16M8 15v2M12 11v6M16 8v9" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span>No chart data for this result.</span>
      </div>
    );
  }

  return <canvas ref={canvasRef} role="img" aria-label={datasets[0]?.label || "Chart"} />;
}
