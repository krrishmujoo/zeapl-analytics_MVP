import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";
import * as api from "./api";

vi.mock("./api", () => ({
  getDatasetProfile: vi.fn(),
  analyzeDataset: vi.fn(),
}));

vi.mock("chart.js/auto", () => ({
  default: class MockChart {
    constructor() {}
    destroy() {}
  },
}));

const PROFILE = {
  dataset_id: "orders",
  dataset_fingerprint: "fp1",
  date_column: "order_date",
  row_count: 500,
  time_range: {
    start: "2025-01-01",
    end: "2025-06-01",
    days: 150,
  },
  metrics: {
    invoice_value: {
      metric_key: "invoice_value",
      field: "invoice_value",
      display_name: "Invoice Value",
      semantic_type: "monetary",
      default_aggregation: "sum",
      forecastable: true,
      supported_granularities: ["day", "week"],
    },
    quantity: {
      metric_key: "quantity",
      field: "quantity",
      display_name: "Quantity",
      semantic_type: "discrete_numeric",
      default_aggregation: "sum",
      forecastable: true,
      supported_granularities: ["day", "week"],
    },
  },
  dimensions: ["region"],
  warnings: [],
};

async function waitForProfileToLoad() {
  const matches = await screen.findAllByText("Invoice Value");
  expect(matches.length).toBeGreaterThan(0);
}

function askQuestion(question) {
  const input = screen.getByLabelText(/ask a question about your dataset/i);
  fireEvent.change(input, { target: { value: question } });
  fireEvent.keyDown(input, { key: "Enter", code: "Enter" });
  return input;
}

beforeEach(() => {
  vi.clearAllMocks();

  api.getDatasetProfile.mockResolvedValue(PROFILE);

  api.analyzeDataset.mockResolvedValue({
    status: "success",
    result: null,
    insights: null,
    chart_spec: null,
    chart_data: null,
    prediction: null,
    errors: [],
  });
});

describe("App", () => {
  it("loads the profile and renders discovered metrics", async () => {
    render(<App />);

    await waitForProfileToLoad();

    expect(screen.getAllByText("Quantity").length).toBeGreaterThan(0);
    expect(api.getDatasetProfile).toHaveBeenCalledTimes(1);
  });

  it("renders profile-driven suggested questions, not hard-coded ones", async () => {
    render(<App />);
    await waitForProfileToLoad();

    expect(screen.getAllByText(/total invoice value/i).length).toBeGreaterThan(0);
  });

  it('asking "total invoice value" calls analyzeDataset (/api/v2/analyze)', async () => {
    api.analyzeDataset.mockResolvedValue({
      status: "success",
      request: {
        operation: "total",
        metric_keys: ["invoice_value"],
      },
      result: {
        operation: "total",
        results: {
          invoice_value: 183252090,
        },
      },
      answer: "The total invoice value is 183,252,090.",
      insights: {
        summary: "Total Invoice Value was 183,252,090.00.",
      },
      chart_spec: null,
      chart_data: null,
      prediction: null,
      errors: [],
      dataset: {
        dataset_id: "orders",
      },
    });

    render(<App />);
    await waitForProfileToLoad();

    askQuestion("What is the total invoice value?");

    await waitFor(() => {
      expect(api.analyzeDataset).toHaveBeenCalledWith(
        "What is the total invoice value?",
        expect.any(Object),
        expect.any(Object)
      );
    });
  });

  it("renders a scalar result as an actual value", async () => {
    api.analyzeDataset.mockResolvedValue({
      status: "success",
      result: {
        operation: "total",
        results: {
          invoice_value: 183252090,
        },
      },
      answer: "The total invoice value is 183,252,090.",
      insights: null,
      chart_spec: null,
      chart_data: null,
      prediction: null,
      errors: [],
    });

    render(<App />);
    await waitForProfileToLoad();

    askQuestion("total invoice value");

    expect(await screen.findByText(/183,252,090/)).toBeInTheDocument();
  });

  it("renders a grouped result as a table", async () => {
    api.analyzeDataset.mockResolvedValue({
      status: "success",
      result: {
        operation: "group_by",
        dimension: "region",
        results: {
          North: 1200,
          South: 800,
        },
      },
      answer: "North has the highest invoice value.",
      insights: null,
      chart_spec: {
        chart_type: "bar",
        x_key: "region",
        y_key: "value",
        title: "Invoice Value by Region",
      },
      chart_data: [
        { region: "North", value: 1200 },
        { region: "South", value: 800 },
      ],
      prediction: null,
      errors: [],
    });

    render(<App />);
    await waitForProfileToLoad();

    askQuestion("invoice value by region");

    expect(await screen.findByText("North")).toBeInTheDocument();
    expect(screen.getByText("South")).toBeInTheDocument();
    expect(screen.getByText("1,200")).toBeInTheDocument();
    expect(screen.getByText("800")).toBeInTheDocument();
  });

  it("renders a forecast response with periods and an explicit quality level", async () => {
    api.analyzeDataset.mockResolvedValue({
      status: "success",
      result: null,
      insights: null,
      chart_spec: null,
      chart_data: null,
      errors: [],
      prediction: {
        status: "success",
        metric: {
          key: "quantity",
          display_name: "Quantity",
        },
        forecast_config: {
          granularity: "week",
          horizon: 2,
        },
        model: {
          algorithm: "Linear Regression",
          validation: {
            mae: 10,
            rmse: 12,
            r2: 0.85,
          },
        },
        quality: {
          score: 0.85,
          level: "good",
          safe_to_act_on: true,
          reasons: [],
        },
        forecast: {
          total: 230,
          average: 115,
          periods: [
            {
              period: "2025-W24",
              predicted_value: 110,
              lower_bound: 100,
              upper_bound: 120,
            },
            {
              period: "2025-W25",
              predicted_value: 120,
              lower_bound: 108,
              upper_bound: 132,
            },
          ],
        },
        warnings: [],
      },
    });

    render(<App />);
    await waitForProfileToLoad();

    askQuestion("forecast quantity");

    expect(await screen.findByText(/quality:\s*good/i)).toBeInTheDocument();
    expect(screen.getByText("2025-W24")).toBeInTheDocument();
    expect(screen.getByText("2025-W25")).toBeInTheDocument();
  });

  it("renders a useful message for a backend error", async () => {
    api.analyzeDataset.mockRejectedValue(
      new Error("No metric matched this question.")
    );

    render(<App />);
    await waitForProfileToLoad();

    askQuestion("show something unknown");

    expect(
      await screen.findByText(/no metric matched this question/i)
    ).toBeInTheDocument();
  });

  it("does not treat an empty success payload as a completed answer", async () => {
    api.analyzeDataset.mockResolvedValue({
      status: "success",
      result: null,
      insights: null,
      chart_spec: null,
      chart_data: null,
      prediction: null,
      errors: [],
    });

    render(<App />);
    await waitForProfileToLoad();

    askQuestion("empty result");

    expect(
      await screen.findByText(/backend returned nothing to show/i)
    ).toBeInTheDocument();
  });

  it("allows cancelling an in-progress request", async () => {
    api.analyzeDataset.mockImplementation(
      (_question, _options, { signal } = {}) =>
        new Promise((_resolve, reject) => {
          if (!signal) return;

          if (signal.aborted) {
            const error = new Error("Aborted");
            error.name = "AbortError";
            reject(error);
            return;
          }

          signal.addEventListener(
            "abort",
            () => {
              const error = new Error("Aborted");
              error.name = "AbortError";
              reject(error);
            },
            { once: true }
          );
        })
    );

    render(<App />);
    await waitForProfileToLoad();

    askQuestion("forecast quantity");

    const cancelButton = await screen.findByRole("button", {
      name: /cancel/i,
    });
    fireEvent.click(cancelButton);

    await waitFor(() => {
      const call = api.analyzeDataset.mock.calls[0];
      expect(call).toBeDefined();
      expect(call[2]?.signal?.aborted).toBe(true);
    });
  });
});