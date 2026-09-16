// Central normalizers for backend responses.
//
// Everything the UI renders comes from here. Components never read raw
// backend field names directly -- if the backend contract changes, this
// file is the only place that needs to change.
//
// The primary contract is the v2 API documented in api.js:
//   GET /api/v2/profile  -> DatasetRuntimeConfig
//   GET /api/v2/analyze  -> unified analysis response

import { parseNumericValue, formatNumber, titleCaseKey } from "./numeric";
import { pickField } from "./keys";

// --- profile ----------------------------------------------------------

export function normalizeProfileResponse(raw) {
  if (!raw || typeof raw !== "object") {
    return {
      datasetId: null,
      fingerprint: null,
      dateColumn: null,
      rowCount: null,
      timeRange: null,
      metrics: [],
      dimensions: [],
      warnings: [],
      forecastableMetrics: [],
    };
  }

  const metricsSource =
    raw.metrics && typeof raw.metrics === "object" ? raw.metrics : {};

  const metrics = Object.entries(metricsSource).map(([key, m]) => {
    const source = m && typeof m === "object" ? m : {};
    const metricKey = source.metric_key || key;

    return {
      key: metricKey,
      field: source.field || metricKey,
      displayName: source.display_name || titleCaseKey(metricKey),
      semanticType: source.semantic_type || null,
      defaultAggregation: source.default_aggregation || null,
      forecastable: Boolean(source.forecastable),
      supportedGranularities: Array.isArray(source.supported_granularities)
        ? source.supported_granularities
        : [],
    };
  });

  return {
    datasetId: raw.dataset_id || null,
    fingerprint: raw.dataset_fingerprint || null,
    dateColumn: raw.date_column || null,
    rowCount: typeof raw.row_count === "number" ? raw.row_count : null,
    timeRange: raw.time_range
      ? {
          start: raw.time_range.start || null,
          end: raw.time_range.end || null,
          days: raw.time_range.days ?? null,
        }
      : null,
    metrics,
    dimensions: Array.isArray(raw.dimensions) ? raw.dimensions : [],
    warnings: Array.isArray(raw.warnings) ? raw.warnings : [],
    forecastableMetrics: metrics.filter((m) => m.forecastable),
  };
}

// --- analysis helpers -------------------------------------------------

function asStringArray(value) {
  if (!Array.isArray(value)) return [];

  return value.filter(
    (v) => typeof v === "string" && v.trim().length > 0
  );
}

function normalizeInsights(raw) {
  const out = [];
  const src = raw?.insights;

  if (typeof src === "string" && src.trim()) {
    out.push(src.trim());
  } else if (Array.isArray(src)) {
    for (const item of src) {
      if (typeof item === "string") {
        out.push(item);
      } else if (
        item &&
        typeof item === "object" &&
        typeof item.text === "string"
      ) {
        out.push(item.text);
      }
    }
  } else if (src && typeof src === "object") {
    if (typeof src.summary === "string" && src.summary.trim()) {
      out.push(src.summary.trim());
    }

    out.push(...asStringArray(src.insight_messages));
    out.push(...asStringArray(src.messages));

    if (typeof src.insight === "string" && src.insight.trim()) {
      out.push(src.insight.trim());
    }
  }

  const nested = raw?.analysis?.insights;

  if (typeof nested === "string" && nested.trim()) {
    out.push(nested.trim());
  } else if (Array.isArray(nested)) {
    out.push(...asStringArray(nested));
  }
  // LLM explanation returned by the V2 backend.
  const explanationInsights = raw?.explanation?.insights;

  if (Array.isArray(explanationInsights)) {
    out.push(...asStringArray(explanationInsights));
  }
  return out;
}

function normalizeRecommendations(raw) {
  const out = [];
  const src = raw?.recommendations;

  if (typeof src === "string" && src.trim()) {
    out.push(src.trim());
  } else if (Array.isArray(src)) {
    for (const item of src) {
      if (typeof item === "string" && item.trim()) {
        out.push(item.trim());
      } else if (item && typeof item === "object") {
        const text =
          item.text || item.recommendation || item.summary || "";

        if (text) {
          out.push({
            text,
            priority: item.priority || null,
            limitations: item.limitations || item.caveat || null,
          });
        }
      }
    }
  }

  if (
    typeof raw?.recommendation === "string" &&
    raw.recommendation.trim()
  ) {
    out.push(raw.recommendation.trim());
  }

  const nested = raw?.analysis?.recommendations;

  if (Array.isArray(nested)) {
    out.push(...asStringArray(nested));
  }

  return out;
}

const PERIOD_KEY_PATTERN =
  /^\d{4}(-\d{2}){0,2}$|^q[1-4]\s?\d{4}$/i;

function looksLikeTimeSeries(results) {
  const keys = Object.keys(results);

  if (keys.length < 2) return false;

  return keys.every((key) =>
    PERIOD_KEY_PATTERN.test(key.trim())
  );
}

function normalizeScalar(result) {
  if (!result || typeof result !== "object") return null;
  if (result.dimension) return null;

  const results = result.results;

    // Ranking results contain an object whose metric value is an
    // array of { dimension, value, ... } rows. These must not be
    // interpreted as ordinary grouped scalar values.
    const rankingRows = [];

    if (
      result.dimension &&
      results &&
      typeof results === "object" &&
      !Array.isArray(results)
    ) {
      for (const [metricKey, rawValue] of Object.entries(results)) {
        if (
          Array.isArray(rawValue) &&
          rawValue.length > 0 &&
          rawValue.every(
            (item) =>
              item &&
              typeof item === "object" &&
              "dimension" in item &&
              "value" in item
          )
        ) {
          for (const item of rawValue) {
            rankingRows.push({
              dimension: String(item.dimension),
              value: parseNumericValue(item.value),
              formatted:
                parseNumericValue(item.value) === null
                  ? "—"
                  : formatNumber(parseNumericValue(item.value)),
            });
          }

          return {
            operation: result.operation || null,
            dimension: result.dimension,
            dimensionLabel: titleCaseKey(result.dimension),
            rows: rankingRows,
            ranking: true,
            metricKey,
            metricLabel: titleCaseKey(metricKey),
          };
        }
      }
    }

  if (
    !results ||
    typeof results !== "object" ||
    Array.isArray(results)
  ) {
    return null;
  }

  if (looksLikeTimeSeries(results)) return null;

  // Multi-metric comparisons contain one period-keyed object per metric.
  // These nested objects are chart series rather than scalar values.
  const isMultiMetricTimeSeries = Object.values(results).every(
    (value) =>
      value &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      looksLikeTimeSeries(value)
  );

  if (isMultiMetricTimeSeries) return null;

  const entries = Object.entries(results).map(
    ([key, rawValue]) => {
      const isRanking =
  Array.isArray(rawValue) &&
  rawValue.every(
    (item) =>
      item &&
      typeof item === "object" &&
      "dimension" in item &&
      "value" in item
  );

      if (isRanking) {
        return {
          key,
          label: titleCaseKey(key),
          value: rawValue,
          formatted: null,
          type: "user-ranking",
        };
      }

      const value = parseNumericValue(rawValue);

      return {
        key,
        label: titleCaseKey(key),
        value,
        formatted:
          value === null
            ? rawValue === null || rawValue === undefined
              ? "—"
              : String(rawValue)
            : formatNumber(value),
        type: "value",
      };
    }
  );

  if (!entries.length) return null;

  return {
    operation: result.operation || null,
    operationLabel: result.operation
      ? titleCaseKey(result.operation)
      : null,
    entries,
    recordCount:
      typeof result.record_count === "number"
        ? result.record_count
        : null,
  };
}

function normalizeNestedRanking(result) {
  if (
    !result ||
    typeof result !== "object" ||
    result.operation !== "nested_ranking" ||
    !Array.isArray(result.results)
  ) {
    return null;
  }

  const rankingLevels = Array.isArray(result.ranking_levels)
    ? result.ranking_levels
    : [];

  if (rankingLevels.length !== 2) {
    return null;
  }

  const outerLevel = rankingLevels[0];
  const innerLevel = rankingLevels[1];
  const metricKey = result.metric || "value";

  const rows = result.results
    .filter(
      (item) =>
        item &&
        typeof item === "object" &&
        Array.isArray(item.children)
    )
    .map((item, outerIndex) => {
      const outerValue = parseNumericValue(item.value);

      return {
        rank:
          typeof item.rank === "number"
            ? item.rank
            : outerIndex + 1,
        entity: String(item.entity ?? "Unknown"),
        value: outerValue,
        formatted:
          outerValue === null
            ? "—"
            : formatNumber(outerValue),
        children: item.children.map(
          (child, innerIndex) => {
            const childValue = parseNumericValue(
              child?.value
            );

            return {
              rank:
                typeof child?.rank === "number"
                  ? child.rank
                  : innerIndex + 1,
              entity: String(
                child?.entity ?? "Unknown"
              ),
              value: childValue,
              formatted:
                childValue === null
                  ? "—"
                  : formatNumber(childValue),
            };
          }
        ),
      };
    });

  return {
    direction: result.direction || "top",
    metricKey,
    metricLabel: titleCaseKey(metricKey),
    outerDimension: outerLevel.dimension,
    outerDimensionLabel: titleCaseKey(
      outerLevel.dimension
    ),
    outerTopN: outerLevel.top_n,
    innerDimension: innerLevel.dimension,
    innerDimensionLabel: titleCaseKey(
      innerLevel.dimension
    ),
    innerTopN: innerLevel.top_n,
    rows,
  };
}

function normalizeGrouped(result) {
  if (!result || typeof result !== "object") return null;

  const results = result.results;

  if (
    !results ||
    typeof results !== "object" ||
    Array.isArray(results)
  ) {
    return null;
  }

  // ---------------------------------------------------------------
  // Generic ranking response
  //
  // Example backend shape:
  //
  // result = {
  //   operation: "top_users",
  //   dimension: "product_id",
  //   results: {
  //     invoice_value: [
  //       {
  //         dimension: "P025",
  //         value: 19923150,
  //         product_id: "P025"
  //       }
  //     ]
  //   }
  // }
  //
  // Do this BEFORE normal grouped-object processing so arrays of
  // objects never become "[object Object]".
  // ---------------------------------------------------------------

  const isRankingOperation =
    result.operation === "top_users" ||
    result.operation === "bottom_users";

  if (isRankingOperation && result.dimension) {
    for (const [metricKey, rawValue] of Object.entries(results)) {
      if (
        Array.isArray(rawValue) &&
        rawValue.length > 0 &&
        rawValue.every(
          (item) =>
            item &&
            typeof item === "object" &&
            "dimension" in item &&
            "value" in item
        )
      ) {
        return {
          operation: result.operation,
          dimension: result.dimension,
          dimensionLabel: titleCaseKey(result.dimension),
          rows: rawValue.map((item) => {
            const value = parseNumericValue(item.value);

            return {
              key: String(item.dimension),
              dimension: String(item.dimension),
              value,
              formatted:
                value === null
                  ? "—"
                  : formatNumber(value),
            };
          }),
          ranking: true,
          metricKey,
          metricLabel: titleCaseKey(metricKey),
        };
      }
    }
  }

  // ---------------------------------------------------------------
  // Normal grouped / trend response
  // ---------------------------------------------------------------

  const explicitDimension = Boolean(result.dimension);

  const inferredTimeSeries =
    !explicitDimension &&
    looksLikeTimeSeries(results);

  if (!explicitDimension && !inferredTimeSeries) {
    return null;
  }

  const rows = Object.entries(results).map(
    ([key, rawValue]) => {
      const value = parseNumericValue(rawValue);

      return {
        key,
        value,
        formatted:
          value === null
            ? rawValue === null || rawValue === undefined
              ? "—"
              : String(rawValue)
            : formatNumber(value),
      };
    }
  );

  if (!rows.length) return null;

  return {
    operation: result.operation || null,
    dimension: result.dimension || "period",
    dimensionLabel: result.dimension
      ? titleCaseKey(result.dimension)
      : "Period",
    rows,
  };
}

function chartFromRows(
  rows,
  xKey,
  yKey,
  type,
  title,
  yKeys = []
) {
  if (!Array.isArray(rows) || !rows.length) return null;

  const xCandidates = [
    xKey,
    "period",
    "date",
    "month",
    "scan_month",
    "scan_date",
    "label",
    "x",
  ].filter(Boolean);

  const labels = rows.map((row) => {
    const value = pickField(row, xCandidates);

    return value === undefined || value === null
      ? ""
      : String(value);
  });

  const seriesKeys =
    Array.isArray(yKeys) && yKeys.length
      ? yKeys
      : yKey
        ? [yKey]
        : [null];

  const datasets = seriesKeys.map((seriesKey) => {
    const yCandidates = seriesKey
      ? [seriesKey, titleCaseKey(seriesKey)]
      : ["value", "y"];

    // Preserve existing fallbacks for single-series responses.
    if (seriesKeys.length === 1) {
      yCandidates.push("value", "y");
    }

    return {
      label:
        seriesKeys.length > 1
          ? titleCaseKey(seriesKey)
          : title ||
            (seriesKey
              ? titleCaseKey(seriesKey)
              : "Value"),
      data: rows.map((row) =>
        parseNumericValue(pickField(row, yCandidates))
      ),
    };
  });

  return {
    type: type || "line",
    title: title || null,
    labels,
    datasets,
  };
}

function normalizeChart(raw) {
  // Accept a pre-normalized chart shape.
  if (raw?.chart && typeof raw.chart === "object") {
    const chart = raw.chart;

    if (
      Array.isArray(chart.labels) &&
      Array.isArray(chart.datasets)
    ) {
      return {
        type: chart.type || "line",
        title: chart.title || null,
        labels: chart.labels,
        datasets: chart.datasets.map((dataset) => ({
          label: dataset.label || "Value",
          data: Array.isArray(dataset.data)
            ? dataset.data.map((value) =>
                parseNumericValue(value)
              )
            : [],
        })),
      };
    }

    if (Array.isArray(chart.data)) {
      return chartFromRows(
        chart.data,
        chart.x,
        chart.y,
        chart.type || "line",
        chart.title || null
      );
    }
  }

  // V2 chart shape: chart_spec describes it and chart_data holds rows.
  const spec = raw?.chart_spec;
  const data = raw?.chart_data;

  if (spec && Array.isArray(data) && data.length) {
    return chartFromRows(
      data,
      spec.x_key,
      spec.y_key,
      spec.chart_type,
      spec.title,
      spec.y_keys
    );
  }

  return null;
}

function normalizeForecast(prediction) {
  if (!prediction || typeof prediction !== "object") {
    return null;
  }

  const forecast = prediction.forecast || {};

  const periods = Array.isArray(forecast.periods)
    ? forecast.periods.map((period) => ({
        period:
          period?.period ??
          period?.date ??
          period?.label ??
          "",
        predicted: parseNumericValue(
          period?.predicted_value ??
            period?.value ??
            period?.y
        ),
        lower: parseNumericValue(period?.lower_bound),
        upper: parseNumericValue(period?.upper_bound),
      }))
    : [];

  const quality = prediction.quality || {};
  const model = prediction.model || {};

  return {
    status: prediction.status || null,
    metricKey: prediction.metric?.key || null,
    metricLabel:
      prediction.metric?.display_name ||
      titleCaseKey(prediction.metric?.key || "metric"),
    granularity:
      prediction.forecast_config?.granularity || null,
    horizon:
      prediction.forecast_config?.horizon ?? null,
    algorithm: model.algorithm || null,
    validation: model.validation
      ? {
          mae: parseNumericValue(model.validation.mae),
          rmse: parseNumericValue(model.validation.rmse),
          r2: parseNumericValue(model.validation.r2),
        }
      : null,
    quality: {
      score: parseNumericValue(quality.score),
      level: quality.level || null,
      safeToActOn: Boolean(quality.safe_to_act_on),
      reasons: asStringArray(quality.reasons),
    },
    total: parseNumericValue(forecast.total),
    average: parseNumericValue(forecast.average),
    periods,
    insightMessages: normalizeInsights({
      insights: prediction.insights,
    }),
    recommendations: normalizeRecommendations({
      recommendations: prediction.recommendations,
    }),
    warnings: asStringArray(prediction.warnings),
    llmUsed: Boolean(prediction.llm_used),
  };
}

// --- analysis entry point ---------------------------------------------

// --- analysis entry point ---------------------------------------------

export function normalizeAnalysisResponse(raw) {
  if (!raw || typeof raw !== "object") {
    return emptyAnalysis(
      "The backend returned an unreadable response."
    );
  }

  // -------------------------------------------------------------------
  // Multi-intent V2 response
  //
  // Backend shape:
  //
  // {
  //   status: "success",
  //   multi_intent: true,
  //   intent_count: 3,
  //   responses: [
  //     {
  //       intent_id: "intent_1",
  //       question: "...",
  //       status: "success",
  //       response: {
  //         request: {},
  //         result: {},
  //         insights: {},
  //         chart_spec: {},
  //         chart_data: []
  //       }
  //     }
  //   ]
  // }
  //
  // Normalize each inner response using the exact same single-intent
  // normalization logic used by the existing frontend.
  // -------------------------------------------------------------------
  if (
    raw.multi_intent === true &&
    Array.isArray(raw.responses)
  ) {
    const responses = raw.responses.map((item, index) => {
      const innerRaw =
        item?.response &&
        typeof item.response === "object"
          ? item.response
          : item;

      return {
        intentId:
          item?.intent_id ||
          `intent_${index + 1}`,

        question:
          typeof item?.question === "string"
            ? item.question
            : "",

        status:
          item?.status ||
          innerRaw?.status ||
          "success",

        analysis: normalizeSingleAnalysisResponse(innerRaw),
      };
    });

    return {
      status: raw.status || null,
      multiIntent: true,
      intentCount:
        typeof raw.intent_count === "number"
          ? raw.intent_count
          : responses.length,
      responses,
      // Keep top-level single-analysis fields null so existing
      // consumers don't accidentally treat this as one scalar result.
      answerText: null,
      scalar: null,
      grouped: null,
      nestedRanking: null,
      chart: null,
      forecast: null,
      insights: [],
      recommendations: [],
      errors: [],
      isEmpty: responses.length === 0,
    };
  }

  // -------------------------------------------------------------------
  // Existing single-intent response
  // -------------------------------------------------------------------
  return normalizeSingleAnalysisResponse(raw);
}


// ---------------------------------------------------------------------
// Single-intent normalization
//
// This is the old normalizeAnalysisResponse() behavior moved into a
// helper so both old and new backend contracts share exactly the same
// normalization rules.
// ---------------------------------------------------------------------

function normalizeSingleAnalysisResponse(raw) {
  if (!raw || typeof raw !== "object") {
    return emptyAnalysis(
      "The backend returned an unreadable response."
    );
  }

  const errors = Array.isArray(raw.errors)
    ? raw.errors.filter(Boolean)
    : [];

  const scalar = normalizeScalar(raw.result);

    const nestedRanking = normalizeNestedRanking(
    raw.result
  );
  const grouped = normalizeGrouped(raw.result);

  let chart = normalizeChart(raw);

  if (!chart && grouped && grouped.dimension === "period") {
    chart = {
      type: "line",
      title: grouped.operation
        ? `${titleCaseKey(grouped.operation)} by period`
        : "Trend",
      labels: grouped.rows.map((row) => row.key),
      datasets: [
        {
          label: grouped.operation
            ? titleCaseKey(grouped.operation)
            : "Value",
          data: grouped.rows.map((row) => row.value),
        },
      ],
    };
  }

  const forecast = normalizeForecast(raw.prediction);
  const insights = normalizeInsights(raw);
  const recommendations = normalizeRecommendations(raw);

  const answerText =
    typeof raw.answer === "string" && raw.answer.trim()
      ? raw.answer.trim()
      : typeof raw.explanation === "string" &&
          raw.explanation.trim()
        ? raw.explanation.trim()
        : typeof raw.explanation?.reasoning === "string" &&
            raw.explanation.reasoning.trim()
          ? raw.explanation.reasoning.trim()
          : insights.length
            ? insights[0]
            : null;

  const isEmpty =
    !scalar &&
    !grouped &&
    !chart &&
    !forecast &&
    !answerText &&
    insights.length === 0 &&
    recommendations.length === 0 &&
    errors.length === 0;

  return {
    status: raw.status || null,
    answerText,
    scalar,
    grouped,
    nestedRanking,
    chart,
    forecast,
    insights,
    recommendations,
    errors,
    isEmpty,
  };
}

function emptyAnalysis(message) {
  return {
    status: "error",
    answerText: null,
    scalar: null,
    grouped: null,
    nestedRanking: null,
    chart: null,
    forecast: null,
    insights: [],
    recommendations: [],
    errors: message ? [message] : [],
    isEmpty: true,
  };
}