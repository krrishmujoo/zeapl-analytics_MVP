// Builds a small set of suggested questions purely from the dataset
// profile the backend reports (metrics + dimensions). Nothing here is
// hard-coded to a particular dataset -- swap the backend's dataset and the
// suggestions change with it.

function titleCase(str) {
  return String(str)
    .replace(/[_-]+/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(" ");
}

export function buildSuggestions(profile) {
  if (!profile || !Array.isArray(profile.metrics) || profile.metrics.length === 0) return [];

  const firstDimension = Array.isArray(profile.dimensions) ? profile.dimensions[0] : null;
  const suggestions = [];

  for (const metric of profile.metrics) {
    const agg = metric.defaultAggregation || "total";
    const aggLabel = agg === "sum" ? "Total" : agg[0].toUpperCase() + agg.slice(1);
    const aggWord = agg === "sum" ? "total" : agg;

    suggestions.push({
      label: `${aggLabel} ${metric.displayName}`,
      question: `What is the ${aggWord} ${metric.displayName.toLowerCase()}?`,
    });

    if (firstDimension) {
      suggestions.push({
        label: `${metric.displayName} by ${titleCase(firstDimension)}`,
        question: `Show ${metric.displayName.toLowerCase()} by ${firstDimension}`,
      });
    }

    if (metric.forecastable) {
      suggestions.push({
        label: `Forecast ${metric.displayName}`,
        question: `Forecast ${metric.displayName.toLowerCase()} for the next 4 weeks`,
      });
    }
  }

  const seen = new Set();
  const unique = [];
  for (const s of suggestions) {
    if (seen.has(s.label)) continue;
    seen.add(s.label);
    unique.push(s);
  }
  return unique.slice(0, 8);
}
