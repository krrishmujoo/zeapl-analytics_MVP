import ChartCanvas from "./ChartCanvas";
import { formatNumber } from "../lib/numeric";

function QualityBadge({ quality }) {
  if (!quality?.level) return null;
  const level = quality.level.toLowerCase();
  const label = quality.level[0].toUpperCase() + quality.level.slice(1);
  return <span className={`badge badge-quality-${level}`}>Quality: {label}</span>;
}

function ForecastPanel({ forecast }) {
  const hasPeriods = Array.isArray(forecast.periods) && forecast.periods.length > 0;
  const chartLabels = hasPeriods ? forecast.periods.map((p) => String(p.period)) : [];
  const chartDatasets = hasPeriods
    ? [{ label: forecast.metricLabel, data: forecast.periods.map((p) => p.predicted) }]
    : [];
  const hasValidation =
    forecast.validation &&
    (forecast.validation.mae !== null || forecast.validation.rmse !== null || forecast.validation.r2 !== null);

  return (
    <section className="glass forecast-panel">
      <div className="panel-head">
        <div>
          <span className="panel-eyebrow">Forecast</span>
          <h3 className="forecast-title">{forecast.metricLabel}</h3>
          <p className="forecast-subtitle">
            {[
              forecast.granularity ? `${forecast.granularity} granularity` : null,
              forecast.horizon ? `${forecast.horizon} periods ahead` : null,
            ]
              .filter(Boolean)
              .join(" · ")}
          </p>
        </div>
        <div className="quality-block">
          <QualityBadge quality={forecast.quality} />
          {typeof forecast.quality?.score === "number" && (
            <span className="quality-score">Score {forecast.quality.score.toFixed(2)}</span>
          )}
          <span className="quality-safe">{forecast.quality?.safeToActOn ? "Safe to act on" : "Use with caution"}</span>
        </div>
      </div>

      {hasPeriods ? (
        <>
          <div className="forecast-chart-wrap">
            <ChartCanvas type="line" labels={chartLabels} datasets={chartDatasets} />
          </div>

          {(forecast.total !== null || forecast.average !== null) && (
            <div className="forecast-totals">
              {forecast.total !== null && (
                <div className="forecast-total-card">
                  <span className="k">Total</span>
                  <span className="v">{formatNumber(forecast.total)}</span>
                </div>
              )}
              {forecast.average !== null && (
                <div className="forecast-total-card">
                  <span className="k">Average</span>
                  <span className="v">{formatNumber(forecast.average)}</span>
                </div>
              )}
            </div>
          )}

          <div className="table-scroll">
            <table className="data-table" aria-label="Forecast periods">
              <thead>
                <tr>
                  <th>Period</th>
                  <th>Predicted</th>
                  <th>Lower</th>
                  <th>Upper</th>
                </tr>
              </thead>
              <tbody>
                {forecast.periods.map((p, i) => (
                  <tr key={`${p.period}-${i}`}>
                    <td>{p.period}</td>
                    <td>{p.predicted === null ? "—" : formatNumber(p.predicted)}</td>
                    <td>{p.lower === null ? "—" : formatNumber(p.lower)}</td>
                    <td>{p.upper === null ? "—" : formatNumber(p.upper)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <p className="forecast-empty">No forecast periods were returned.</p>
      )}

      {(forecast.algorithm || hasValidation) && (
        <div className="forecast-section">
          <div className="model-grid">
            {forecast.algorithm && (
              <div>
                <span className="k">Algorithm</span>
                <span>{forecast.algorithm}</span>
              </div>
            )}
            {forecast.validation?.mae !== null && forecast.validation?.mae !== undefined && (
              <div>
                <span className="k">MAE</span>
                <span>{formatNumber(forecast.validation.mae)}</span>
              </div>
            )}
            {forecast.validation?.rmse !== null && forecast.validation?.rmse !== undefined && (
              <div>
                <span className="k">RMSE</span>
                <span>{formatNumber(forecast.validation.rmse)}</span>
              </div>
            )}
            {forecast.validation?.r2 !== null && forecast.validation?.r2 !== undefined && (
              <div>
                <span className="k">R&sup2;</span>
                <span>{formatNumber(forecast.validation.r2)}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {forecast.insightMessages.length > 0 && (
        <ul className="signal-bullets">
          {forecast.insightMessages.map((m, i) => (
            <li key={i}>{m}</li>
          ))}
        </ul>
      )}

      {forecast.warnings.length > 0 && (
        <ul className="warning-list">
          {forecast.warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}
    </section>
  );
}

export default function AnalysisWorkspace({ analysis }) {
  if (!analysis) return null;

  // ---------------------------------------------------------------
  // Multi-intent response
  //
  // The backend can now return:
  //
  // {
  //   multi_intent: true,
  //   responses: [
  //     {
  //       intentId,
  //       question,
  //       analysis: { ...normalized single response... }
  //     }
  //   ]
  // }
  //
  // Render every normalized response using the same components
  // already used for normal single-intent analysis.
  // ---------------------------------------------------------------
  if (
    analysis.multiIntent === true &&
    Array.isArray(analysis.responses)
  ) {
    if (analysis.responses.length === 0) {
      return (
        <section className="glass empty-state">
          <p>
            The backend returned no analytical results for this request.
          </p>
        </section>
      );
    }

    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 20,
        }}
      >
        {analysis.responses.map((item, index) => (
          <section
            key={item.intentId || `intent-${index + 1}`}
            className="analysis-intent"
          >
            {item.question && (
              <div className="intent-heading">
                <span className="panel-eyebrow">
                  Analysis {index + 1}
                </span>
                <h3 className="chart-title">
                  {item.question}
                </h3>
              </div>
            )}

            <SingleAnalysis
              analysis={item.analysis}
            />
          </section>
        ))}
      </div>
    );
  }

  // ---------------------------------------------------------------
  // Existing single-intent response
  // ---------------------------------------------------------------
  return <SingleAnalysis analysis={analysis} />;
}


// ------------------------------------------------------------------
// Existing single-analysis renderer
//
// This keeps the original rendering behavior intact.
// ------------------------------------------------------------------

function SingleAnalysis({ analysis }) {
  if (!analysis) return null;

  if (analysis.isEmpty) {
    return (
      <section className="glass empty-state">
        <p>
          The backend returned nothing to show for this question.
          Try rephrasing, or ask for an overview of the dataset.
        </p>
      </section>
    );
  }

  const {
    answerText,
    scalar,
    grouped,
    nestedRanking,
    chart,
    forecast,
    insights,
    recommendations,
    errors,
  } = analysis;

  const hasSignals =
    insights.length > 0 ||
    recommendations.length > 0 ||
    errors.length > 0;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 20,
      }}
    >
      {answerText && (
        <section className="glass answer-summary">
          <p>{answerText}</p>
        </section>
      )}

      {scalar && (
  <section>
    <div className="scalar-grid">
      {scalar.entries.map((entry) => (
        <div key={entry.key} className="glass scalar-card">
          {scalar.operationLabel && (
            <span className="scalar-op">
              {scalar.operationLabel}
            </span>
          )}

          <span className="scalar-metric">
            {entry.label}
          </span>

          {entry.type === "user-ranking" ? (
            <div className="ranking-list">
              {entry.value.map((item, index) => {
                const numericValue = Number(item.value);

                return (
                  <div
                    key={item.dimension || index}
                    className="ranking-row"
                  >
                    <span className="ranking-position">
                      {index + 1}.
                    </span>

                    <strong className="ranking-user">
                      {item.dimension || "Unknown"}
                    </strong>

                    <span className="ranking-value">
                      {Number.isFinite(numericValue)
                        ? numericValue.toLocaleString("en-IN")
                        : "—"}
                    </span>
                  </div>
                );
              })}
            </div>
          ) : (
            <span className="scalar-value">
              {entry.formatted}
            </span>
          )}
        </div>
      ))}
    </div>

          {typeof scalar.recordCount === "number" && (
            <p className="scalar-record-note">
              Based on{" "}
              {scalar.recordCount.toLocaleString("en-US")} records.
            </p>
          )}
        </section>
      )}
      {nestedRanking && (
        <section className="glass nested-ranking-panel">
          <div className="nested-ranking-head">
            <div>
              <span className="panel-eyebrow">
                Nested ranking
              </span>

              <h3 className="chart-title">
                {nestedRanking.outerDimensionLabel}
                {" → "}
                {nestedRanking.innerDimensionLabel}
                {" by "}
                {nestedRanking.metricLabel}
              </h3>
            </div>

            <span className="nested-ranking-direction">
              {nestedRanking.direction === "bottom"
                ? "Bottom"
                : "Top"}
              {" "}
              {nestedRanking.outerTopN}
              {" × "}
              {nestedRanking.innerTopN}
            </span>
          </div>

          <div className="nested-ranking-list">
            {nestedRanking.rows.map((outerRow) => (
              <details
                key={`${outerRow.rank}-${outerRow.entity}`}
                className="nested-ranking-group"
                open={outerRow.rank === 1}
              >
                <summary className="nested-ranking-summary">
                  <span className="ranking-position">
                    {outerRow.rank}.
                  </span>

                  <strong className="nested-ranking-entity">
                    {outerRow.entity}
                  </strong>

                  <span className="nested-ranking-value">
                    {outerRow.formatted}
                  </span>
                </summary>

                <div className="nested-ranking-children">
                  <div className="nested-ranking-columns">
                    <span>
                      {nestedRanking.innerDimensionLabel}
                    </span>
                    <span>
                      {nestedRanking.metricLabel}
                    </span>
                  </div>

                  {outerRow.children.map((child) => (
                    <div
                      key={`${outerRow.entity}-${child.rank}-${child.entity}`}
                      className="nested-ranking-child"
                    >
                      <span className="ranking-position">
                        {child.rank}.
                      </span>

                      <strong className="ranking-user">
                        {child.entity}
                      </strong>

                      <span className="ranking-value">
                        {child.formatted}
                      </span>
                    </div>
                  ))}

                  {outerRow.children.length === 0 && (
                    <p className="empty-state">
                      No inner ranking rows were returned.
                    </p>
                  )}
                </div>
              </details>
            ))}
          </div>
        </section>
      )}

      {grouped && (
        <section className="glass table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>{grouped.dimensionLabel}</th>
                <th>
                  {grouped.ranking
                    ? grouped.metricLabel
                    : "Value"}
                </th>
              </tr>
            </thead>

            <tbody>
              {grouped.rows.map((row, index) => (
                <tr key={`${row.dimension || row.key}-${index}`}>
                  <td>
                    {grouped.ranking
                      ? row.dimension
                      : row.key}
                  </td>
                  <td>
                    {row.formatted}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {chart && (
        <section className="glass chart-panel">
          <div className="panel-head">
            <div>
              <span className="panel-eyebrow">
                Chart
              </span>

              <h3 className="chart-title">
                {chart.title || "Result chart"}
              </h3>
            </div>
          </div>

          <div className="chart-wrap">
            <ChartCanvas
              type={chart.type}
              labels={chart.labels}
              datasets={chart.datasets}
            />
          </div>
        </section>
      )}

      {forecast && (
        <ForecastPanel forecast={forecast} />
      )}

      {hasSignals && (
        <section className="glass signals-panel">
          <h3 className="signals-title">
            Signals
          </h3>

          <div
            className="signals-list"
            style={{ marginTop: 14 }}
          >
            {insights.map((text, i) => (
              <div
                className="signal-card"
                data-kind="insight"
                key={`insight-${i}`}
              >
                <span className="signal-kicker">
                  Insight
                </span>

                <div className="signal-body">
                  {text}
                </div>
              </div>
            ))}

            {recommendations.map((rec, i) => {
              const text =
                typeof rec === "string"
                  ? rec
                  : rec.text;

              const limitations =
                typeof rec === "object"
                  ? rec.limitations
                  : null;

              return (
                <div
                  className="signal-card recommendation-card"
                  data-kind="ai"
                  key={`rec-${i}`}
                >
                  <span className="signal-kicker">
                    Recommendation
                  </span>

                  <p>{text}</p>

                  {limitations && (
                    <p className="recommendation-limitations">
                      {limitations}
                    </p>
                  )}
                </div>
              );
            })}

            {errors.map((err, i) => (
              <div
                className="signal-card"
                data-kind="error"
                key={`err-${i}`}
              >
                <span className="signal-kicker">
                  Error
                </span>

                <div className="signal-body">
                  {typeof err === "string"
                    ? err
                    : JSON.stringify(err)}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}