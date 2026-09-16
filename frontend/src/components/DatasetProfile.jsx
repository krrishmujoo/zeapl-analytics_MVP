export default function DatasetProfile({ profile, loading, error, onRefresh }) {
  return (
    <section className="glass profile-panel" aria-label="Dataset profile">
      <div>
        <span className="panel-eyebrow">Dataset</span>
        <h2 className="profile-title">
          {loading ? (
            <span className="skeleton-text">Loading dataset&hellip;</span>
          ) : (
            profile?.datasetId || "Untitled dataset"
          )}
        </h2>
      </div>

      {loading && <p className="profile-loading">Reading dataset profile&hellip;</p>}

      {!loading && error && !profile && <p className="profile-error">{error}</p>}

      {!loading && profile && (
        <>
          <div className="profile-facts">
            <div className="profile-fact">
              <span className="k">Rows</span>
              <span className="v">
                {profile.rowCount != null ? profile.rowCount.toLocaleString("en-US") : "—"}
              </span>
            </div>
            <div className="profile-fact">
              <span className="k">Date column</span>
              <span className="v">{profile.dateColumn || "—"}</span>
            </div>
            {profile.timeRange && (profile.timeRange.start || profile.timeRange.end) && (
              <div className="profile-fact">
                <span className="k">Range</span>
                <span className="v">
                  {profile.timeRange.start || "?"} &rarr; {profile.timeRange.end || "?"}
                </span>
              </div>
            )}
          </div>

          <div className="profile-section">
            <p className="profile-section-title">
              Metrics <span className="count-pill">{profile.metrics.length}</span>
            </p>
            {profile.metrics.length ? (
              <div className="profile-metric-list">
                {profile.metrics.map((m) => (
                  <div className="profile-metric-row" key={m.key}>
                    <span className="profile-metric-name">{m.displayName}</span>
                    <span className="profile-metric-meta">
                      {m.semanticType || ""}
                      {m.forecastable && <span className="badge badge-forecastable">Forecastable</span>}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="profile-empty">No metrics detected.</p>
            )}
          </div>

          {profile.dimensions.length > 0 && (
            <div className="profile-section">
              <p className="profile-section-title">
                Dimensions <span className="count-pill">{profile.dimensions.length}</span>
              </p>
              <div className="chip-row">
                {profile.dimensions.map((d) => (
                  <span key={d} className="chip chip-static">
                    {d}
                  </span>
                ))}
              </div>
            </div>
          )}

          {profile.warnings.length > 0 && (
            <div className="profile-section">
              <p className="profile-section-title">Warnings</p>
              <ul className="profile-warning-list">
                {profile.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          )}

          <button type="button" className="btn btn-ghost btn-small" onClick={onRefresh}>
            Refresh
          </button>
        </>
      )}
    </section>
  );
}
