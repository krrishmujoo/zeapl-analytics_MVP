const STATUS_LABEL = {
  connecting: "Connecting",
  live: "Live",
  thinking: "Thinking",
  error: "Offline",
};

export default function TopBar({
  status = "connecting",
  datasetId,
  onNewConversation,
}) {
  const label = STATUS_LABEL[status] || status;

  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true">
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
          >
            <path
              d="M4 19h16M7 19V9m5 10V5m5 14v-7"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>

        <div className="brand-text">
          <span className="brand-name">
            Ledger <span className="brand-accent">//</span>{" "}
            Revenue Intelligence
          </span>

          <span className="brand-eyebrow">
            {datasetId
              ? `Dataset: ${datasetId}`
              : "Profile-driven dynamic analysis"}
          </span>
        </div>
      </div>

      <div className="topbar-actions">
        <button
          type="button"
          className="btn btn-ghost"
          onClick={onNewConversation}
          disabled={
            status === "connecting" ||
            typeof onNewConversation !== "function"
          }
        >
          New conversation
        </button>

        <div className="status-pill" data-state={status}>
          <span className="status-dot" aria-hidden="true" />
          {label}
        </div>
      </div>
    </header>
  );
}