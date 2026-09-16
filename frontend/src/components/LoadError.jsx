export default function LoadError({ message, onRetry }) {
  return (
    <section className="glass load-error" role="alert">
      <strong>Couldn't load the dataset profile</strong>
      <p>
        <code>{message || "Unknown error"}</code>
      </p>
      <button type="button" className="btn btn-primary" onClick={onRetry}>
        Retry
      </button>
    </section>
  );
}
