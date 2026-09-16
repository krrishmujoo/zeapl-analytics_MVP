import { Component } from "react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // This is the only trace of a render crash once the fallback UI
    // replaces the broken tree, so it stays even outside development.
    console.error("Ledger UI crashed:", error, info?.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="app">
          <section className="glass load-error" role="alert">
            <strong>Something went wrong rendering the dashboard</strong>
            <p>{this.state.error.message || String(this.state.error)}</p>
            <button type="button" className="btn btn-primary" onClick={() => this.setState({ error: null })}>
              Try again
            </button>
          </section>
        </div>
      );
    }
    return this.props.children;
  }
}
