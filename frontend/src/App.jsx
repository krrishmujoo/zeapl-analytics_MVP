import React, { useCallback, useEffect, useRef, useState } from "react";
import BackgroundField from "./components/BackgroundField";
import TopBar from "./components/TopBar";
import DatasetProfile from "./components/DatasetProfile";
import QueryComposer, {
  clearHistory,
  loadHistory,
  pushHistory,
} from "./components/QueryComposer";
import AnalysisWorkspace from "./components/AnalysisWorkspace";
import LoadError from "./components/LoadError";
import { getDatasetProfile, analyzeDataset } from "./api";
import { normalizeProfileResponse, normalizeAnalysisResponse } from "./lib/adapters";
import { buildSuggestions } from "./lib/promptSuggestions";


export default function App() {
  const [status, setStatus] = useState("connecting");

  const [profile, setProfile] = useState(null);
  const [profileLoading, setProfileLoading] = useState(true);
  const [profileError, setProfileError] = useState(null);

  const [analysis, setAnalysis] = useState(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);

  const [history, setHistory] = useState(() => loadHistory());
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const abortRef = useRef(null);

  const [conversationId, setConversationId] = useState(() => {
    const key = "ledger-os:conversation-id";

    try {
      const existing = window.sessionStorage?.getItem(key);

      if (existing) {
        return existing;
      }

      const id = crypto.randomUUID();
      window.sessionStorage?.setItem(key, id);
      return id;
    } catch {
      return crypto.randomUUID();
    }
  });


  const loadProfile = useCallback(async () => {
    setProfileLoading(true);
    setProfileError(null);
    setStatus("connecting");
    try {
      const raw = await getDatasetProfile();
      setProfile(normalizeProfileResponse(raw));
      setStatus("live");
    } catch (err) {
      setProfileError(err.message || String(err));
      setStatus("error");
    } finally {
      setProfileLoading(false);
    }
  }, []);

  useEffect(() => {
    loadProfile();
  }, [loadProfile]);

  async function handleAsk(question, options) {
    // Cancel any in-flight request before starting a new one.
    if (abortRef.current) {
      abortRef.current.abort();
    }

    const controller = new AbortController();
    abortRef.current = controller;

    setAnalysisLoading(true);
    setAnalysisError(null);
    setStatus("thinking");

    try {
      const raw = await analyzeDataset(
        question,
        {
          ...options,
          conversationId,
        },
        { signal: controller.signal }
      );

      setAnalysis(normalizeAnalysisResponse(raw));
      setHistory(pushHistory(question));
      setStatus("live");
    } catch (err) {
      if (err.name === "AbortError") return;

      setAnalysisError(err.message || String(err));
      setStatus("live");
    } finally {
      if (abortRef.current === controller) {
        setAnalysisLoading(false);
        abortRef.current = null;
      }
    }
  }

  function handleCancel() {
    abortRef.current?.abort();
    abortRef.current = null;
    setAnalysisLoading(false);
    setStatus("live");
  }

  function handleNewConversation() {
    abortRef.current?.abort();
    abortRef.current = null;

    const nextConversationId = crypto.randomUUID();

    try {
      window.sessionStorage?.setItem(
        "ledger-os:conversation-id",
        nextConversationId,
      );
    } catch {
      // The in-memory ID still changes if storage is unavailable.
    }

    setConversationId(nextConversationId);
    setAnalysis(null);
    setAnalysisError(null);
    setAnalysisLoading(false);
    setHistory(clearHistory());
    setStatus("live");
  }

  const suggestions = profile ? buildSuggestions(profile) : [];

  return (
    <>
      <BackgroundField />
      <div className="app">
       <TopBar
          status={status}
          datasetId={profile?.datasetId}
          onNewConversation={handleNewConversation}
        />
        <div className="scanline" aria-hidden="true" />

        {status === "error" && profileError && !profile ? (
          <LoadError message={profileError} onRetry={loadProfile} />
        ) : (
          <div className="workspace-layout">
            <button
              className="btn btn-ghost sidebar-toggle"
              onClick={() => setSidebarOpen((v) => !v)}
              aria-expanded={sidebarOpen}
              aria-controls="profile-sidebar"
            >
              {sidebarOpen ? "Hide dataset panel" : "Show dataset panel"}
            </button>

            <div id="profile-sidebar" className={`sidebar-column${sidebarOpen ? " is-open" : ""}`}>
              <DatasetProfile
                profile={profile}
                loading={profileLoading}
                error={profileError}
                onRefresh={loadProfile}
              />
            </div>

            <div className="main-column">
              <QueryComposer
                suggestions={suggestions}
                history={history}
                isLoading={analysisLoading}
                onSubmit={handleAsk}
                onCancel={handleCancel}
                forecastableMetrics={profile?.forecastableMetrics || []}
              />

              <div aria-live="polite" className="sr-only">
                {analysisLoading ? "Loading answer…" : ""}
              </div>

              {analysisError && (
                <section className="glass error-state" role="alert">
                  <p>{analysisError}</p>
                </section>
              )}

              <AnalysisWorkspace analysis={analysis} />
            </div>
          </div>
        )}

        <footer className="foot">
          <span>Profile-driven dynamic analysis &middot; Enter to ask, Shift+Enter for a new line</span>
        </footer>
      </div>
    </>
  );
}
