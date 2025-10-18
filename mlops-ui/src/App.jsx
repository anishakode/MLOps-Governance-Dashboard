import React, { useEffect, useMemo, useState } from "react";

/**
 * Minimal, single-file React UI for your MLOps Governance API
 * - Models (create/list/view)
 * - Promotions (request/approve/reject) with policy check
 * - Drift baseline & PSI evaluation
 * - Alerts (list/resolve)
 * - Audit (trail + Audit Pack save/download)
 * - Slack webhook runtime config
 * - Audit files & Audit packs browser
 *
 * Tailwind-friendly. Paste into a Vite + React project (src/App.jsx) and run.
 * Default API base: http://localhost:8000 (change in UI top bar)
 */

function classNames(...xs) {
  return xs.filter(Boolean).join(" ");
}

const Section = ({ title, children, right = null }) => (
  <div className="bg-white shadow rounded-2xl p-4 mb-5">
    <div className="flex items-center justify-between mb-3">
      <h2 className="text-lg font-semibold">{title}</h2>
      {right}
    </div>
    <div>{children}</div>
  </div>
);

const Input = (props) => (
  <input
    {...props}
    className={classNames(
      "border rounded-lg px-3 py-2 w-full",
      props.className || ""
    )}
  />
);

const Button = ({ children, variant = "primary", className = "", ...rest }) => {
  const base =
    "px-4 py-2 rounded-xl text-sm font-medium transition disabled:opacity-50 disabled:cursor-not-allowed";
  const theme =
    variant === "primary"
      ? "bg-black text-white hover:bg-gray-800"
      : variant === "ghost"
      ? "bg-transparent hover:bg-gray-100"
      : variant === "danger"
      ? "bg-red-600 text-white hover:bg-red-700"
      : "bg-gray-200 hover:bg-gray-300";
  return (
    <button className={classNames(base, theme, className)} {...rest}>
      {children}
    </button>
  );
};

const Tag = ({ children, color = "gray" }) => (
  <span
    className={classNames(
      "inline-block text-xs px-2 py-1 rounded-full border",
      color === "green" && "border-green-600 text-green-700 bg-green-50",
      color === "amber" && "border-amber-500 text-amber-700 bg-amber-50",
      color === "red" && "border-red-600 text-red-700 bg-red-50",
      color === "gray" && "border-gray-300 text-gray-700 bg-gray-50"
    )}
  >
    {children}
  </span>
);

function useApi(baseUrl) {
  const get = async (path) => {
    const r = await fetch(`${baseUrl}${path}`);
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  };
  const post = async (path, body) => {
    const r = await fetch(`${baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  };
  return { get, post };
}

function prettyError(err) {
  try {
    const j = JSON.parse(err.message);
    if (j?.detail) return typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
  } catch {}
  return err.message;
}

const TopBar = ({ baseUrl, setBaseUrl, api }) => {
  const [url, setUrl] = useState("");
  const [webhook, setWebhook] = useState("");
  const [saving, setSaving] = useState(false);
  const [info, setInfo] = useState("");

  useEffect(() => {
    setUrl(baseUrl);
  }, [baseUrl]);

  const fetchWebhook = async () => {
    try {
      const j = await api.get("/config/slack-webhook");
      setInfo(j.configured ? `Configured (${j.webhook_url_preview})` : "Not set");
    } catch (e) {
      setInfo("—");
    }
  };

  useEffect(() => {
    fetchWebhook();
  }, []);

  const saveWebhook = async () => {
    if (!webhook.startsWith("https://hooks.slack.com/")) {
      alert("Invalid Slack webhook URL");
      return;
    }
    setSaving(true);
    try {
      await api.post("/config/slack-webhook", { webhook_url: webhook });
      setWebhook("");
      await fetchWebhook();
    } catch (e) {
      alert(prettyError(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="bg-white shadow rounded-2xl p-4 mb-5">
      <div className="grid md:grid-cols-2 gap-3 items-end">
        <div>
          <label className="text-xs text-gray-500">API Base URL</label>
          <div className="flex gap-2">
            <Input value={url} onChange={(e) => setUrl(e.target.value)} />
            <Button onClick={() => setBaseUrl(url)}>Use</Button>
          </div>
          <p className="text-xs text-gray-500 mt-1">Default: http://localhost:8000</p>
        </div>
        <div>
          <label className="text-xs text-gray-500">Slack Webhook</label>
          <div className="flex gap-2">
            <Input placeholder="https://hooks.slack.com/services/..." value={webhook} onChange={(e) => setWebhook(e.target.value)} />
            <Button onClick={saveWebhook} disabled={saving}>Save</Button>
          </div>
          <p className="text-xs text-gray-500 mt-1">Current: {info}</p>
        </div>
      </div>
    </div>
  );
};

const Models = ({ api, onSelect, selectedId }) => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  const [form, setForm] = useState({ name: "", description: "", version: "1.0.0" });

  const load = async () => {
    setLoading(true);
    setErr("");
    try {
      const r = await api.get("/api/models");
      setModels(Array.isArray(r) ? r : [r]);
    } catch (e) {
      setErr(prettyError(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const create = async (e) => {
    e.preventDefault();
    try {
      await api.post("/api/models", form);
      setForm({ name: "", description: "", version: "1.0.0" });
      await load();
    } catch (e2) {
      alert(prettyError(e2));
    }
  };

  return (
    <Section
      title="Models"
      right={<Button variant="ghost" onClick={load}>Refresh</Button>}
    >
      <div className="grid md:grid-cols-3 gap-4">
        <div className="md:col-span-2">
          {loading ? (
            <p className="text-sm text-gray-500">Loading…</p>
          ) : err ? (
            <p className="text-sm text-red-600">{err}</p>
          ) : (
            <div className="grid gap-3">
              {models.map((m) => (
                <div
                  key={m.id}
                  className={classNames(
                    "border rounded-xl p-3 flex items-center justify-between",
                    selectedId === m.id ? "border-black" : "border-gray-200"
                  )}
                >
                  <div>
                    <div className="font-medium">{m.name} <span className="text-gray-400">v{m.version}</span></div>
                    <div className="text-xs text-gray-500">id {m.id} • stage {m.stage} • status {m.status}</div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Tag color={m.stage === "production" ? "green" : m.stage === "staging" ? "amber" : "gray"}>{m.stage}</Tag>
                    <Button variant="ghost" onClick={() => onSelect(m.id)}>Open</Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div>
          <form onSubmit={create} className="border rounded-xl p-3 space-y-2">
            <div className="font-medium mb-1">Create Model</div>
            <Input placeholder="name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
            <Input placeholder="description" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
            <Input placeholder="version" value={form.version} onChange={(e) => setForm({ ...form, version: e.target.value })} />
            <Button type="submit" className="w-full">Create</Button>
          </form>
        </div>
      </div>
    </Section>
  );
};

const ModelDetail = ({ api, id }) => {
  const [m, setM] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);

  const [prom, setProm] = useState({ target: "staging", justification: "", requested_by: "alice" });
  const [approvalId, setApprovalId] = useState("");
  const [decision, setDecision] = useState({ decided_by: "bob", decision: "approved", note: "" });

  const load = async () => {
    setLoading(true);
    setErr("");
    try {
      const r = await api.get(`/api/models/${id}`);
      setM(r);
    } catch (e) {
      setErr(prettyError(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [id]);

  const requestPromotion = async () => {
    try {
      const r = await api.post(`/api/models/${id}/promotions`, prom);
      setApprovalId(r.approval_id);
      await load();
    } catch (e) { alert(prettyError(e)); }
  };

  const sendDecision = async () => {
    if (!approvalId) { alert("Enter approval id (from request response)"); return; }
    try {
      await api.post(`/api/approvals/${approvalId}/decision`, decision);
      await load();
    } catch (e) { alert(prettyError(e)); }
  };

  const [policyTarget, setPolicyTarget] = useState("production");
  const [policy, setPolicy] = useState(null);
  const checkPolicy = async () => {
    try { setPolicy(await api.get(`/api/models/${id}/policy/check?target=${policyTarget}`)); }
    catch (e) { alert(prettyError(e)); }
  };

  if (loading) return <Section title="Model">Loading…</Section>;
  if (err) return <Section title="Model"><div className="text-red-600 text-sm">{err}</div></Section>;
  if (!m) return null;

  return (
    <Section title={`Model: ${m.name} (id ${m.id})`} right={<Button variant="ghost" onClick={load}>Refresh</Button>}>
      <div className="grid md:grid-cols-2 gap-4">
        <div className="space-y-3">
          <div className="border rounded-xl p-3">
            <div className="text-sm">Stage</div>
            <div className="flex items-center gap-2 mt-1">
              <Tag color={m.stage === "production" ? "green" : m.stage === "staging" ? "amber" : "gray"}>{m.stage}</Tag>
              <Tag>{m.status}</Tag>
            </div>
            <div className="text-xs text-gray-500 mt-1">created {m.created_at}</div>
          </div>

          <div className="border rounded-xl p-3 space-y-2">
            <div className="font-medium">Request Promotion</div>
            <div className="grid grid-cols-2 gap-2">
              <select className="border rounded-lg px-2 py-2" value={prom.target} onChange={(e)=>setProm({...prom, target:e.target.value})}>
                <option value="staging">staging</option>
                <option value="production">production</option>
              </select>
              <Input placeholder="requested_by" value={prom.requested_by} onChange={(e)=>setProm({...prom, requested_by:e.target.value})} />
            </div>
            <Input placeholder="justification" value={prom.justification} onChange={(e)=>setProm({...prom, justification:e.target.value})} />
            <Button onClick={requestPromotion}>Request</Button>
            {approvalId && <div className="text-xs text-gray-600">approval_id: <code>{approvalId}</code></div>}
          </div>

          <div className="border rounded-xl p-3 space-y-2">
            <div className="font-medium">Decision</div>
            <Input placeholder="approval_id" value={approvalId} onChange={(e)=>setApprovalId(e.target.value)} />
            <div className="grid grid-cols-2 gap-2">
              <Input placeholder="decided_by" value={decision.decided_by} onChange={(e)=>setDecision({...decision, decided_by:e.target.value})} />
              <select className="border rounded-lg px-2 py-2" value={decision.decision} onChange={(e)=>setDecision({...decision, decision:e.target.value})}>
                <option>approved</option>
                <option>rejected</option>
              </select>
            </div>
            <Input placeholder="note" value={decision.note} onChange={(e)=>setDecision({...decision, note:e.target.value})} />
            <Button onClick={sendDecision}>Submit Decision</Button>
          </div>
        </div>

        <div className="space-y-3">
          <div className="border rounded-xl p-3 space-y-2">
            <div className="font-medium">Policy Check</div>
            <div className="flex gap-2 items-center">
              <select className="border rounded-lg px-2 py-2" value={policyTarget} onChange={(e)=>setPolicyTarget(e.target.value)}>
                <option value="staging">staging</option>
                <option value="production">production</option>
              </select>
              <Button variant="ghost" onClick={checkPolicy}>Run</Button>
            </div>
            {policy && (
              <div className="text-xs mt-2">
                <div className="mb-1">pass: {String(policy.pass)}</div>
                <pre className="bg-gray-50 p-2 rounded-lg overflow-auto max-h-44">{JSON.stringify(policy, null, 2)}</pre>
              </div>
            )}
          </div>

          <AuditAndAlerts api={api} modelId={id} />
        </div>
      </div>
    </Section>
  );
};

const csvToFloats = (csv) => csv.split(/[ ,]+/).filter(Boolean).map((x) => parseFloat(x));

const Drift = ({ api, modelId }) => {
  const [feature, setFeature] = useState("age_bucket");
  const [expected, setExpected] = useState("0.10,0.20,0.30,0.25,0.15");
  const [actual, setActual] = useState("0.02,0.08,0.20,0.35,0.35");
  const [out, setOut] = useState(null);

  const setBaseline = async () => {
    try {
      const body = { feature, expected: csvToFloats(expected) };
      const r = await api.post(`/api/models/${modelId}/drift/baseline`, body);
      setOut({ baseline: r });
    } catch (e) { alert(prettyError(e)); }
  };

  const evalPsi = async () => {
    try {
      const body = { feature, actual: csvToFloats(actual) };
      const r = await api.post(`/api/models/${modelId}/drift/psi`, body);
      setOut({ psi: r });
    } catch (e) { alert(prettyError(e)); }
  };

  return (
    <Section title="Drift (PSI)">
      <div className="grid md:grid-cols-2 gap-3">
        <div className="space-y-2">
          <Input placeholder="feature" value={feature} onChange={(e)=>setFeature(e.target.value)} />
          <Input placeholder="expected (csv)" value={expected} onChange={(e)=>setExpected(e.target.value)} />
          <Button onClick={setBaseline}>Set Baseline</Button>
        </div>
        <div className="space-y-2">
          <Input placeholder="actual (csv)" value={actual} onChange={(e)=>setActual(e.target.value)} />
          <Button onClick={evalPsi}>Evaluate PSI</Button>
        </div>
      </div>
      {out && (
        <pre className="bg-gray-50 p-2 rounded-lg overflow-auto mt-3 text-xs">{JSON.stringify(out, null, 2)}</pre>
      )}
    </Section>
  );
};

const AuditAndAlerts = ({ api, modelId }) => {
  const [alerts, setAlerts] = useState([]);
  const [audit, setAudit] = useState([]);
  const [packResp, setPackResp] = useState(null);

  const loadAlerts = async () => {
    try { setAlerts(await api.get(`/api/alerts?model_id=${modelId}`)); }
    catch (e) { alert(prettyError(e)); }
  };
  const loadAudit = async () => {
    try { setAudit(await api.get(`/api/models/${modelId}/audit`)); }
    catch (e) { alert(prettyError(e)); }
  };
  const genPack = async () => {
    try { setPackResp(await api.get(`/api/models/${modelId}/audit-pack`)); }
    catch (e) { alert(prettyError(e)); }
  };
  const resolve = async (id) => {
    try { await api.post(`/api/alerts/${id}/resolve`); await loadAlerts(); }
    catch (e) { alert(prettyError(e)); }
  };

  useEffect(() => { loadAlerts(); loadAudit(); }, [modelId]);

  return (
    <div className="space-y-3">
      <Section title="Alerts" right={<Button variant="ghost" onClick={loadAlerts}>Refresh</Button>}>
        <div className="grid gap-2">
          {alerts.length === 0 && <div className="text-sm text-gray-500">No alerts</div>}
          {alerts.map((a) => (
            <div key={a.id} className="border rounded-lg p-3 flex items-center justify-between">
              <div>
                <div className="font-medium text-sm">{a.message}</div>
                <div className="text-xs text-gray-500">severity {a.severity} • status {a.status} • id {a.id}</div>
              </div>
              {a.status === "open" && (
                <Button variant="danger" onClick={() => resolve(a.id)}>Resolve</Button>
              )}
            </div>
          ))}
        </div>
      </Section>

      <Section title="Audit Trail" right={<Button variant="ghost" onClick={loadAudit}>Refresh</Button>}>
        <pre className="bg-gray-50 p-2 rounded-lg overflow-auto max-h-72 text-xs">{JSON.stringify(audit, null, 2)}</pre>
      </Section>

      <Section title="Audit Pack">
        <div className="flex items-center gap-2">
          <Button onClick={genPack}>Generate & Save</Button>
          {packResp?._saved_to && (
            <a className="text-sm underline" href="#" onClick={(e)=>e.preventDefault()}>Saved to: {packResp._saved_to}</a>
          )}
        </div>
      </Section>
    </div>
  );
};

const FilesBrowser = ({ api, title, listPath, downloadPathBase }) => {
  const [files, setFiles] = useState([]);
  const load = async () => {
    try { setFiles(await api.get(listPath)); } catch (e) { alert(prettyError(e)); }
  };
  useEffect(() => { load(); }, []);
  return (
    <Section title={title} right={<Button variant="ghost" onClick={load}>Refresh</Button>}>
      <div className="grid gap-2">
        {files.length === 0 && <div className="text-sm text-gray-500">No files</div>}
        {files.map((f) => (
          <div key={f.name} className="border rounded-lg p-3 flex items-center justify-between">
            <div>
              <div className="text-sm font-medium">{f.name}</div>
              <div className="text-xs text-gray-500">{(f.size_bytes/1024).toFixed(1)} KB • {f.modified}</div>
            </div>
            <a className="text-sm underline" href={`${downloadPathBase}/${encodeURIComponent(f.name)}`} target="_blank" rel="noreferrer">Download</a>
          </div>
        ))}
      </div>
    </Section>
  );
};

export default function App() {
  const [baseUrl, setBaseUrl] = useState("http://localhost:8000");
  const api = useMemo(() => useApi(baseUrl), [baseUrl]);

  const [selectedId, setSelectedId] = useState(null);
  const [tab, setTab] = useState("models");

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-6xl mx-auto px-4 py-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold">MLOps Governance Dashboard</h1>
          <div className="flex gap-1">
            {["models","drift","files"].map((t) => (
              <Button key={t} variant={tab===t?"primary":"ghost"} onClick={()=>setTab(t)}>{t}</Button>
            ))}
          </div>
        </div>

        <TopBar baseUrl={baseUrl} setBaseUrl={setBaseUrl} api={api} />

        {tab === "models" && (
          <>
            <Models api={api} onSelect={(id)=>{ setSelectedId(id); }} selectedId={selectedId} />
            {selectedId && (
              <>
                <ModelDetail api={api} id={selectedId} />
                <Drift api={api} modelId={selectedId} />
              </>
            )}
          </>
        )}

        {tab === "drift" && selectedId && (
          <>
            <ModelDetail api={api} id={selectedId} />
            <Drift api={api} modelId={selectedId} />
          </>
        )}

        {tab === "files" && (
          <div className="grid md:grid-cols-2 gap-4">
            <FilesBrowser api={api} title="Audit Files (.jsonl)" listPath="/api/audit-files" downloadPathBase={`${baseUrl}/api/audit-files`} />
            <FilesBrowser api={api} title="Audit Packs (.json)" listPath="/api/audit-packs" downloadPathBase={`${baseUrl}/api/audit-packs`} />
          </div>
        )}

        <div className="text-xs text-gray-400 mt-8">Tip: select a model in the Models tab to unlock promotion, policy check, alerts, audit, and drift actions.</div>
      </div>
    </div>
  );
}
