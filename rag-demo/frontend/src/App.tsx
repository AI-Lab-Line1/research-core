import { useEffect, useState } from "react";
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Boxes,
  Check,
  ChevronRight,
  CircleHelp,
  Database,
  Download,
  FileText,
  GitBranch,
  Layers3,
  ListFilter,
  LoaderCircle,
  MessageSquareText,
  Network,
  RefreshCw,
  Scale,
  Search,
  Scissors,
  Sparkles,
  TerminalSquare,
  Timer,
  TriangleAlert,
  X,
} from "lucide-react";
import {
  api,
  Chunk,
  ComparisonConfig,
  ComparisonResponse,
  IndexStatus,
  MethodOption,
  MethodStatus,
  PipelineConfig,
  PipelineStep,
  QueryResponse,
  RetrievalHit,
  RuntimeStatus,
  SourceDocument,
} from "./lib/api";

type View = "ask" | "source" | "chunks" | "index" | "retrieval" | "context" | "compare" | "trace";
type ComparisonMode = "retrieval" | "chunking" | "generation";

const DEFAULT_CONFIG: PipelineConfig = {
  chunking_method: "structure",
  retrieval_method: "tfidf",
  reranking_method: "none",
  generation_method: "extractive",
  chunk_size: 260,
  chunk_overlap: 40,
  semantic_threshold: 0.05,
  semantic_max_chars: 620,
};

const navItems: { id: View; label: string; icon: typeof MessageSquareText }[] = [
  { id: "ask", label: "问答工作台", icon: MessageSquareText },
  { id: "source", label: "知识库原文", icon: BookOpen },
  { id: "chunks", label: "文本切分", icon: Scissors },
  { id: "index", label: "表示与索引", icon: Database },
  { id: "retrieval", label: "检索与重排", icon: Search },
  { id: "context", label: "上下文", icon: Layers3 },
  { id: "compare", label: "方法对比", icon: Scale },
  { id: "trace", label: "完整 Trace", icon: Activity },
];

const pipelineStages = [
  { id: "load", label: "加载", icon: FileText },
  { id: "chunk", label: "切分", icon: Scissors },
  { id: "index", label: "索引", icon: Boxes },
  { id: "retrieve", label: "召回", icon: Search },
  { id: "rerank", label: "重排", icon: ListFilter },
  { id: "context", label: "上下文", icon: Layers3 },
  { id: "generate", label: "回答", icon: Sparkles },
];

const examples = [
  "创新学社有哪些勋章？",
  "加入学社需要经过哪些流程？",
  "图书馆什么时候开放，可以借多少本书？",
];

function App() {
  const [view, setView] = useState<View>("ask");
  const [document, setDocument] = useState<SourceDocument | null>(null);
  const [methods, setMethods] = useState<MethodOption[]>([]);
  const [status, setStatus] = useState<IndexStatus | null>(null);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [buildTrace, setBuildTrace] = useState<PipelineStep[]>([]);
  const [config, setConfig] = useState<PipelineConfig>(DEFAULT_CONFIG);
  const [question, setQuestion] = useState("");
  const [comparisonQuestion, setComparisonQuestion] = useState(examples[2]);
  const [comparisonMode, setComparisonMode] = useState<ComparisonMode>("retrieval");
  const [topK, setTopK] = useState(4);
  const [loading, setLoading] = useState(true);
  const [asking, setAsking] = useState(false);
  const [building, setBuilding] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const chunkingMethod = findMethod(methods, config.chunking_method);
  const retrievalMethod = findMethod(methods, config.retrieval_method);
  const rerankingMethod = findMethod(methods, config.reranking_method);
  const generationMethod = findMethod(methods, config.generation_method);
  const activeTrace = result?.trace || buildTrace;
  const currentTitle = navItems.find((item) => item.id === view)?.label || "问答工作台";
  const configApplied = Boolean(
    status
      && status.chunking_method === config.chunking_method
      && status.retrieval_method === config.retrieval_method
      && (config.chunking_method !== "fixed_length"
        || (status.chunk_size === config.chunk_size && status.chunk_overlap === config.chunk_overlap))
      && (config.chunking_method !== "semantic"
        || (status.semantic_threshold === config.semantic_threshold
          && status.semantic_max_chars === config.semantic_max_chars)),
  );

  useEffect(() => {
    Promise.all([api.document(), api.methods(), api.runtime(), api.build(DEFAULT_CONFIG)])
      .then(([documentData, methodData, runtimeData, built]) => {
        setDocument(documentData);
        setMethods(methodData);
        setRuntime(runtimeData);
        setStatus(built.status);
        setChunks(built.chunks);
        setBuildTrace(built.trace);
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "后端服务未连接"))
      .finally(() => setLoading(false));
  }, []);

  function updateConfig(patch: Partial<PipelineConfig>) {
    setConfig((current) => ({ ...current, ...patch }));
  }

  async function handleBuild(targetView: View = "index") {
    setBuilding(true);
    setError(null);
    try {
      const built = await api.build(config);
      setStatus(built.status);
      setChunks(built.chunks);
      setBuildTrace(built.trace);
      setResult(null);
      setView(targetView);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "索引构建失败");
    } finally {
      setBuilding(false);
    }
  }

  async function handleAsk(event: React.FormEvent) {
    event.preventDefault();
    if (!question.trim()) return;
    setAsking(true);
    setError(null);
    try {
      const response = await api.query(question.trim(), topK, config);
      const activeChunks = await api.chunks();
      setResult(response);
      setStatus(response.index_status);
      setChunks(activeChunks);
      setView("retrieval");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "问答失败");
    } finally {
      setAsking(false);
    }
  }

  async function handleCompare(event: React.FormEvent) {
    event.preventDefault();
    if (!comparisonQuestion.trim()) return;
    const category = comparisonMode === "retrieval" ? "retrieval" : comparisonMode === "chunking" ? "chunking" : "generation";
    const variants = methods.filter((method) => method.category === category && method.status === "available");
    const configs: ComparisonConfig[] = variants.map((method) => ({
      ...config,
      label: method.name,
      ...(comparisonMode === "retrieval"
        ? { retrieval_method: method.id }
        : comparisonMode === "chunking"
          ? { chunking_method: method.id }
          : { generation_method: method.id }),
    }));
    if (configs.length < 2) {
      setError("当前可用方法不足两个，暂时无法形成对比");
      return;
    }

    setComparing(true);
    setError(null);
    try {
      setComparison(await api.compare(comparisonQuestion.trim(), topK, configs));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "方法对比失败");
    } finally {
      setComparing(false);
    }
  }

  function handleComparisonMode(mode: ComparisonMode) {
    setComparisonMode(mode);
    setComparison(null);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-mark"><Network size={19} strokeWidth={2.4} /></div>
          <div><strong>RAG LAB</strong><span>learning workspace</span></div>
        </div>

        <div className="side-section-label">WORKSPACE</div>
        <nav className="main-nav">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button key={item.id} className={`nav-item ${view === item.id ? "active" : ""}`} onClick={() => setView(item.id)}>
                <Icon size={16} /><span>{item.label}</span>
                {view === item.id && <ChevronRight size={14} className="nav-arrow" />}
              </button>
            );
          })}
        </nav>

        <div className="side-methods">
          <div className="side-section-label">CURRENT METHOD</div>
          <MethodPill icon={<Scissors size={13} />} label={chunkingMethod?.name || config.chunking_method} status="available" />
          <MethodPill icon={<Search size={13} />} label={retrievalMethod?.name || config.retrieval_method} status="available" />
          <MethodPill icon={<ListFilter size={13} />} label={rerankingMethod?.name || config.reranking_method} status="available" />
          <MethodPill icon={<Sparkles size={13} />} label={generationMethod?.name || config.generation_method} status="available" />
        </div>

        <div className="sidebar-footer">
          <div className="connection-dot"><span /> API CONNECTED</div>
          <div className={`connection-dot ${runtime?.longcat_configured ? "" : "disabled"}`}><span /> LONGCAT {runtime?.longcat_configured ? "READY" : "NOT CONFIGURED"}</div>
          <span>v{runtime?.api_version || "-"} / local demo</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb"><span>RAG LAB</span><ChevronRight size={13} /><strong>{currentTitle}</strong></div>
          <div className="topbar-actions">
            <div className="live-status"><span /> {configApplied ? "CONFIG APPLIED" : "CONFIG CHANGED"}</div>
            <button className="icon-button" title="按当前配置重建索引" onClick={() => handleBuild()} disabled={building}>
              {building ? <LoaderCircle size={16} className="spin" /> : <RefreshCw size={16} />}
            </button>
          </div>
        </header>

        <div className="content-wrap">
          {error && <ErrorNotice message={error} onClose={() => setError(null)} />}
          {loading ? <LoadingState /> : (
            <>
              <section className="hero-row">
                <div>
                  <div className="eyebrow"><span className="eyebrow-line" /> RAG / TRACEABLE DEMO</div>
                  <h1>{view === "ask" ? "让每一步都看得见。" : currentTitle}</h1>
                  <p className="hero-copy">基于 <strong>{document?.name || "知识库.md"}</strong> 的可解释检索增强生成实验。</p>
                </div>
                <div className="hero-meta">
                  <span className="meta-label">CURRENT INDEX</span>
                  <strong>{status?.ready ? "READY" : "NOT BUILT"}</strong>
                  <span>{status?.chunk_count || 0} chunks · {status?.vector_dimension || 0} terms</span>
                </div>
              </section>

              <PipelineRail trace={activeTrace} view={view} onStageClick={setView} />

              {view === "ask" && (
                <AskView
                  question={question} setQuestion={setQuestion} topK={topK} setTopK={setTopK}
                  asking={asking} onAsk={handleAsk} onExample={setQuestion} onNavigate={setView}
                  methods={methods} config={config} updateConfig={updateConfig}
                />
              )}
              {view === "source" && <SourceView document={document} />}
              {view === "chunks" && (
                <ChunksView chunks={chunks} methods={methods} config={config} updateConfig={updateConfig}
                  onBuild={() => handleBuild("chunks")} building={building} configApplied={configApplied} />
              )}
              {view === "index" && <IndexView chunks={chunks} status={status} trace={buildTrace} methods={methods} />}
              {view === "retrieval" && <RetrievalView result={result} onAsk={() => setView("ask")} />}
              {view === "context" && <ContextView result={result} />}
              {view === "compare" && (
                <ComparisonView question={comparisonQuestion} setQuestion={setComparisonQuestion} mode={comparisonMode}
                  setMode={handleComparisonMode} comparing={comparing} onCompare={handleCompare} comparison={comparison} />
              )}
              {view === "trace" && <TraceView trace={activeTrace} />}
            </>
          )}
        </div>
      </main>
    </div>
  );
}

function findMethod(methods: MethodOption[], id: string) {
  return methods.find((method) => method.id === id);
}

function MethodPill({ icon, label, status }: { icon: React.ReactNode; label: string; status: MethodStatus }) {
  return <div className="method-pill"><span className="pill-icon">{icon}</span><span className="pill-label">{label}</span><span className={`status-dot ${status}`} /></div>;
}

function PipelineRail({ trace, view, onStageClick }: { trace: PipelineStep[]; view: View; onStageClick: (view: View) => void }) {
  const completed = new Set(trace.map((step) => step.id));
  const stageView = (stage: string): View => {
    if (stage === "load") return "source";
    if (stage === "chunk") return "chunks";
    if (stage === "retrieve" || stage === "rerank") return "retrieval";
    if (stage === "generate") return "ask";
    return stage as View;
  };
  return (
    <section className="pipeline-rail" aria-label="RAG pipeline stages">
      <div className="rail-caption"><GitBranch size={15} /> PIPELINE TRACE <span>{trace.length ? "· recorded" : "· waiting"}</span></div>
      <div className="pipeline-steps">
        {pipelineStages.map((stage, index) => {
          const Icon = stage.icon;
          const done = completed.has(stage.id);
          const targetView = stageView(stage.id);
          const focused = view === targetView && !(view === "retrieval" && stage.id === "rerank");
          return (
            <div className="pipeline-step-wrap" key={stage.id}>
              <button className={`pipeline-step ${done ? "done" : "pending"} ${focused ? "focused" : ""}`} onClick={() => onStageClick(targetView)}>
                <span className="step-icon">{done ? <Check size={14} /> : <Icon size={14} />}</span>
                <span className="step-label">{stage.label}</span>
              </button>
              {index < pipelineStages.length - 1 && <span className={`step-connector ${done && completed.has(pipelineStages[index + 1].id) ? "done" : ""}`} />}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function AskView({ question, setQuestion, topK, setTopK, asking, onAsk, onExample, onNavigate, methods, config, updateConfig }: {
  question: string; setQuestion: (value: string) => void; topK: number; setTopK: (value: number) => void;
  asking: boolean; onAsk: (event: React.FormEvent) => void; onExample: (value: string) => void;
  onNavigate: (view: View) => void; methods: MethodOption[]; config: PipelineConfig;
  updateConfig: (patch: Partial<PipelineConfig>) => void;
}) {
  return (
    <div className="ask-layout">
      <section className="query-panel surface-panel">
        <div className="panel-heading">
          <div><span className="panel-kicker">01 / ASK</span><h2>向知识库提问</h2></div>
          <span className="mode-chip"><Sparkles size={13} /> {findMethod(methods, config.generation_method)?.name || config.generation_method}</span>
        </div>
        <form onSubmit={onAsk}>
          <div className="query-box">
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="输入一个关于知识库的问题..." rows={3} />
            <div className="query-box-footer">
              <span><CircleHelp size={14} /> 答案将附带证据引用</span>
              <button className="submit-button" type="submit" disabled={asking || !question.trim()}>
                {asking ? <LoaderCircle size={16} className="spin" /> : <ArrowUpRight size={16} />}
                {asking ? "处理中" : "开始检索"}
              </button>
            </div>
          </div>
        </form>
        <div className="examples-row"><span className="examples-label">TRY AN EXAMPLE</span>{examples.map((example) => <button key={example} onClick={() => onExample(example)}>{example}</button>)}</div>
      </section>

      <aside className="ask-side">
        <div className="control-block">
          <div className="control-label"><span>TOP-K 最终证据数</span><strong>{topK}</strong></div>
          <input type="range" min="1" max="8" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
          <div className="range-labels"><span>少量证据</span><span>更多候选</span></div>
        </div>
        <MethodControls methods={methods} config={config} updateConfig={updateConfig} />
        <button className="text-link" onClick={() => onNavigate("chunks")}>观察当前配置如何切分 <ArrowUpRight size={14} /></button>
      </aside>
    </div>
  );
}

function MethodControls({ methods, config, updateConfig }: { methods: MethodOption[]; config: PipelineConfig; updateConfig: (patch: Partial<PipelineConfig>) => void }) {
  const options = (category: MethodOption["category"]) => methods.filter((method) => method.category === category && method.status === "available");
  const updateChunkSize = (chunkSize: number) => updateConfig({
    chunk_size: chunkSize,
    chunk_overlap: Math.min(config.chunk_overlap, chunkSize - 20),
  });
  return (
    <div className="method-summary">
      <div className="summary-header"><span>ACTIVE CONFIG</span><ListFilter size={14} /></div>
      <label className="select-row"><span><Scissors size={14} />切分</span><select value={config.chunking_method} onChange={(event) => updateConfig({ chunking_method: event.target.value })}>{options("chunking").map((method) => <option value={method.id} key={method.id}>{method.name}</option>)}</select></label>
      <label className="select-row"><span><Search size={14} />检索</span><select value={config.retrieval_method} onChange={(event) => updateConfig({ retrieval_method: event.target.value })}>{options("retrieval").map((method) => <option value={method.id} key={method.id}>{method.name}</option>)}</select></label>
      <label className="select-row"><span><ListFilter size={14} />重排</span><select value={config.reranking_method} onChange={(event) => updateConfig({ reranking_method: event.target.value })}>{options("reranking").map((method) => <option value={method.id} key={method.id}>{method.name}</option>)}</select></label>
      <label className="select-row"><span><Sparkles size={14} />生成</span><select value={config.generation_method} onChange={(event) => updateConfig({ generation_method: event.target.value })}>{options("generation").map((method) => <option value={method.id} key={method.id}>{method.name}</option>)}</select></label>
      {config.chunking_method === "fixed_length" && (
        <div className="window-controls">
          <label><span>窗口大小 <strong>{config.chunk_size}</strong></span><input type="range" min="80" max="600" step="20" value={config.chunk_size} onChange={(event) => updateChunkSize(Number(event.target.value))} /></label>
          <label><span>重叠字符 <strong>{config.chunk_overlap}</strong></span><input type="range" min="0" max={Math.min(200, config.chunk_size - 20)} step="10" value={config.chunk_overlap} onChange={(event) => updateConfig({ chunk_overlap: Number(event.target.value) })} /></label>
        </div>
      )}
      {config.chunking_method === "semantic" && (
        <div className="window-controls semantic-controls">
          <label><span>边界阈值 <strong>{config.semantic_threshold.toFixed(2)}</strong></span><input type="range" min="0" max="0.3" step="0.01" value={config.semantic_threshold} onChange={(event) => updateConfig({ semantic_threshold: Number(event.target.value) })} /></label>
          <label><span>最大长度 <strong>{config.semantic_max_chars}</strong></span><input type="range" min="200" max="1200" step="20" value={config.semantic_max_chars} onChange={(event) => updateConfig({ semantic_max_chars: Number(event.target.value) })} /></label>
          <div className="semantic-scale"><span>更易合并</span><span>更易断开</span></div>
        </div>
      )}
    </div>
  );
}

function SourceView({ document }: { document: SourceDocument | null }) {
  if (!document) return <EmptyState label="知识库尚未加载" />;
  return <div className="source-layout"><section className="surface-panel source-preview"><div className="panel-heading"><div><span className="panel-kicker">SOURCE DOCUMENT</span><h2>{document.name}</h2></div><span className="file-badge"><FileText size={14} /> MARKDOWN</span></div><pre>{document.content}</pre></section><StatsPanel document={document} /></div>;
}

function StatsPanel({ document }: { document: SourceDocument }) {
  return <aside className="stats-panel"><div className="panel-kicker">DOCUMENT PROFILE</div><div className="big-stat"><strong>{document.character_count.toLocaleString()}</strong><span>characters</span></div><div className="stat-list"><div><span>段落</span><strong>{document.paragraph_count}</strong></div><div><span>识别章节</span><strong>{document.section_count}</strong></div><div><span>文件格式</span><strong>.md</strong></div></div><div className="section-list"><span className="panel-kicker">SECTIONS</span>{document.sections.map((section, index) => <div key={section}><span>{String(index + 1).padStart(2, "0")}</span>{section}</div>)}</div></aside>;
}

function ChunksView({ chunks, methods, config, updateConfig, onBuild, building, configApplied }: {
  chunks: Chunk[]; methods: MethodOption[]; config: PipelineConfig; updateConfig: (patch: Partial<PipelineConfig>) => void;
  onBuild: () => void; building: boolean; configApplied: boolean;
}) {
  const chunkingMethods = methods.filter((method) => method.category === "chunking");
  const active = findMethod(methods, config.chunking_method);
  return (
    <div className="detail-stack">
      <div className="view-toolbar">
        <div><span className="panel-kicker">CHUNKING</span><h2>文本被切成了什么样？</h2><p>{active?.description}</p></div>
        <button className="secondary-button" onClick={onBuild} disabled={building}>{building ? <LoaderCircle size={15} className="spin" /> : <RefreshCw size={15} />} {configApplied ? "重新切分" : "应用配置"}</button>
      </div>
      <div className="method-strip">
        {chunkingMethods.map((method) => (
          <button className={method.id === config.chunking_method ? "active-method" : "planned-method"} key={method.id}
            disabled={method.status === "planned"} onClick={() => updateConfig({ chunking_method: method.id })}>
            <span>{method.id === config.chunking_method ? "SELECTED" : method.status.toUpperCase()}</span><strong>{method.name}</strong><small>{method.status === "available" ? method.advantages[0] : method.limitations[0]}</small>
          </button>
        ))}
      </div>
      {(config.chunking_method === "fixed_length" || config.chunking_method === "semantic") && <MethodControls methods={methods} config={config} updateConfig={updateConfig} />}
      {!configApplied && <div className="pending-notice"><TriangleAlert size={15} /> 当前列表仍是上一次索引结果，点击“应用配置”后刷新。</div>}
      <div className="chunk-grid">{chunks.map((chunk) => <ChunkCard chunk={chunk} key={chunk.id} />)}</div>
    </div>
  );
}

function ChunkCard({ chunk, hit }: { chunk: Chunk; hit?: RetrievalHit }) {
  return (
    <article className={`chunk-card ${hit?.rank === 1 ? "top-hit" : ""}`}>
      <div className="chunk-card-top">
        <span className="chunk-id">{hit ? `#${hit.rank}` : chunk.id}</span><span className="chunk-section">{chunk.section}</span>
        {hit && <strong className="score-badge">{hit.score.toFixed(4)}</strong>}
      </div>
      {hit && <div className="score-detail"><span>{hit.score_label}</span>{Object.entries(hit.score_components).map(([name, value]) => <span key={name}>{name.toUpperCase()} {value.toFixed(3)}</span>)}{hit.rerank_score !== null && <span>重排 {hit.rerank_score.toFixed(3)}</span>}</div>}
      {hit && hit.rerank_score !== null && hit.retrieval_rank !== hit.rank && <div className="rank-shift">初排 #{hit.retrieval_rank}<ArrowRight size={12} />重排 #{hit.rank}</div>}
      {chunk.split_reason && <div className={`semantic-boundary ${chunk.split_reason}`}><GitBranch size={12} /><span>{semanticReasonLabel(chunk.split_reason)}</span>{chunk.boundary_similarity !== null && <strong>相似度 {chunk.boundary_similarity.toFixed(3)}</strong>}<small>{chunk.semantic_unit_count} units</small></div>}
      <p>{chunk.text}</p>
      <div className="chunk-card-bottom"><span>{chunk.character_count} chars</span><span>位置 {chunk.start_char}-{chunk.end_char}</span>{chunk.overlap_chars > 0 && <span>overlap {chunk.overlap_chars}</span>}{hit && hit.matched_terms.length > 0 && <span className="matched-terms">命中 {hit.matched_terms.slice(0, 4).join(" / ")}</span>}</div>
    </article>
  );
}

function semanticReasonLabel(reason: string) {
  if (reason === "semantic_drop") return "语义相似度下降";
  if (reason === "max_chars") return "达到长度上限";
  return "文档起点";
}

function IndexView({ chunks, status, trace, methods }: { chunks: Chunk[]; status: IndexStatus | null; trace: PipelineStep[]; methods: MethodOption[] }) {
  const sample = chunks[0];
  const method = status ? findMethod(methods, status.retrieval_method) : null;
  const maxWeight = Math.max(...(sample?.vector?.top_terms || []).map((term) => term.weight), 1);
  return (
    <div className="detail-stack">
      <div className="view-toolbar"><div><span className="panel-kicker">INDEX / {status?.retrieval_method.toUpperCase()}</span><h2>文本表示如何生成并写入？</h2><p>{method?.description}</p></div><div className="ready-badge"><span /> {status?.ready ? "INDEX READY" : "NOT READY"}</div></div>
      <div className="index-metrics"><Metric label="索引记录" value={String(status?.chunk_count || 0)} suffix="chunks" icon={<Database size={16} />} /><Metric label="词表维度" value={String(status?.vector_dimension || 0)} suffix="terms" icon={<Boxes size={16} />} /><Metric label="当前方法" value={status?.retrieval_method.toUpperCase() || "-"} suffix="" icon={<Check size={16} />} /><Metric label="构建耗时" value={trace.find((step) => step.id === "index")?.duration_ms.toFixed(1) || "-"} suffix="ms" icon={<Timer size={16} />} /></div>
      <div className="index-explain">
        <div className="explain-graphic"><div className="vector-source"><FileText size={18} /><span>{sample?.id || "chunk-001"}</span></div><ArrowUpRight size={18} /><div className="vector-box"><Boxes size={18} /><span>{status?.retrieval_method.toUpperCase()}</span><small>文本表示</small></div><ArrowUpRight size={18} /><div className="vector-store"><Database size={18} /><span>INDEX</span><small>{status?.chunk_count || 0} 条记录</small></div></div>
        <div className="vector-sample"><div className="sample-heading"><span>TERM PREVIEW / {sample?.id || "-"}</span><span>{sample?.vector?.nonzero_count || 0} non-zero terms</span></div><div className="term-bars">{(sample?.vector?.top_terms || []).map((term) => <div className="term-row" key={term.term}><span>{term.term}</span><div><i style={{ width: `${Math.max(6, term.weight / maxWeight * 100)}%` }} /></div><b>{term.weight.toFixed(4)}</b></div>)}</div></div>
      </div>
    </div>
  );
}

function Metric({ label, value, suffix, icon }: { label: string; value: string; suffix: string; icon: React.ReactNode }) { return <div className="metric"><span className="metric-icon">{icon}</span><span className="metric-label">{label}</span><strong>{value}<small>{suffix}</small></strong></div>; }

function RetrievalView({ result, onAsk }: { result: QueryResponse | null; onAsk: () => void }) {
  if (!result) return <EmptyState label="先在问答工作台提交问题" action="返回提问" onAction={onAsk} />;
  const rerankStep = result.trace.find((step) => step.id === "rerank");
  return (
    <div className="detail-stack">
      <div className="result-header"><div><span className="panel-kicker">RETRIEVAL / {result.index_status.retrieval_method.toUpperCase()}</span><h2>这次检索和重排发生了什么？</h2><p className="asked-question">“{result.question}”</p></div><div className="result-actions"><button className="secondary-button" onClick={() => downloadQueryReport(result)}><Download size={14} /> 导出演示报告</button><div className="duration"><Timer size={15} /><strong>{result.total_duration_ms.toFixed(1)} ms</strong><span>total trace</span></div></div></div>
      <div className="answer-panel">
        <div className="answer-label"><Sparkles size={15} /> ANSWER / {result.answer_mode}<span className="intent-badge">{intentLabel(result.answer_intent)}</span></div>
        <div className="generation-meta">
          <span>{result.generation_metadata.provider}</span>
          <span>{result.generation_metadata.model || result.generation_metadata.effective_method}</span>
          {result.generation_metadata.total_tokens !== null && <span>{result.generation_metadata.total_tokens} tokens</span>}
          {result.generation_metadata.finish_reason && <span>finish: {result.generation_metadata.finish_reason}</span>}
          {result.generation_metadata.fallback_used && <span className="fallback-chip">FALLBACK</span>}
        </div>
        {result.generation_warning && <div className="generation-warning"><TriangleAlert size={14} />{result.generation_warning}</div>}
        {result.answer_points.length ? (
          <div className="answer-points">{result.answer_points.map((point, index) => (
            <div className="answer-point" key={`${point.chunk_id}-${index}`}>
              <span className="answer-point-index">{String(index + 1).padStart(2, "0")}</span>
              <div><p>{point.text}</p><small>{point.selection_reason} · {point.chunk_id}</small></div>
              <span className="point-citation">{point.citation}</span>
            </div>
          ))}</div>
        ) : <p>{result.answer}</p>}
        <div className="citation-row">{result.citations.length ? result.citations.map((citation) => <span key={citation}>{citation}</span>) : <span className="muted">没有可用引用</span>}</div>
      </div>
      <div className="ranking-summary"><div><span>初次召回</span><strong>{(rerankStep?.detail.before as string[] | undefined)?.length || result.retrieval_hits.length}</strong></div><ArrowRight size={17} /><div><span>最终上下文</span><strong>{result.retrieval_hits.length}</strong></div><div className="ranking-method"><span>RERANK</span><strong>{String(rerankStep?.detail.method || "none")}</strong></div></div>
      <div className="hits-heading"><span className="panel-kicker">RANKED EVIDENCE</span><span>卡片同时保留初排分数、融合分量和重排变化</span></div>
      <div className="hits-list">{result.retrieval_hits.map((hit) => <ChunkCard chunk={hit.chunk} hit={hit} key={hit.chunk.id} />)}</div>
    </div>
  );
}

function ContextView({ result }: { result: QueryResponse | null }) {
  if (!result) return <EmptyState label="提交问题后查看送入生成阶段的上下文" />;
  return <div className="context-layout"><section className="surface-panel context-panel"><div className="panel-heading"><div><span className="panel-kicker">CONTEXT WINDOW</span><h2>送入生成阶段的资料</h2></div><span className="file-badge">{result.context.length} BLOCKS</span></div>{result.context.map((block) => <div className="context-block" key={block.chunk_id}><div><span className="citation-mark">{block.citation}</span><span>{block.section}</span><code>{block.chunk_id}</code></div><p>{block.text}</p></div>)}</section><section className="prompt-panel"><div className="panel-kicker">PROMPT PREVIEW</div><pre>{result.prompt_preview}</pre></section></div>;
}

function ComparisonView({ question, setQuestion, mode, setMode, comparing, onCompare, comparison }: {
  question: string; setQuestion: (value: string) => void; mode: ComparisonMode; setMode: (mode: ComparisonMode) => void;
  comparing: boolean; onCompare: (event: React.FormEvent) => void; comparison: ComparisonResponse | null;
}) {
  return (
    <div className="detail-stack">
      <section className="compare-toolbar surface-panel">
        <div className="compare-controls">
          <div><span className="panel-kicker">CONTROLLED COMPARISON</span><h2>同一个问题，只改变一种方法</h2></div>
          <div className="segmented-control"><button className={mode === "retrieval" ? "active" : ""} onClick={() => setMode("retrieval")}>检索</button><button className={mode === "chunking" ? "active" : ""} onClick={() => setMode("chunking")}>切分</button><button className={mode === "generation" ? "active" : ""} onClick={() => setMode("generation")}>生成</button></div>
        </div>
        <form onSubmit={onCompare} className="compare-form"><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="输入用于对比的问题" />{comparison && <button type="button" className="secondary-button" onClick={() => downloadComparisonReport(comparison)}><Download size={14} /> 导出对比</button>}<button className="submit-button" disabled={comparing || !question.trim()}>{comparing ? <LoaderCircle size={15} className="spin" /> : <Scale size={15} />}{comparing ? "对比中" : "运行对比"}</button></form>
      </section>
      {!comparison ? <EmptyState label="运行一次对比后，这里会并列展示各方法的召回与回答" /> : (
        <div className="comparison-grid">{comparison.runs.map((run) => (
          <article className="comparison-card" key={`${run.config.chunking_method}-${run.config.retrieval_method}-${run.config.generation_method}`}>
            <div className="comparison-card-head"><div><span>METHOD</span><h3>{run.config.label}</h3></div><strong>{run.result.total_duration_ms.toFixed(1)} ms</strong></div>
            <div className="comparison-tags"><span>{run.config.chunking_method}</span><span>{run.config.retrieval_method}</span><span>{run.config.reranking_method}</span><span>{run.config.generation_method}</span></div>
            <div className="comparison-metrics"><div><strong>{run.result.index_status.chunk_count}</strong><span>chunks</span></div><div><strong>{run.result.retrieval_hits.length}</strong><span>hits</span></div><div><strong>{run.result.context.length}</strong><span>context</span></div></div>
            <div className="comparison-generation"><span>{run.result.generation_metadata.provider}</span><strong>{run.result.generation_metadata.model || run.result.generation_metadata.effective_method}</strong><b>{run.result.generation_metadata.total_tokens !== null ? `${run.result.generation_metadata.total_tokens} tokens` : "local"}</b></div>
            <div className="comparison-answer"><span>ANSWER / {intentLabel(run.result.answer_intent)}</span>{run.result.answer_points.length ? <ol>{run.result.answer_points.map((point, index) => <li key={`${point.chunk_id}-${index}`}>{point.text} <b>{point.citation}</b></li>)}</ol> : <p>{run.result.answer}</p>}</div>
            <div className="compact-hits">{run.result.retrieval_hits.slice(0, 3).map((hit) => <div key={hit.chunk.id}><strong>#{hit.rank}</strong><span>{hit.chunk.section}</span><b>{hit.score.toFixed(3)}</b></div>)}</div>
          </article>
        ))}</div>
      )}
    </div>
  );
}

function intentLabel(intent: QueryResponse["answer_intent"]) {
  return ({ list: "枚举", process: "流程", fact: "事实", general: "综合", fallback: "资料不足" } as const)[intent];
}

function downloadQueryReport(result: QueryResponse) {
  const metadata = result.generation_metadata;
  const lines = [
    "# RAG 单次问答报告",
    "",
    `- 问题：${result.question}`,
    `- 切分方法：${result.index_status.chunking_method}`,
    `- 检索方法：${result.index_status.retrieval_method}`,
    `- 生成方法：${metadata.effective_method}`,
    `- 模型：${metadata.model || "本地规则"}`,
    `- 总耗时：${result.total_duration_ms.toFixed(2)} ms`,
    `- Token：${metadata.total_tokens ?? "不适用"}`,
    `- 是否回退：${metadata.fallback_used ? "是" : "否"}`,
    "",
    "## 回答",
    "",
    result.answer,
    "",
    "## 引用",
    "",
    ...(result.citations.length ? result.citations.map((citation) => `- ${citation}`) : ["- 无有效引用"]),
    "",
    "## 检索结果",
    "",
    "| 排名 | Chunk | 章节 | 分数 | 命中词 |",
    "| --- | --- | --- | ---: | --- |",
    ...result.retrieval_hits.map((hit) => `| ${hit.rank} | ${hit.chunk.id} | ${escapeTable(hit.chunk.section)} | ${hit.score.toFixed(4)} | ${escapeTable(hit.matched_terms.join("、"))} |`),
    "",
    "## Pipeline Trace",
    "",
    "| 阶段 | 状态 | 耗时 | 摘要 |",
    "| --- | --- | ---: | --- |",
    ...result.trace.map((step) => `| ${step.id} | ${step.status} | ${step.duration_ms.toFixed(2)} ms | ${escapeTable(step.summary)} |`),
    "",
  ];
  downloadMarkdown(`rag-query-report-${Date.now()}.md`, lines.join("\n"));
}

function downloadComparisonReport(comparison: ComparisonResponse) {
  const lines = ["# RAG 方法对比报告", "", `问题：${comparison.question}`, ""];
  for (const run of comparison.runs) {
    lines.push(
      `## ${run.config.label}`,
      "",
      `- 切分：${run.config.chunking_method}`,
      `- 检索：${run.config.retrieval_method}`,
      `- 重排：${run.config.reranking_method}`,
      `- 生成：${run.config.generation_method}`,
      `- 耗时：${run.result.total_duration_ms.toFixed(2)} ms`,
      `- Token：${run.result.generation_metadata.total_tokens ?? "不适用"}`,
      "",
      run.result.answer,
      "",
    );
  }
  downloadMarkdown(`rag-comparison-report-${Date.now()}.md`, lines.join("\n"));
}

function escapeTable(value: string) {
  return value.replace(/\|/g, "\\|").replace(/\s+/g, " ");
}

function downloadMarkdown(filename: string, content: string) {
  const url = URL.createObjectURL(new Blob([content], { type: "text/markdown;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function TraceView({ trace }: { trace: PipelineStep[] }) {
  if (!trace.length) return <EmptyState label="提交问题或构建索引后查看 trace" />;
  return <div className="trace-layout"><div className="trace-timeline">{trace.map((step, index) => <article className="trace-item" key={`${step.id}-${index}`}><div className="trace-marker"><span>{String(index + 1).padStart(2, "0")}</span>{index < trace.length - 1 && <i />}</div><div className="trace-body"><div className="trace-top"><div><span className="trace-stage">{step.id.toUpperCase()} · {step.status}</span><h3>{step.name}</h3></div><strong>{step.duration_ms.toFixed(2)} ms</strong></div><p>{step.summary}</p><details><summary>查看阶段输出</summary><pre>{JSON.stringify(step.detail, null, 2)}</pre></details></div></article>)}</div><aside className="trace-note"><TerminalSquare size={18} /><span>WHY TRACE?</span><p>RAG 调试通常要判断问题发生在召回、重排还是生成。保留中间结果，才能解释一次回答为何成立或失败。</p></aside></div>;
}

function EmptyState({ label, action, onAction }: { label: string; action?: string; onAction?: () => void }) { return <div className="empty-state"><CircleHelp size={22} /><p>{label}</p>{action && onAction && <button className="secondary-button" onClick={onAction}>{action}</button>}</div>; }
function LoadingState() { return <div className="loading-state"><LoaderCircle size={20} className="spin" /><span>正在读取知识库与索引状态...</span></div>; }
function ErrorNotice({ message, onClose }: { message: string; onClose: () => void }) { return <div className="error-notice"><TriangleAlert size={17} /><span>{message}</span><button title="关闭" onClick={onClose}><X size={15} /></button></div>; }

export default App;
