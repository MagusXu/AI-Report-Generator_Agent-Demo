import { FormEvent, Fragment, useEffect, useMemo, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

type Project = {
  id: string;
  name: string;
  industry: string;
  year: string;
  language: string;
};

type DocumentItem = {
  id: string;
  name: string;
  type: string;
  selected: boolean;
  summary: string;
  file_path?: string | null;
  upload_status?: string;
  parse_status?: string;
  chunk_count?: number;
  indexed_at?: string | null;
  error_message?: string | null;
};

type ChunkQuality = {
  status: "good" | "warn" | "poor";
  score: number;
  chunk_count: number;
  avg_chars: number;
  min_chars: number;
  max_chars: number;
  short_chunks: number;
  long_chunks: number;
  heading_coverage: number;
  parent_coverage: number;
  table_chunks: number;
  warnings: string[];
};

type DocumentPreview = DocumentItem & {
  quality_metrics: ChunkQuality;
  chunks: {
    id: string;
    chunk_index: number;
    text: string;
    source_locator: string;
    created_at: string;
    metadata: {
      block_type?: string;
      heading_path?: string;
      page_start?: string | number;
      page_end?: string | number;
      parent_id?: string;
      parent_text?: string;
      quality_status?: string;
      quality_score?: string | number;
    };
  }[];
};

type Citation = {
  id: string;
  kind: "document" | "chunk";
  document_id: string;
  document_name: string;
  document_type: string;
  source_locator: string;
  text: string;
};

type SectionVersion = {
  id: string;
  section_id: string;
  version_number: number;
  source: string;
  summary: string;
  content: string;
  reference_ids: string[];
  created_at: string;
};

type ReportSection = {
  id: string;
  number: string;
  title: string;
  level: number;
  parent_id: string | null;
  order_index: number;
  prompt: string;
  current_version_id: string | null;
  confirmed: boolean;
  versions: SectionVersion[];
  current_version: SectionVersion | null;
};

type ExportRecord = {
  id: string;
  format: string;
  status: string;
  issues: string[];
  created_at: string;
};

type AICallLog = {
  id: string;
  operation: string;
  model: string;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  metadata: Record<string, unknown>;
  created_at: string;
};

type Workspace = {
  project: Project;
  documents: DocumentItem[];
  sections: ReportSection[];
  citations: Record<string, Citation>;
  ai_call_logs: AICallLog[];
  export_records: ExportRecord[];
};

type PromptAssist = {
  objective: string;
  key_questions: string;
  geography: string;
  risk_dimensions: string;
  tone_length: string;
  exclusions: string;
};

type RuntimeInfo = {
  llm?: {
    default_system_prompt?: string;
  };
};

type GenerationSettings = {
  style: "strict" | "balanced" | "creative";
  length: "short" | "medium" | "long";
  stream: boolean;
  retrieval_top_k: number;
  per_document_limit: number;
  use_parent_context: boolean;
  temperature: number;
  max_tokens: number;
};

type PromptMessage = {
  role: string;
  content: string;
};

type RetrievedLogChunk = {
  id?: string;
  document_name?: string;
  document_type?: string;
  heading_path?: string;
  source_locator?: string;
  distance?: number;
  text?: string;
  parent_text?: string;
};

const emptyWorkspace: Workspace = {
  project: { id: "", name: "", industry: "", year: "", language: "" },
  documents: [],
  sections: [],
  citations: {},
  ai_call_logs: [],
  export_records: [],
};

const emptyAssist: PromptAssist = {
  objective: "",
  key_questions: "",
  geography: "",
  risk_dimensions: "",
  tone_length: "",
  exclusions: "",
};

const fallbackSystemPrompt = `你是一名投行内部行业风险研究员。你必须只基于给定资料写作，不要编造具体数据。

写作要求：
1. 使用投行内部行业风险研究报告口吻。
2. 结构为：核心判断、关键依据、风险提示、银行业务含义。
3. 每个主要判断后尽量给出引用标注，格式必须是 [ref:chunk_id]。
4. 如果资料不足，请明确写“现有资料不足以支持更细判断”，不要编造。`;

const defaultGenerationSettings: GenerationSettings = {
  style: "strict",
  length: "medium",
  stream: true,
  retrieval_top_k: 20,
  per_document_limit: 3,
  use_parent_context: true,
  temperature: 0.2,
  max_tokens: 2200,
};

async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
  const body = options?.body;
  const isFormData = body instanceof FormData;
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: isFormData ? options?.headers : { "Content-Type": "application/json", ...(options?.headers ?? {}) },
  });

  if (!response.ok) {
    const text = await response.text();
    let message = text;
    try {
      const data = JSON.parse(text);
      message = data.detail ?? text;
    } catch {
      message = text;
    }
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

type GenerateStreamEvent =
  | { event: "status"; data: { phase?: string; message?: string } }
  | { event: "delta"; data: { content: string } }
  | { event: "done"; data: { workspace: Workspace } }
  | { event: "error"; data: { detail: string } };

function parseSseBlock(block: string): GenerateStreamEvent | null {
  let event = "message";
  let data = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return null;
  return { event, data: JSON.parse(data) } as GenerateStreamEvent;
}

async function consumeGenerateStream(
  response: Response,
  onEvent: (event: GenerateStreamEvent) => void,
): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("浏览器不支持流式响应");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const parsed = parseSseBlock(raw);
      if (parsed) onEvent(parsed);
      boundary = buffer.indexOf("\n\n");
    }
  }
}

function formatTime(value?: string | null) {
  if (!value) return "未入库";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function sectionLabel(section: ReportSection) {
  return section.level === 1 ? `${section.number}、${section.title}` : `${section.number}. ${section.title}`;
}

function statusLabel(document: DocumentItem) {
  if (document.parse_status === "indexed") return `已入库 · ${document.chunk_count ?? 0} chunks`;
  if (document.parse_status === "parsing") return "解析中";
  if (document.parse_status === "failed") return "入库失败";
  return "未入库";
}

function tokenText(value: number | null) {
  return typeof value === "number" ? value.toLocaleString("zh-CN") : "未返回";
}

function percentText(value: number) {
  return `${Math.round(value * 100)}%`;
}

function metadataText(value: unknown) {
  if (typeof value === "string") return value;
  return JSON.stringify(value ?? {}, null, 2);
}

function promptMessages(log: AICallLog | null): PromptMessage[] {
  const value = log?.metadata.prompt_structure;
  return Array.isArray(value) ? (value as PromptMessage[]) : [];
}

function retrievedLogChunks(log: AICallLog | null): RetrievedLogChunk[] {
  const value = log?.metadata.retrieved_chunks;
  return Array.isArray(value) ? (value as RetrievedLogChunk[]) : [];
}

function App() {
  const [workspace, setWorkspace] = useState<Workspace>(emptyWorkspace);
  const [activeSectionId, setActiveSectionId] = useState("");
  const [promptDraft, setPromptDraft] = useState("");
  const [systemPromptDraft, setSystemPromptDraft] = useState(fallbackSystemPrompt);
  const [contentDraft, setContentDraft] = useState("");
  const [projectDraft, setProjectDraft] = useState<Project>(emptyWorkspace.project);
  const [fileDraft, setFileDraft] = useState<File | null>(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [fileType, setFileType] = useState("研报");
  const [assistDraft, setAssistDraft] = useState<PromptAssist>(emptyAssist);
  const [assistOpen, setAssistOpen] = useState(false);
  const [previewDocument, setPreviewDocument] = useState<DocumentPreview | null>(null);
  const [selectedLog, setSelectedLog] = useState<AICallLog | null>(null);
  const [generationSettings, setGenerationSettings] = useState<GenerationSettings>(defaultGenerationSettings);
  const [liveContent, setLiveContent] = useState("");
  const [generating, setGenerating] = useState(false);
  const [viewMode, setViewMode] = useState<"section" | "preview">("section");
  const [status, setStatus] = useState("加载中");
  const [busy, setBusy] = useState(false);

  const activeSection = useMemo(
    () => workspace.sections.find((section) => section.id === activeSectionId) ?? workspace.sections[0],
    [activeSectionId, workspace.sections],
  );

  const selectedDocumentIds = useMemo(
    () => workspace.documents.filter((document) => document.selected).map((document) => document.id),
    [workspace.documents],
  );

  const selectedIndexedCount = useMemo(
    () =>
      workspace.documents.filter(
        (document) => document.selected && document.parse_status === "indexed" && (document.chunk_count ?? 0) > 0,
      ).length,
    [workspace.documents],
  );

  const previewIssues = useMemo(() => {
    const topLevelMissing = workspace.sections
      .filter((section) => section.level === 1 && !section.current_version)
      .map((section) => section.title);
    const unconfirmed = workspace.sections
      .filter((section) => section.current_version && !section.confirmed)
      .map((section) => section.title);
    const issues = [];
    if (topLevelMissing.length) issues.push(`缺失一级章节：${topLevelMissing.join("、")}`);
    if (unconfirmed.length) issues.push(`存在未确认版本：${unconfirmed.slice(0, 6).join("、")}`);
    if (selectedDocumentIds.length < 2) issues.push("当前参考文档少于 2 份");
    if (!selectedIndexedCount) issues.push("当前没有已入库的真实参考文档");
    return issues;
  }, [selectedDocumentIds.length, selectedIndexedCount, workspace.sections]);

  async function restoreDefaultSystemPrompt() {
    try {
      const runtime = await requestJson<RuntimeInfo>("/api/runtime");
      const next = runtime.llm?.default_system_prompt?.trim() || fallbackSystemPrompt;
      setSystemPromptDraft(next);
      setStatus("System Prompt 已恢复为默认值");
    } catch {
      setSystemPromptDraft(fallbackSystemPrompt);
      setStatus("无法读取后端默认值，已使用本地默认 System Prompt");
    }
  }

  async function loadWorkspace() {
    const [data, runtime] = await Promise.all([
      requestJson<Workspace>("/api/workspace"),
      requestJson<RuntimeInfo>("/api/runtime").catch((): RuntimeInfo => ({})),
    ]);
    setWorkspace(data);
    if (runtime.llm?.default_system_prompt) {
      setSystemPromptDraft(runtime.llm.default_system_prompt);
    }
    setProjectDraft(data.project);
    const firstSection = data.sections[0];
    if (!activeSectionId && firstSection) {
      setActiveSectionId(firstSection.id);
    }
    setStatus("本地后端已连接");
  }

  useEffect(() => {
    loadWorkspace().catch(() => setStatus("后端未连接"));
  }, []);

  useEffect(() => {
    if (!activeSection) return;
    setPromptDraft(activeSection.prompt);
    setContentDraft(activeSection.current_version?.content ?? "");
  }, [activeSection?.id, activeSection?.current_version_id]);

  async function runAction(action: () => Promise<Workspace>, doneText: string) {
    setBusy(true);
    try {
      const data = await action();
      setWorkspace(data);
      setProjectDraft(data.project);
      setStatus(doneText);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "操作失败");
    } finally {
      setBusy(false);
    }
  }

  async function generateCurrentSection() {
    if (!activeSection) return;

    const payload = {
      prompt: promptDraft,
      system_prompt: systemPromptDraft,
      reference_ids: selectedDocumentIds,
      ...generationSettings,
    };

    if (!generationSettings.stream) {
      await runAction(
        () =>
          requestJson<Workspace>(`/api/sections/${activeSection.id}/generate`, {
            method: "POST",
            body: JSON.stringify(payload),
          }),
        "已基于真实 RAG 生成新版本",
      );
      return;
    }

    setBusy(true);
    setGenerating(true);
    setLiveContent("");
    setContentDraft("");
    setStatus("检索资料并准备生成…");

    try {
      const response = await fetch(`${API_BASE}/api/sections/${activeSection.id}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const text = await response.text();
        let message = text;
        try {
          const data = JSON.parse(text);
          message = data.detail ?? text;
        } catch {
          message = text;
        }
        throw new Error(message || `Request failed: ${response.status}`);
      }

      let streamed = "";
      await consumeGenerateStream(response, (event) => {
        if (event.event === "status") {
          if (event.data.message) setStatus(event.data.message);
          return;
        }
        if (event.event === "delta") {
          streamed += event.data.content;
          setLiveContent(streamed);
          setContentDraft(streamed);
          return;
        }
        if (event.event === "error") {
          throw new Error(event.data.detail || "流式生成失败");
        }
        if (event.event === "done") {
          setWorkspace(event.data.workspace);
          setProjectDraft(event.data.workspace.project);
          setStatus("已基于真实 RAG 流式生成新版本");
        }
      });
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "流式生成失败");
    } finally {
      setGenerating(false);
      setLiveContent("");
      setBusy(false);
    }
  }

  function renderContentWithCitations(content: string) {
    const parts = content.split(/(\[ref:[^\]]+\])/g);
    return parts.map((part, index) => {
      const match = part.match(/^\[ref:([^\]]+)\]$/);
      if (!match) {
        return <Fragment key={`${part}-${index}`}>{part}</Fragment>;
      }

      const citation = workspace.citations[match[1]];
      return (
        <span key={`${part}-${index}`} className="citation" tabIndex={0}>
          [{citation?.document_type ?? "资料"}]
          <span className="citation-popover">
            <strong>{citation?.document_name ?? match[1]}</strong>
            <small>{citation?.source_locator ?? "来源位置待确认"}</small>
            <small>{citation?.text ?? "当前引用暂未找到对应片段。"}</small>
          </span>
        </span>
      );
    });
  }

  function saveProject(event: FormEvent) {
    event.preventDefault();
    runAction(
      () =>
        requestJson<Workspace>("/api/project", {
          method: "PATCH",
          body: JSON.stringify(projectDraft),
        }),
      "报告基础信息已保存",
    );
  }

  function createNewReport() {
    const confirmed = window.confirm("新建报告会清空所有章节版本和导出记录，参考文档会保留。确认继续？");
    if (!confirmed) return;
    runAction(
      () =>
        requestJson<Workspace>("/api/project/new", {
          method: "POST",
          body: JSON.stringify(projectDraft),
        }),
      "已新建报告并清空章节内容",
    );
  }

  function uploadFile(event: FormEvent) {
    event.preventDefault();
    if (!fileDraft) return;

    const formData = new FormData();
    formData.append("file", fileDraft);
    formData.append("type", fileType);
    runAction(
      () =>
        requestJson<Workspace>("/api/documents/upload-file", {
          method: "POST",
          body: formData,
        }),
      "文档已解析并写入向量库",
    );
    setFileDraft(null);
    setFileInputKey((key) => key + 1);
  }

  async function openDocumentPreview(documentId: string) {
    setBusy(true);
    try {
      const data = await requestJson<DocumentPreview>(`/api/documents/${documentId}/preview`);
      setPreviewDocument(data);
      setStatus("参考文档预览已打开");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "文档预览失败");
    } finally {
      setBusy(false);
    }
  }

  function deleteDocument(document: DocumentItem) {
    const confirmed = window.confirm(`删除参考文档《${document.name}》？已生成的章节版本不会被删除。`);
    if (!confirmed) return;
    runAction(
      () =>
        requestJson<Workspace>(`/api/documents/${document.id}`, {
          method: "DELETE",
        }),
      "参考文档已删除",
    );
  }

  function reindexDocument(document: DocumentItem) {
    const confirmed = window.confirm(`使用新版 chunking 策略重新解析《${document.name}》？`);
    if (!confirmed) return;
    runAction(async () => {
      const data = await requestJson<Workspace>(`/api/documents/${document.id}/reindex`, {
        method: "POST",
      });
      if (previewDocument?.id === document.id) {
        const updatedPreview = await requestJson<DocumentPreview>(`/api/documents/${document.id}/preview`);
        setPreviewDocument(updatedPreview);
      }
      return data;
    }, "参考文档已重新解析入库");
  }

  function updateStyle(style: GenerationSettings["style"]) {
    const temperatureMap = { strict: 0.2, balanced: 0.4, creative: 0.7 };
    setGenerationSettings({ ...generationSettings, style, temperature: temperatureMap[style] });
  }

  function updateLength(length: GenerationSettings["length"]) {
    const maxTokensMap = { short: 1000, medium: 2200, long: 3600 };
    setGenerationSettings({ ...generationSettings, length, max_tokens: maxTokensMap[length] });
  }

  if (!workspace.sections.length) {
    return (
      <main className="loading-screen">
        <h1>AI Report Generator</h1>
        <p>{status}</p>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <aside className="left-pane">
        <form className="project-form" onSubmit={saveProject}>
          <p className="eyebrow">报告项目</p>
          <input
            className="project-name"
            value={projectDraft.name}
            onChange={(event) => setProjectDraft({ ...projectDraft, name: event.target.value })}
            aria-label="报告名称"
          />
          <div className="project-grid">
            <label>
              行业
              <input
                value={projectDraft.industry}
                onChange={(event) => setProjectDraft({ ...projectDraft, industry: event.target.value })}
              />
            </label>
            <label>
              年份
              <input
                value={projectDraft.year}
                onChange={(event) => setProjectDraft({ ...projectDraft, year: event.target.value })}
              />
            </label>
            <label>
              语言
              <input
                value={projectDraft.language}
                onChange={(event) => setProjectDraft({ ...projectDraft, language: event.target.value })}
              />
            </label>
          </div>
          <div className="project-actions">
            <button type="submit" disabled={busy}>保存信息</button>
            <button type="button" disabled={busy} onClick={createNewReport}>新建报告</button>
          </div>
        </form>

        <section className="reference-panel">
          <div className="section-heading">
            <span>参考文档</span>
            <small>{selectedIndexedCount} 份已入库并勾选</small>
          </div>
          <div className="reference-list">
            {workspace.documents.map((document) => (
              <div key={document.id} className={`reference-item ${document.parse_status ?? "not_indexed"}`}>
                <label className="reference-check">
                  <input
                    type="checkbox"
                    checked={document.selected}
                    onChange={(event) =>
                      runAction(
                        () =>
                          requestJson<Workspace>(`/api/documents/${document.id}/selection`, {
                            method: "PATCH",
                            body: JSON.stringify({ selected: event.target.checked }),
                          }),
                        "参考范围已更新",
                      )
                    }
                  />
                  <span>{document.name}</span>
                  <em>{document.type}</em>
                </label>
                <div className="reference-meta">
                  <small className="doc-status">{statusLabel(document)}</small>
                  {document.error_message ? <small className="doc-error">{document.error_message}</small> : null}
                </div>
                <div className="document-actions">
                  <button type="button" disabled={busy} onClick={() => openDocumentPreview(document.id)}>
                    预览
                  </button>
                  {document.file_path ? (
                    <button type="button" disabled={busy} onClick={() => reindexDocument(document)}>
                      重解析
                    </button>
                  ) : null}
                  <button type="button" className="text-danger" disabled={busy} onClick={() => deleteDocument(document)}>
                    删除
                  </button>
                </div>
              </div>
            ))}
          </div>

          <form className="upload-card" onSubmit={uploadFile}>
            <label className="file-picker">
              <span>{fileDraft ? fileDraft.name : "选择本地文件"}</span>
              <small>.txt / .md / .pdf / .docx / .xlsx</small>
              <input
                key={fileInputKey}
                type="file"
                accept=".txt,.md,.pdf,.docx,.xlsx"
                onChange={(event) => setFileDraft(event.target.files?.[0] ?? null)}
              />
            </label>
            <div className="upload-controls">
              <select value={fileType} onChange={(event) => setFileType(event.target.value)}>
                <option>研报</option>
                <option>政策</option>
                <option>数据</option>
                <option>年报</option>
                <option>资料</option>
              </select>
              <button type="submit" disabled={busy || !fileDraft}>
                上传入库
              </button>
            </div>
          </form>
        </section>
      </aside>

      <section className="center-pane">
        <div className="topbar">
          <div className="view-switch">
            <button
              type="button"
              className={viewMode === "section" ? "active" : ""}
              onClick={() => setViewMode("section")}
            >
              章节编辑
            </button>
            <button
              type="button"
              className={viewMode === "preview" ? "active" : ""}
              onClick={() => setViewMode("preview")}
            >
              整份报告预览
            </button>
          </div>
          <label className="section-select">
            章节
            <select value={activeSection?.id} onChange={(event) => setActiveSectionId(event.target.value)}>
              {workspace.sections.map((section) => (
                <option key={section.id} value={section.id}>
                  {section.level === 2 ? "  " : ""}
                  {sectionLabel(section)}
                </option>
              ))}
            </select>
          </label>
          <span className="status-pill">{status}</span>
        </div>

        {viewMode === "section" && activeSection ? (
          <article className="document">
            <div className="doc-header">
              <p className="doc-kicker">投行内部行业风险研究</p>
              <h1>{sectionLabel(activeSection)}</h1>
              <span className={activeSection.confirmed ? "version-state confirmed" : "version-state"}>
                {activeSection.confirmed ? "当前版本已确认" : "当前版本待确认"}
              </span>
            </div>

            {generating || activeSection.current_version ? (
              <div className={generating ? "content-preview streaming" : "content-preview"}>
                {renderContentWithCitations(
                  generating ? liveContent : (activeSection.current_version?.content ?? ""),
                )}
              </div>
            ) : (
              <div className="empty-document">
                <h2>尚未生成章节内容</h2>
                <p>上传并勾选已入库资料后，在右侧点击“生成当前章节”。</p>
              </div>
            )}

            <textarea
              className="content-editor"
              value={contentDraft}
              onChange={(event) => setContentDraft(event.target.value)}
              placeholder="生成后可在这里直接编辑章节正文。"
            />
          </article>
        ) : (
          <article className="document report-preview">
            <div className="doc-header">
              <p className="doc-kicker">完整交付预览</p>
              <h1>{workspace.project.name}</h1>
              <span className="version-state">{previewIssues.length ? "导出前需复核" : "可模拟导出"}</span>
            </div>

            <div className="check-strip">
              {previewIssues.length ? previewIssues.map((issue) => <span key={issue}>{issue}</span>) : <span>所有一级章节已生成并确认</span>}
            </div>

            {workspace.sections.map((section) => (
              <section key={section.id} className={section.level === 1 ? "preview-section major" : "preview-section"}>
                <h2>{sectionLabel(section)}</h2>
                {section.current_version ? (
                  <p>{renderContentWithCitations(section.current_version.content)}</p>
                ) : (
                  <p className="missing-copy">本章节尚未生成内容。</p>
                )}
              </section>
            ))}
          </article>
        )}
      </section>

      <aside className="right-pane">
        {activeSection && (
          <>
            <section className="action-panel">
              <p className="eyebrow">Prompt 操作</p>
              <h2>{activeSection.title}</h2>
              <textarea
                className="prompt-editor"
                value={promptDraft}
                onChange={(event) => setPromptDraft(event.target.value)}
              />
              <details className="settings-panel">
                <summary>System Prompt</summary>
                <textarea
                  className="system-prompt-editor"
                  value={systemPromptDraft}
                  onChange={(event) => setSystemPromptDraft(event.target.value)}
                />
                <div className="button-row">
                  <button type="button" onClick={() => void restoreDefaultSystemPrompt()}>
                    恢复默认
                  </button>
                  <button type="button" onClick={() => setStatus("System Prompt 已更新，将用于下一次生成")}>
                    应用修改
                  </button>
                </div>
              </details>
              <div className="button-row">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() =>
                    runAction(
                      () =>
                        requestJson<Workspace>(`/api/sections/${activeSection.id}/prompt`, {
                          method: "PATCH",
                          body: JSON.stringify({ prompt: promptDraft }),
                        }),
                      "Prompt 已保存",
                    )
                  }
                >
                  保存 Prompt
                </button>
                <button type="button" onClick={() => setAssistOpen(true)}>AI 辅助生成</button>
              </div>
              <details className="settings-panel">
                <summary>生成参数</summary>
                <div className="settings-group">
                  <span>生成风格</span>
                  <div className="segmented-control">
                    <button type="button" className={generationSettings.style === "strict" ? "active" : ""} onClick={() => updateStyle("strict")}>严谨</button>
                    <button type="button" className={generationSettings.style === "balanced" ? "active" : ""} onClick={() => updateStyle("balanced")}>平衡</button>
                    <button type="button" className={generationSettings.style === "creative" ? "active" : ""} onClick={() => updateStyle("creative")}>发散</button>
                  </div>
                </div>
                <div className="settings-group">
                  <span>输出长度</span>
                  <div className="segmented-control">
                    <button type="button" className={generationSettings.length === "short" ? "active" : ""} onClick={() => updateLength("short")}>短</button>
                    <button type="button" className={generationSettings.length === "medium" ? "active" : ""} onClick={() => updateLength("medium")}>中</button>
                    <button type="button" className={generationSettings.length === "long" ? "active" : ""} onClick={() => updateLength("long")}>长</button>
                  </div>
                </div>
                <label className="switch-row">
                  <input
                    type="checkbox"
                    checked={generationSettings.stream}
                    onChange={(event) => setGenerationSettings({ ...generationSettings, stream: event.target.checked })}
                  />
                  流式输出
                </label>
                <label className="switch-row">
                  <input
                    type="checkbox"
                    checked={generationSettings.use_parent_context}
                    onChange={(event) =>
                      setGenerationSettings({ ...generationSettings, use_parent_context: event.target.checked })
                    }
                  />
                  Parent 上下文
                </label>
                <div className="settings-grid">
                  <label>
                    筛选片段数
                    <input
                      type="number"
                      min="1"
                      max="40"
                      value={generationSettings.retrieval_top_k}
                      onChange={(event) =>
                        setGenerationSettings({ ...generationSettings, retrieval_top_k: Number(event.target.value) })
                      }
                    />
                  </label>
                  <label>
                    每文档最多片段
                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={generationSettings.per_document_limit}
                      onChange={(event) =>
                        setGenerationSettings({ ...generationSettings, per_document_limit: Number(event.target.value) })
                      }
                    />
                  </label>
                  <label>
                    temperature
                    <input
                      type="number"
                      min="0"
                      max="1.99"
                      step="0.1"
                      value={generationSettings.temperature}
                      onChange={(event) =>
                        setGenerationSettings({ ...generationSettings, temperature: Number(event.target.value) })
                      }
                    />
                  </label>
                  <label>
                    max_tokens
                    <input
                      type="number"
                      min="300"
                      max="6000"
                      step="100"
                      value={generationSettings.max_tokens}
                      onChange={(event) =>
                        setGenerationSettings({ ...generationSettings, max_tokens: Number(event.target.value) })
                      }
                    />
                  </label>
                </div>
              </details>
              <button
                type="button"
                className="primary-action"
                disabled={busy || selectedIndexedCount === 0}
                onClick={() => void generateCurrentSection()}
              >
                {generating ? "生成中…" : "生成当前章节"}
              </button>
              {selectedIndexedCount === 0 ? <p className="panel-copy">需要先上传并勾选已入库资料。</p> : null}
            </section>

            <section className="action-panel">
              <p className="eyebrow">正文操作</p>
              <div className="button-row">
                <button
                  type="button"
                  disabled={busy || !contentDraft.trim()}
                  onClick={() =>
                    runAction(
                      () =>
                        requestJson<Workspace>(`/api/sections/${activeSection.id}/manual-edit`, {
                          method: "POST",
                          body: JSON.stringify({ content: contentDraft }),
                        }),
                      "人工编辑版本已保存",
                    )
                  }
                >
                  保存编辑版本
                </button>
                <button
                  type="button"
                  disabled={busy || !activeSection.current_version}
                  onClick={() =>
                    runAction(
                      () =>
                        requestJson<Workspace>(`/api/sections/${activeSection.id}/confirm`, {
                          method: "POST",
                        }),
                      "当前章节版本已确认",
                    )
                  }
                >
                  确认当前版本
                </button>
              </div>
              <button
                type="button"
                className="danger-action"
                disabled={busy || !activeSection.current_version}
                onClick={() => {
                  const confirmed = window.confirm("清空当前章节会删除该章节所有版本，确认继续？");
                  if (!confirmed) return;
                  runAction(
                    () =>
                      requestJson<Workspace>(`/api/sections/${activeSection.id}/clear`, {
                        method: "POST",
                      }),
                    "当前章节内容已清空",
                  );
                }}
              >
                清空当前章节
              </button>
            </section>

            <section className="action-panel">
              <div className="section-heading">
                <span>章节版本</span>
                <small>{activeSection.versions.length} 个版本</small>
              </div>
              <div className="version-list">
                {activeSection.versions.length ? (
                  activeSection.versions.map((version) => (
                    <button
                      key={version.id}
                      type="button"
                      className={version.id === activeSection.current_version_id ? "version-item active" : "version-item"}
                      onClick={() =>
                        runAction(
                          () =>
                            requestJson<Workspace>(`/api/sections/${activeSection.id}/select-version`, {
                              method: "POST",
                              body: JSON.stringify({ version_id: version.id }),
                            }),
                          "已切换当前版本",
                        )
                      }
                    >
                      <span>V{version.version_number} · {version.source}</span>
                      <small>{formatTime(version.created_at)}</small>
                      <em>{version.summary}</em>
                    </button>
                  ))
                ) : (
                  <p className="panel-copy">当前章节还没有版本。</p>
                )}
              </div>
            </section>
          </>
        )}

        <section className="action-panel">
          <p className="eyebrow">交付导出</p>
          <div className="export-buttons">
            <button
              type="button"
              onClick={() =>
                runAction(
                  () =>
                    requestJson<Workspace>("/api/exports", {
                      method: "POST",
                      body: JSON.stringify({ format: "Word" }),
                    }),
                  "已生成 Word 模拟导出记录",
                )
              }
            >
              导出 Word
            </button>
            <button
              type="button"
              onClick={() =>
                runAction(
                  () =>
                    requestJson<Workspace>("/api/exports", {
                      method: "POST",
                      body: JSON.stringify({ format: "PDF" }),
                    }),
                  "已生成 PDF 模拟导出记录",
                )
              }
            >
              导出 PDF
            </button>
          </div>
          <div className="export-list">
            {workspace.export_records.slice(0, 4).map((record) => (
              <div key={record.id} className="export-record">
                <strong>{record.format}</strong>
                <span>{record.status}</span>
                <small>{formatTime(record.created_at)}</small>
              </div>
            ))}
          </div>
        </section>

        <section className="action-panel">
          <div className="section-heading">
            <span>模型生成日志</span>
            <small>最近 {workspace.ai_call_logs.length} 次</small>
          </div>
          <div className="usage-list">
            {workspace.ai_call_logs.slice(0, 8).map((log) => (
              <div key={log.id} className="usage-item">
                <strong>{log.operation}</strong>
                <span>{log.model}</span>
                <small>{formatTime(log.created_at)}</small>
                <dl>
                  <div>
                    <dt>输入</dt>
                    <dd>{tokenText(log.prompt_tokens)}</dd>
                  </div>
                  <div>
                    <dt>输出</dt>
                    <dd>{tokenText(log.completion_tokens)}</dd>
                  </div>
                  <div>
                    <dt>合计</dt>
                    <dd>{tokenText(log.total_tokens)}</dd>
                  </div>
                </dl>
                <button type="button" onClick={() => setSelectedLog(log)}>
                  查看完整日志
                </button>
              </div>
            ))}
            {!workspace.ai_call_logs.length ? <p className="panel-copy">还没有模型调用记录。</p> : null}
          </div>
        </section>
      </aside>

      {assistOpen && activeSection && (
        <div className="modal-backdrop" role="presentation">
          <form
            className="modal"
            onSubmit={(event) => {
              event.preventDefault();
              runAction(async () => {
                const result = await requestJson<{ prompt: string }>(`/api/sections/${activeSection.id}/enhance-prompt`, {
                  method: "POST",
                  body: JSON.stringify(assistDraft),
                });
                setPromptDraft(result.prompt);
                setAssistOpen(false);
                return workspace;
              }, "AI 辅助 Prompt 已生成");
            }}
          >
            <div className="modal-header">
              <div>
                <p className="eyebrow">AI 辅助生成 Prompt</p>
                <h2>{activeSection.title}</h2>
              </div>
              <button type="button" onClick={() => setAssistOpen(false)}>关闭</button>
            </div>
            <div className="assist-grid">
              <label>
                写作目标
                <textarea value={assistDraft.objective} onChange={(event) => setAssistDraft({ ...assistDraft, objective: event.target.value })} />
              </label>
              <label>
                重点问题
                <textarea value={assistDraft.key_questions} onChange={(event) => setAssistDraft({ ...assistDraft, key_questions: event.target.value })} />
              </label>
              <label>
                地域范围
                <input value={assistDraft.geography} onChange={(event) => setAssistDraft({ ...assistDraft, geography: event.target.value })} />
              </label>
              <label>
                风险维度
                <input value={assistDraft.risk_dimensions} onChange={(event) => setAssistDraft({ ...assistDraft, risk_dimensions: event.target.value })} />
              </label>
              <label>
                语气和篇幅
                <input value={assistDraft.tone_length} onChange={(event) => setAssistDraft({ ...assistDraft, tone_length: event.target.value })} />
              </label>
              <label>
                排除内容
                <input value={assistDraft.exclusions} onChange={(event) => setAssistDraft({ ...assistDraft, exclusions: event.target.value })} />
              </label>
            </div>
            <button type="submit" className="primary-action" disabled={busy}>生成 Prompt</button>
          </form>
        </div>
      )}

      {previewDocument && (
        <div className="modal-backdrop" role="presentation">
          <section className="modal document-preview-modal">
            <div className="modal-header">
              <div>
                <p className="eyebrow">参考文档预览</p>
                <h2>{previewDocument.name}</h2>
              </div>
              <button type="button" onClick={() => setPreviewDocument(null)}>关闭</button>
            </div>
            <div className="preview-meta">
              <span>{previewDocument.type}</span>
              <span>{statusLabel(previewDocument)}</span>
              <span>{formatTime(previewDocument.indexed_at)}</span>
            </div>
            <section className={`quality-card ${previewDocument.quality_metrics.status}`}>
              <div className="quality-header">
                <div>
                  <p className="eyebrow">本次 Chunk 质量</p>
                  <strong>{previewDocument.quality_metrics.score}/100</strong>
                </div>
                {previewDocument.file_path ? (
                  <button type="button" disabled={busy} onClick={() => reindexDocument(previewDocument)}>
                    重新解析入库
                  </button>
                ) : null}
              </div>
              <div className="quality-grid">
                <span>数量 <strong>{previewDocument.quality_metrics.chunk_count}</strong></span>
                <span>平均长度 <strong>{previewDocument.quality_metrics.avg_chars}</strong></span>
                <span>最短 <strong>{previewDocument.quality_metrics.min_chars}</strong></span>
                <span>最长 <strong>{previewDocument.quality_metrics.max_chars}</strong></span>
                <span>标题覆盖 <strong>{percentText(previewDocument.quality_metrics.heading_coverage)}</strong></span>
                <span>Parent 覆盖 <strong>{percentText(previewDocument.quality_metrics.parent_coverage)}</strong></span>
                <span>短片段 <strong>{previewDocument.quality_metrics.short_chunks}</strong></span>
                <span>超长片段 <strong>{previewDocument.quality_metrics.long_chunks}</strong></span>
                <span>表格片段 <strong>{previewDocument.quality_metrics.table_chunks}</strong></span>
              </div>
              {previewDocument.quality_metrics.warnings.length ? (
                <div className="quality-warnings">
                  {previewDocument.quality_metrics.warnings.map((warning) => (
                    <small key={warning}>{warning}</small>
                  ))}
                </div>
              ) : (
                <p className="panel-copy">当前 chunk 长度、标题路径和 parent 上下文覆盖较稳定。</p>
              )}
            </section>
            <p className="document-summary">{previewDocument.summary}</p>
            <div className="chunk-preview-list">
              {previewDocument.chunks.length ? (
                previewDocument.chunks.map((chunk) => (
                  <article key={chunk.id} className="chunk-preview-item">
                    <div>
                      <strong>片段 {chunk.chunk_index + 1}</strong>
                      <small>{chunk.metadata.block_type ?? "text"} · {chunk.source_locator}</small>
                    </div>
                    {chunk.metadata.heading_path ? (
                      <span className="chunk-path">{chunk.metadata.heading_path}</span>
                    ) : null}
                    <p>{chunk.text}</p>
                    {chunk.metadata.parent_id ? (
                      <details className="parent-context">
                        <summary>查看 parent 上下文</summary>
                        <p>{chunk.metadata.parent_text}</p>
                      </details>
                    ) : null}
                  </article>
                ))
              ) : (
                <p className="panel-copy">这份文档还没有解析片段。</p>
              )}
            </div>
          </section>
        </div>
      )}

      {selectedLog && (
        <div className="modal-backdrop" role="presentation">
          <section className="modal model-log-modal">
            <div className="modal-header">
              <div>
                <p className="eyebrow">模型调用完整日志</p>
                <h2>{selectedLog.operation}</h2>
              </div>
              <button type="button" onClick={() => setSelectedLog(null)}>关闭</button>
            </div>
            <div className="log-summary">
              <span>{selectedLog.model}</span>
              <span>{formatTime(selectedLog.created_at)}</span>
              <span>输入 {tokenText(selectedLog.prompt_tokens)}</span>
              <span>输出 {tokenText(selectedLog.completion_tokens)}</span>
              <span>合计 {tokenText(selectedLog.total_tokens)}</span>
            </div>

            {selectedLog.metadata.generation_settings ? (
              <section className="log-section">
                <h3>生成参数</h3>
                <pre>{metadataText(selectedLog.metadata.generation_settings)}</pre>
              </section>
            ) : null}

            {selectedLog.metadata.llm_api_request ? (
              <section className="log-section">
                <h3>LLM 接口请求 Payload</h3>
                <pre>{metadataText(selectedLog.metadata.llm_api_request)}</pre>
              </section>
            ) : null}

            {selectedLog.metadata.llm_api_response ? (
              <section className="log-section">
                <h3>LLM 接口响应 Payload</h3>
                <pre>{metadataText(selectedLog.metadata.llm_api_response)}</pre>
              </section>
            ) : null}

            {promptMessages(selectedLog).length ? (
              <section className="log-section">
                <h3>Prompt 结构</h3>
                {promptMessages(selectedLog).map((message, index) => (
                  <article key={`${message.role}-${index}`} className="prompt-message">
                    <strong>{message.role}</strong>
                    <pre>{message.content}</pre>
                  </article>
                ))}
              </section>
            ) : null}

            {retrievedLogChunks(selectedLog).length ? (
              <section className="log-section">
                <h3>检索引用</h3>
                <div className="log-reference-list">
                  {retrievedLogChunks(selectedLog).map((chunk, index) => (
                    <article key={`${chunk.id}-${index}`} className="log-reference-item">
                      <div>
                        <strong>{chunk.document_name ?? chunk.id}</strong>
                        <small>{chunk.document_type ?? "资料"} · {chunk.source_locator ?? "位置未记录"}</small>
                      </div>
                      {chunk.heading_path ? <span>{chunk.heading_path}</span> : null}
                      <p>{chunk.text}</p>
                      {chunk.parent_text ? (
                        <details>
                          <summary>parent_text</summary>
                          <p>{chunk.parent_text}</p>
                        </details>
                      ) : null}
                    </article>
                  ))}
                </div>
              </section>
            ) : null}

            {selectedLog.metadata.citation_map ? (
              <section className="log-section">
                <h3>引用映射</h3>
                <p className="log-explain">{metadataText(selectedLog.metadata.citation_rendering_rule)}</p>
                <pre>{metadataText(selectedLog.metadata.citation_map)}</pre>
              </section>
            ) : null}

            {selectedLog.metadata.output_content ? (
              <section className="log-section">
                <h3>LLM 输出</h3>
                <pre>{metadataText(selectedLog.metadata.output_content)}</pre>
              </section>
            ) : null}

            <details className="log-section">
              <summary>调试 Metadata</summary>
              <pre>{metadataText(selectedLog.metadata)}</pre>
            </details>
          </section>
        </div>
      )}
    </main>
  );
}

export default App;
