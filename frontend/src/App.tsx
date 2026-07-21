import { FormEvent, Fragment, ReactNode, useEffect, useMemo, useState } from "react";

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
  export_issues: string[];
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

type TableColumnDraft = {
  name: string;
  description: string;
};

type TableCandidatePreview = {
  parsed: boolean;
  columns: string[];
  rows: string[][];
  truncated: boolean;
  total_rows: number;
};

type TableCandidate = {
  chunk_id: string;
  document_id: string;
  document_name: string;
  source_locator: string;
  preview_text: string;
  table_preview: TableCandidatePreview | null;
  block_type: string;
  score?: number;
};

type TableDraft = {
  id: string;
  mode: "synthesize" | "verbatim";
  title: string;
  columns: TableColumnDraft[];
  notes: string;
  description: string;
  confirmedChunkId: string;
  candidates: TableCandidate[];
  searching: boolean;
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

type Screen = "home" | "create" | "report-center" | "management" | "workspace";
type WorkspaceMode = "editor" | "preview" | "ai-map";

const emptyWorkspace: Workspace = {
  project: { id: "", name: "", industry: "", year: "", language: "" },
  documents: [],
  sections: [],
  citations: {},
  ai_call_logs: [],
  export_records: [],
  export_issues: [],
};

const fallbackSystemPrompt = `你是一个面向银行内部使用的行业研究报告生成助手，目标是生成一份关于“消费电子行业风险分析”的中文书面报告，用于商业银行的授信审批、行业限额管理、内部评级和风险政策支持等场景。报告必须满足以下全局约束，并在所有章节中保持风格与逻辑的一致性。

### 一、整体定位与语气

- 报告定位：面向商业银行的风险管理、授信审批、行业研究等专业读者，服务于信用风险识别、额度与结构设计和内部政策制定。
- 语气要求：使用正式、专业、审慎的书面中文，避免口语化、情绪化或营销式表达；保持冷静、客观、中性，不使用夸张和煽动性措辞。
- 立场与视角：全文站在银行风险管理与授信决策视角，而不是企业自我宣传或纯投资研究视角；任何结论和判断，都需要回到“对信用风险和授信安全性的影响”这一主线。

### 二、结构与逻辑一致性

- 逻辑主线：所有章节内容都应围绕以下链条进行隐含或显式的叙述：“宏观与政策环境 → 行业景气与结构 → 企业商业模式与运营财务特征 → 信用风险与银行资产质量/资本占用”。
- 章节之间的衔接：即使不显式写出章节标题，段落内容也应在逻辑上相互支撑，避免前后结论冲突；后面的分析（财务、授信建议等）要以此前的行业和风险特征为基础，而不是孤立罗列。
- 论证方式：优先采用“现象描述 → 原因分析 → 风险影响 → 银行关注点/应对方向”的结构展开论述，每个核心段落尽量保持这一模式。

### 三、内容边界与信息使用规范

- 行业边界：报告默认涵盖消费电子整机、零部件与模组、代工制造、品牌渠道与电商等相关业务，但不深入到与消费电子关联度较低的其他电子或硬件行业。
- 数据与事实使用：以趋势、区间和定性描述为主，避免编造具体年份、精确数据或引用真实公司未公开信息；可以使用“近年来”“过去几个年度”“整体呈现增长/放缓趋势”等模糊时间与趋势表达。

### 四、风险视角与评估框架

- 风险类型覆盖：全文需要系统覆盖宏观环境风险、行业结构性风险、商业模式与盈利模式风险、财务与信用风险、运营与供应链风险、技术与产品生命周期风险、法律合规与 ESG 风险等维度。
- 关注重点：在每一类风险的分析中，都要明确指出其对企业现金流、偿债能力、授信回收风险、不良率和资本占用的影响路径，而不仅仅停留在行业现象层面。
- 风险程度与方向：采用“风险较高/中等/偏低”“波动较大/相对稳定”“集中度较高/分散”等方向性判断，不要求也不允许给出精确量化评级结果，但要避免模糊到无法指导授信实践。

### 五、写作风格与表达规范

- 语言规范：使用规范金融与风险管理术语，如“信用风险、违约概率、资产质量、资本占用、授信政策、额度管理、担保与抵质押、营运资本、现金流压力”等；避免网络用语和口语化表达。
- 段落结构：段落内部保持信息密度适中，不使用长句堆砌导致理解困难；对于复杂内容可适度使用分句和分段，但保持整体连贯性与严谨性。
- 价值判断：所有判断应基于行业通行认知或合理推理，避免出现绝对化表达（如“必然”“肯定”“毫无风险”），改用“可能”“倾向于”“在一定条件下”等审慎措辞。

### 六、风格一致性与可读性

- 全文一致性：在不同章节生成内容时，保持语气、用词、风险视角和逻辑主线一致，不出现明显风格断裂或立场变化。
- 信息密度控制：保持适度高的信息密度和分析深度，但避免堆砌概念和重复表述；对同一风险点不在不同章节无意义重复，而是通过不同角度加深理解。
- 可读性：内容面向具备金融与风控背景的读者，不需要对基础概念进行过度科普，但应确保段落结构清晰，方便授信与风险管理人员快速获取结论与重点。

在生成具体章节内容时，始终遵守上述全局约束，并默认读者是银行内部的风险管理与授信专业人员，以消费电子行业的信用风险识别和授信决策支持为最核心目标。`;

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

function formatApiDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object" && "msg" in item) return String((item as { msg: unknown }).msg);
        return "";
      })
      .filter(Boolean);
    if (parts.length) return parts.join("；");
  }
  if (detail && typeof detail === "object") {
    try {
      return JSON.stringify(detail);
    } catch {
      /* ignore */
    }
  }
  return fallback;
}

function networkErrorMessage(error: unknown, fallback: string): string {
  if (!(error instanceof Error)) return fallback;
  const message = error.message || "";
  if (
    error.name === "TypeError" ||
    /failed to fetch|networkerror|load failed|fetch failed/i.test(message)
  ) {
    return "后端未连接或无法访问，请确认本地 API（8000）已启动";
  }
  return message || fallback;
}

async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
  const body = options?.body;
  const isFormData = body instanceof FormData;
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: isFormData ? options?.headers : { "Content-Type": "application/json", ...(options?.headers ?? {}) },
    });
  } catch (error) {
    throw new Error(networkErrorMessage(error, "后端未连接或无法访问"));
  }

  if (!response.ok) {
    const text = await response.text();
    let message = text;
    try {
      const data = JSON.parse(text);
      message = formatApiDetail(data.detail, text);
    } catch {
      message = text;
    }
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

type GenerateStreamEvent =
  | { event: "status"; data: { phase?: string; message?: string; content?: string; table_warnings?: string[] } }
  | { event: "delta"; data: { content: string } }
  | { event: "done"; data: { workspace: Workspace } }
  | { event: "error"; data: { detail: string } };

type GenerationStep = {
  phase: string;
  message: string;
};

function renderTableCandidatePreview(candidate: TableCandidate) {
  const preview = candidate.table_preview;
  if (!preview?.parsed || !preview.columns.length) {
    return candidate.preview_text ? <em className="candidate-text-preview">{candidate.preview_text}</em> : null;
  }

  return (
    <div className="candidate-table-wrap">
      <table className="candidate-table-preview">
        <thead>
          <tr>
            {preview.columns.map((column, index) => (
              <th key={`${candidate.chunk_id}-col-${index}`}>{column || `列 ${index + 1}`}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {preview.rows.map((row, rowIndex) => (
            <tr key={`${candidate.chunk_id}-row-${rowIndex}`}>
              {preview.columns.map((_, columnIndex) => (
                <td key={`${candidate.chunk_id}-cell-${rowIndex}-${columnIndex}`}>
                  {row[columnIndex] ?? ""}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {preview.truncated ? (
        <small className="candidate-table-note">仅展示前 {preview.rows.length} 行，共 {preview.total_rows} 行</small>
      ) : null}
    </div>
  );
}

function isMarkdownTableSeparator(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed.includes("-") || !trimmed.includes("|")) return false;
  return /^[\s|:.-]+$/.test(trimmed);
}

function isMarkdownTableRow(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed.includes("|")) return false;
  if (isMarkdownTableSeparator(trimmed)) return true;
  return trimmed.split("|").filter((part) => part.trim().length > 0).length >= 2;
}

function splitMarkdownTableCells(line: string): string[] {
  let trimmed = line.trim();
  if (trimmed.startsWith("|")) trimmed = trimmed.slice(1);
  if (trimmed.endsWith("|")) trimmed = trimmed.slice(0, -1);
  return trimmed.split("|").map((cell) => cell.trim());
}

type ContentBlock =
  | { type: "text"; text: string }
  | { type: "table"; headers: string[]; rows: string[][]; caption?: string };

function isTableCaptionLine(line: string): string | null {
  const match = line.trim().match(/^\[\[表题[：:](.+?)\]\]$/);
  return match ? match[1].trim() : null;
}

function peelTrailingHeading(lines: string[]): { remaining: string[]; caption: string | null } {
  let end = lines.length - 1;
  while (end >= 0 && !lines[end].trim()) end -= 1;
  if (end < 0) return { remaining: lines, caption: null };

  const match = lines[end].trim().match(/^#{1,6}\s+(.+)$/);
  if (!match) return { remaining: lines, caption: null };

  const remaining = lines.slice(0, end);
  while (remaining.length && !remaining[remaining.length - 1].trim()) remaining.pop();
  return { remaining, caption: match[1].trim() };
}

function splitContentBlocks(content: string): ContentBlock[] {
  const lines = content.split("\n");
  const blocks: ContentBlock[] = [];
  let textBuffer: string[] = [];
  let index = 0;

  const flushText = () => {
    if (!textBuffer.length) return;
    blocks.push({ type: "text", text: textBuffer.join("\n") });
    textBuffer = [];
  };

  while (index < lines.length) {
    const captionOnly = isTableCaptionLine(lines[index]);
    if (captionOnly) {
      const last = blocks[blocks.length - 1];
      if (last?.type === "table" && !last.caption) {
        last.caption = captionOnly;
      }
      index += 1;
      continue;
    }

    if (isMarkdownTableRow(lines[index])) {
      const tableLines: string[] = [];
      while (index < lines.length && isMarkdownTableRow(lines[index])) {
        tableLines.push(lines[index]);
        index += 1;
      }
      const dataLines = tableLines.filter((line) => !isMarkdownTableSeparator(line));
      if (dataLines.length >= 2) {
        const peeled = peelTrailingHeading(textBuffer);
        textBuffer = peeled.remaining;
        flushText();

        let caption = peeled.caption;
        while (index < lines.length && !lines[index].trim()) index += 1;
        if (index < lines.length) {
          const markerCaption = isTableCaptionLine(lines[index]);
          if (markerCaption) {
            caption = markerCaption;
            index += 1;
          }
        }

        const [headerLine, ...bodyLines] = dataLines;
        blocks.push({
          type: "table",
          headers: splitMarkdownTableCells(headerLine),
          rows: bodyLines.map(splitMarkdownTableCells),
          caption: caption || undefined,
        });
      } else {
        textBuffer.push(...tableLines);
      }
      continue;
    }

    textBuffer.push(lines[index]);
    index += 1;
  }

  flushText();
  return blocks;
}

function createTableDraft(mode: TableDraft["mode"] = "synthesize"): TableDraft {
  return {
    id: `table_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
    mode,
    title: "",
    columns: [{ name: "", description: "" }],
    notes: "",
    description: "",
    confirmedChunkId: "",
    candidates: [],
    searching: false,
  };
}

function buildTablesPayload(tables: TableDraft[]) {
  return tables.map((table) => {
    if (table.mode === "verbatim") {
      return {
        mode: "verbatim" as const,
        title: table.title.trim(),
        description: table.description.trim(),
        confirmed_chunk_id: table.confirmedChunkId,
      };
    }
    return {
      mode: "synthesize" as const,
      title: table.title.trim(),
      columns: table.columns
        .filter((column) => column.name.trim())
        .map((column) => ({ name: column.name.trim(), description: column.description.trim() })),
      notes: table.notes.trim(),
    };
  });
}

function validateTableDrafts(includeTable: boolean, tables: TableDraft[]): string | null {
  if (!includeTable) return null;
  if (!tables.length) return "请至少添加一张表格，或关闭「需要添加表格」。";
  for (const [index, table] of tables.entries()) {
    if (table.mode === "verbatim") {
      if (!table.title.trim()) return `表格 ${index + 1} 请填写标题。`;
      if (!table.description.trim()) return `表格 ${index + 1} 请填写要检索的原表描述。`;
      if (!table.confirmedChunkId) return `表格 ${index + 1} 请先检索并确认引用的原表。`;
    } else {
      if (!table.title.trim()) return `表格 ${index + 1} 请填写标题。`;
      if (!table.columns.some((column) => column.name.trim())) return `表格 ${index + 1} 至少需要 1 列定义。`;
    }
  }
  return null;
}

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

function blockTypeLabel(value?: string) {
  if (value === "text") return "文本";
  if (value === "table") return "表格";
  if (value === "heading") return "标题";
  return value ?? "文本";
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
  const [screen, setScreen] = useState<Screen>("home");
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>("editor");
  const [activeSectionId, setActiveSectionId] = useState("");
  const [promptDraft, setPromptDraft] = useState("");
  const [systemPromptDraft, setSystemPromptDraft] = useState(fallbackSystemPrompt);
  const [contentDraft, setContentDraft] = useState("");
  const [projectDraft, setProjectDraft] = useState<Project>(emptyWorkspace.project);
  const [fileDraft, setFileDraft] = useState<File | null>(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [fileType, setFileType] = useState("研报");
  const [assistContent, setAssistContent] = useState("");
  const [assistOpen, setAssistOpen] = useState(false);
  const [assistGenerating, setAssistGenerating] = useState(false);
  const [previewDocument, setPreviewDocument] = useState<DocumentPreview | null>(null);
  const [selectedLog, setSelectedLog] = useState<AICallLog | null>(null);
  const [generationSettings, setGenerationSettings] = useState<GenerationSettings>(defaultGenerationSettings);
  const [includeTable, setIncludeTable] = useState(false);
  const [tableDrafts, setTableDrafts] = useState<TableDraft[]>([]);
  const [liveContent, setLiveContent] = useState("");
  const [generating, setGenerating] = useState(false);
  const [generationSteps, setGenerationSteps] = useState<GenerationStep[]>([]);
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

  const previewIssues = workspace.export_issues ?? [];
  const generatedSections = workspace.sections.filter((section) => section.current_version).length;
  const confirmedSections = workspace.sections.filter((section) => section.confirmed).length;
  const unconfirmedSections = workspace.sections.filter((section) => section.current_version && !section.confirmed);
  const indexedDocuments = workspace.documents.filter((document) => document.parse_status === "indexed");
  const latestExport = workspace.export_records[0];
  const latestLog = workspace.ai_call_logs[0];


  const tableValidationError = useMemo(
    () => validateTableDrafts(includeTable, tableDrafts),
    [includeTable, tableDrafts],
  );

  function updateTableDraft(tableId: string, patch: Partial<TableDraft>) {
    setTableDrafts((current) => current.map((table) => (table.id === tableId ? { ...table, ...patch } : table)));
  }

  function removeTableDraft(tableId: string) {
    setTableDrafts((current) => current.filter((table) => table.id !== tableId));
  }

  async function searchTableCandidates(tableId: string) {
    if (!activeSection) return;
    const table = tableDrafts.find((item) => item.id === tableId);
    if (!table?.description.trim()) {
      setStatus("请先填写原表描述");
      return;
    }
    updateTableDraft(tableId, { searching: true, candidates: [], confirmedChunkId: "" });
    try {
      const data = await requestJson<{ candidates: TableCandidate[] }>(
        `/api/sections/${activeSection.id}/table-candidates`,
        {
          method: "POST",
          body: JSON.stringify({
            description: table.description.trim(),
            reference_ids: selectedDocumentIds,
          }),
        },
      );
      updateTableDraft(tableId, { searching: false, candidates: data.candidates });
      setStatus(data.candidates.length ? `表格候选已返回 ${data.candidates.length} 条` : "未找到匹配的表格候选");
    } catch (error) {
      updateTableDraft(tableId, { searching: false });
      setStatus(networkErrorMessage(error, "检索表格候选失败"));
    }
  }

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

    const tableError = validateTableDrafts(includeTable, tableDrafts);
    if (tableError) {
      setStatus(tableError);
      return;
    }

    const payload = {
      prompt: promptDraft,
      system_prompt: systemPromptDraft,
      reference_ids: selectedDocumentIds,
      tables: includeTable ? buildTablesPayload(tableDrafts) : [],
      ...generationSettings,
    };

    if (!generationSettings.stream) {
      await runAction(
        () =>
          requestJson<Workspace>(`/api/sections/${activeSection.id}/generate`, {
            method: "POST",
            body: JSON.stringify(payload),
          }),
        "已生成新版本",
      );
      return;
    }

    setBusy(true);
    setGenerating(true);
    setLiveContent("");
    setContentDraft("");
    setGenerationSteps([{ phase: "prepare", message: "开始生成…" }]);
    setStatus("开始生成…");

    try {
      let response: Response;
      try {
        response = await fetch(`${API_BASE}/api/sections/${activeSection.id}/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      } catch (error) {
        throw new Error(networkErrorMessage(error, "后端未连接或无法访问"));
      }

      if (!response.ok) {
        const text = await response.text();
        let message = text;
        try {
          const data = JSON.parse(text);
          message = formatApiDetail(data.detail, text);
        } catch {
          message = text;
        }
        throw new Error(message || `Request failed: ${response.status}`);
      }

      let streamed = "";
      let tableWarningCount = 0;
      await consumeGenerateStream(response, (event) => {
        if (event.event === "status") {
          const warnings = event.data.table_warnings ?? [];
          const message = event.data.message ?? "处理中…";
          const phase = event.data.phase ?? "status";
          if (warnings.length) {
            tableWarningCount = warnings.length;
            setStatus(`${message}（${warnings.length} 处待核实）`);
            setGenerationSteps((prev) => [...prev, { phase, message: `${message}（${warnings.length} 处待核实）` }]);
          } else {
            setStatus(message);
            setGenerationSteps((prev) => [...prev, { phase, message }]);
          }
          if (event.data.content) {
            setLiveContent(event.data.content);
            setContentDraft(event.data.content);
          }
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
          setGenerationSteps((prev) => [...prev, { phase: "done", message: "章节生成完成" }]);
          setStatus(
            tableWarningCount
              ? `章节已流式生成（${tableWarningCount} 处表格数据待核实）`
              : "章节已流式生成",
          );
        }
      });
    } catch (error) {
      const message = networkErrorMessage(error, "流式生成失败");
      setStatus(message);
      setGenerationSteps((prev) => [...prev, { phase: "error", message }]);
    } finally {
      setGenerating(false);
      setLiveContent("");
      setBusy(false);
    }
  }

  function renderInlineMarkdown(text: string, keyPrefix: string) {
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, index) => {
      const boldMatch = part.match(/^\*\*([^*]+)\*\*$/);
      if (boldMatch) {
        return (
          <strong key={`${keyPrefix}-b-${index}`} className="content-md-strong">
            {boldMatch[1]}
          </strong>
        );
      }
      return <Fragment key={`${keyPrefix}-t-${index}`}>{part}</Fragment>;
    });
  }

  function renderInlineContent(content: string, keyPrefix = "inline") {
    const pattern = /(【ref:[^】]+】|\[ref:[^\]]+\]|<<TABLE:\d+>>|\[\[\?[\s\S]*?\?\]\])/g;
    const parts = content.split(pattern);
    return parts.map((part, index) => {
      const citationMatch = part.match(/^(?:\[ref:([^\]]+)\]|【ref:([^】]+)】)$/);
      if (citationMatch) {
        const chunkId = citationMatch[1] || citationMatch[2];
        const citation = workspace.citations[chunkId];
        return (
          <span key={`${keyPrefix}-cite-${index}`} className="citation" tabIndex={0}>
            [{citation?.document_type ?? "资料"}]
            <span className="citation-popover">
              <strong>{citation?.document_name ?? chunkId}</strong>
              <small>{citation?.source_locator ?? "来源位置待确认"}</small>
              <small>{citation?.text ?? "当前引用暂未找到对应片段。"}</small>
            </span>
          </span>
        );
      }

      const placeholderMatch = part.match(/^<<TABLE:(\d+)>>$/);
      if (placeholderMatch) {
        return (
          <span key={`${keyPrefix}-ph-${index}`} className="table-placeholder">
            【表格 {placeholderMatch[1]} 将在此插入】
          </span>
        );
      }

      const unverifiedMatch = part.match(/^\[\[\?([\s\S]*?)\?\]\]$/);
      if (unverifiedMatch) {
        return (
          <span key={`${keyPrefix}-uv-${index}`} className="table-cell-unverified" title="该单元格未能与引用资料对齐，请人工核实">
            {renderInlineMarkdown(unverifiedMatch[1], `${keyPrefix}-uv-md-${index}`)}
          </span>
        );
      }

      return <Fragment key={`${keyPrefix}-text-${index}`}>{renderInlineMarkdown(part, `${keyPrefix}-md-${index}`)}</Fragment>;
    });
  }

  function renderRichTextBlock(text: string, keyPrefix: string) {
    const lines = text.split("\n");
    const nodes: ReactNode[] = [];
    let paragraph: string[] = [];

    const flushParagraph = (suffix: string) => {
      if (!paragraph.length) return;
      const joined = paragraph.join("\n");
      nodes.push(
        <p key={`${keyPrefix}-p-${suffix}`} className="content-md-p">
          {joined.split("\n").map((line, lineIndex) => (
            <Fragment key={`${keyPrefix}-pl-${suffix}-${lineIndex}`}>
              {lineIndex > 0 ? <br /> : null}
              {renderInlineContent(line, `${keyPrefix}-pl-${suffix}-${lineIndex}`)}
            </Fragment>
          ))}
        </p>,
      );
      paragraph = [];
    };

    lines.forEach((line, lineIndex) => {
      const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
      if (headingMatch) {
        flushParagraph(`pre-${lineIndex}`);
        const level = Math.min(headingMatch[1].length, 4);
        const Tag = `h${level + 1}` as "h2" | "h3" | "h4" | "h5";
        nodes.push(
          <Tag key={`${keyPrefix}-h-${lineIndex}`} className={`content-md-heading content-md-h${level}`}>
            {renderInlineContent(headingMatch[2], `${keyPrefix}-h-${lineIndex}`)}
          </Tag>,
        );
        return;
      }

      if (!line.trim()) {
        flushParagraph(`blank-${lineIndex}`);
        return;
      }

      paragraph.push(line);
    });

    flushParagraph("end");
    return nodes;
  }

  function renderContentWithCitations(content: string) {
    const displayContent = content.replace(/(?:\n|^)[ \t]*资料来源：[^\n]*\s*$/g, "").trimEnd();
    return splitContentBlocks(displayContent).map((block, blockIndex) => {
      if (block.type === "table") {
        const columnCount = Math.max(block.headers.length, ...block.rows.map((row) => row.length), 1);
        return (
          <figure key={`table-block-${blockIndex}`} className="content-table-block">
            <div className="content-table-wrap">
              <table className="content-table">
                <thead>
                  <tr>
                    {Array.from({ length: columnCount }, (_, columnIndex) => (
                      <th key={`th-${blockIndex}-${columnIndex}`}>
                        {renderInlineContent(block.headers[columnIndex] ?? "", `th-${blockIndex}-${columnIndex}`)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {block.rows.map((row, rowIndex) => (
                    <tr key={`tr-${blockIndex}-${rowIndex}`}>
                      {Array.from({ length: columnCount }, (_, columnIndex) => (
                        <td key={`td-${blockIndex}-${rowIndex}-${columnIndex}`}>
                          {renderInlineContent(row[columnIndex] ?? "", `td-${blockIndex}-${rowIndex}-${columnIndex}`)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {block.caption ? <figcaption className="content-table-caption">{block.caption}</figcaption> : null}
          </figure>
        );
      }

      return (
        <div key={`text-block-${blockIndex}`} className="content-md-block">
          {renderRichTextBlock(block.text, `text-${blockIndex}`)}
        </div>
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
    setScreen("workspace");
    setWorkspaceMode("editor");
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

  async function submitPromptAssist(event: FormEvent) {
    event.preventDefault();
    if (!activeSection || !assistContent.trim()) return;

    setAssistGenerating(true);
    setBusy(true);
    try {
      const result = await requestJson<{ prompt: string; workspace: Workspace }>(
        `/api/sections/${activeSection.id}/enhance-prompt`,
        {
          method: "POST",
          body: JSON.stringify({ content: assistContent.trim() }),
        },
      );
      setPromptDraft(result.prompt);
      if (result.workspace) {
        setWorkspace(result.workspace);
        setProjectDraft(result.workspace.project);
      }
      setAssistOpen(false);
      setAssistContent("");
      setStatus("Prompt 已生成，请确认后保存");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Prompt 生成失败");
    } finally {
      setAssistGenerating(false);
      setBusy(false);
    }
  }


  function renderTopbar() {
    const portalScreen = screen !== "workspace";
    return (
      <header className="app-topbar">
        <div className="brand">
          <span className="brand-mark">R</span>
          <div className="brand-title">
            <strong>AI 报告生成助手</strong>
            <span>{portalScreen ? "投行内部行业风险研究" : workspace.project.name}</span>
          </div>
        </div>

        {portalScreen ? (
          <nav className="mode-tabs" aria-label="门户导航">
            <button type="button" className={screen === "home" ? "active" : ""} onClick={() => setScreen("home")}>首页</button>
            <button type="button" className={screen === "create" ? "active" : ""} onClick={() => setScreen("create")}>创建报告</button>
            <button type="button" className={screen === "report-center" ? "active" : ""} onClick={() => setScreen("report-center")}>报告中心</button>
            <button type="button" className={screen === "management" ? "active" : ""} onClick={() => setScreen("management")}>管理中心</button>
          </nav>
        ) : (
          <nav className="mode-tabs" aria-label="工作台模式">
            <button type="button" className={workspaceMode === "editor" ? "active" : ""} onClick={() => setWorkspaceMode("editor")}>章节编辑</button>
            <button type="button" className={workspaceMode === "preview" ? "active" : ""} onClick={() => setWorkspaceMode("preview")}>整份预览</button>
            <button type="button" className={workspaceMode === "ai-map" ? "active" : ""} onClick={() => setWorkspaceMode("ai-map")}>模型调用日志</button>
          </nav>
        )}

        <div className="top-actions">
          {portalScreen ? (
            <span className={status === "后端未连接" ? "connection-state offline" : "connection-state"}>{busy || generating ? "处理中" : status}</span>
          ) : (
            <>
              <button type="button" className="ghost-btn" onClick={() => setScreen("home")}>首页</button>
              <button type="button" className="ghost-btn" onClick={() => { setWorkspaceMode("preview"); setStatus(previewIssues.length ? `发现 ${previewIssues.length} 项导出前问题` : "导出前检查通过"); }}>导出前检查</button>
              <button type="button" className="primary-btn" onClick={() => setWorkspaceMode("preview")}>预览并导出</button>
            </>
          )}
        </div>
      </header>
    );
  }

  function renderEntryCard(title: string, description: string, target: Screen, tone = "") {
    return (
      <button type="button" className={`entry-card ${tone}`} onClick={() => setScreen(target)}>
        <strong>{title}</strong>
        <span>{description}</span>
      </button>
    );
  }

  function renderHome() {
    return (
      <section className="portal-page">
        <header className="portal-head">
          <div>
            <p className="crumb">首页 / AI 报告生成助手</p>
            <h1>报告工作台</h1>
            <p>从创建报告进入章节编辑，资料、导出和日志都围绕当前单报告组织。</p>
          </div>
          <button type="button" className="primary-btn" onClick={() => { setScreen("workspace"); setWorkspaceMode("editor"); }}>继续编辑当前报告</button>
        </header>

        <div className="entry-grid">
          {renderEntryCard("创建报告", "填写基础信息，沿用当前章节结构，进入章节编辑流程。", "create", "entry-primary")}
          {renderEntryCard("报告中心", "查看当前单报告状态、完成度、导出和模型调用。", "report-center")}
          {renderEntryCard("管理中心", "查看章节结构、Prompt 和资料入库状态。", "management", "entry-muted")}
        </div>

        <section className="portal-section">
          <div className="portal-section-head">
            <div>
              <h2>当前报告</h2>
              <p>后端当前只维护一个报告工作区，这里不展示虚假的历史列表。</p>
            </div>
            <span className="small-note">{confirmedSections}/{workspace.sections.length} 已确认</span>
          </div>
          {renderCurrentReportTable(false)}
        </section>
      </section>
    );
  }

  function renderCurrentReportTable(showActions = true) {
    return (
      <div className="report-table-wrap">
        <table className="report-table">
          <thead>
            <tr>
              <th>报告名称</th>
              <th>行业</th>
              <th>年份</th>
              <th>语言</th>
              <th>章节</th>
              <th>资料</th>
              <th>最近导出</th>
              <th>最近调用</th>
              {showActions ? <th>操作</th> : null}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><button type="button" className="report-name-link" onClick={() => { setScreen("workspace"); setWorkspaceMode("editor"); }}>{workspace.project.name}</button></td>
              <td>{workspace.project.industry}</td>
              <td>{workspace.project.year}</td>
              <td>{workspace.project.language}</td>
              <td>{generatedSections}/{workspace.sections.length} 已生成，{confirmedSections} 已确认</td>
              <td>{indexedDocuments.length} 已入库，{selectedIndexedCount} 已勾选</td>
              <td>{latestExport ? `${latestExport.format} · ${latestExport.status}` : "暂无"}</td>
              <td>{latestLog ? `${latestLog.operation} · ${formatTime(latestLog.created_at)}` : "暂无"}</td>
              {showActions ? (
                <td>
                  <div className="table-actions">
                    <button type="button" onClick={() => { setScreen("workspace"); setWorkspaceMode("editor"); }}>编辑</button>
                    <button type="button" onClick={() => { setScreen("workspace"); setWorkspaceMode("preview"); }}>预览</button>
                    <button type="button" onClick={() => setScreen("create")}>新建</button>
                  </div>
                </td>
              ) : null}
            </tr>
          </tbody>
        </table>
      </div>
    );
  }

  function renderReportCenter() {
    return (
      <section className="portal-page report-center-page">
        <header className="portal-head">
          <div>
            <p className="crumb">报告中心 / 当前报告</p>
            <h1>当前单报告记录</h1>
            <p>当前版本没有多报告后端，这里只呈现真实工作区状态。</p>
          </div>
          <button type="button" className="primary-btn" onClick={() => setScreen("create")}>创建报告</button>
        </header>
        <section className="portal-section">
          <div className="portal-section-head">
            <div>
              <h2>报告状态</h2>
              <p>编辑、预览和新建都会继续使用现有后端接口。</p>
            </div>
            <span className="small-note">1 条记录</span>
          </div>
          {renderCurrentReportTable(true)}
        </section>
      </section>
    );
  }

  function renderManagement() {
    const promptReady = workspace.sections.filter((section) => section.prompt.trim()).length;
    return (
      <section className="portal-page management-page">
        <header className="portal-head">
          <div>
            <p className="crumb">管理中心 / 单报告配置</p>
            <h1>管理中心</h1>
            <p>模板、审批和权限只做当前报告状态壳，不新增后端数据结构。</p>
          </div>
          <button type="button" className="primary-btn" onClick={() => { setScreen("workspace"); setWorkspaceMode("editor"); }}>进入章节编辑</button>
        </header>

        <div className="management-grid">
          <section className="portal-section">
            <div className="portal-section-head"><h2>章节结构</h2><span className="small-note">{workspace.sections.length} 章</span></div>
            <div className="managed-list">
              {workspace.sections.map((section) => (
                <button key={section.id} type="button" className="managed-row" onClick={() => { setActiveSectionId(section.id); setScreen("workspace"); setWorkspaceMode("editor"); }}>
                  <span>{sectionLabel(section)}</span>
                  <em>{section.current_version ? (section.confirmed ? "已确认" : "待确认") : "待生成"}</em>
                </button>
              ))}
            </div>
          </section>

          <section className="portal-section">
            <div className="portal-section-head"><h2>Prompt 与资料</h2><span className="small-note">真实接口状态</span></div>
            <div className="summary-metrics">
              <div><strong>{promptReady}</strong><span>Prompt 已配置</span></div>
              <div><strong>{indexedDocuments.length}</strong><span>资料已入库</span></div>
              <div><strong>{workspace.ai_call_logs.length}</strong><span>模型调用</span></div>
              <div><strong>{previewIssues.length}</strong><span>导出问题</span></div>
            </div>
          </section>

          <section className="portal-section muted-section">
            <div className="portal-section-head"><h2>审批与权限</h2><span className="small-note">暂未接入</span></div>
            <div className="empty-note">当前后端没有审批流、用户权限和模板库接口。本页只保留信息架构位置，避免展示假数据。</div>
          </section>
        </div>
      </section>
    );
  }

  function renderCreateLeftPane() {
    return (
      <aside className="left-pane create-rail">
        <section className="rail-section">
          <div className="section-heading"><span>基础信息</span><small>创建前填写</small></div>
          <form className="project-grid" onSubmit={(event) => { event.preventDefault(); createNewReport(); }}>
            <label>报告名称<input value={projectDraft.name} onChange={(event) => setProjectDraft({ ...projectDraft, name: event.target.value })} /></label>
            <label>分析行业<input value={projectDraft.industry} onChange={(event) => setProjectDraft({ ...projectDraft, industry: event.target.value })} /></label>
            <label>年份<input value={projectDraft.year} onChange={(event) => setProjectDraft({ ...projectDraft, year: event.target.value })} /></label>
            <label>语言<input value={projectDraft.language} onChange={(event) => setProjectDraft({ ...projectDraft, language: event.target.value })} /></label>
          </form>
        </section>
        <section className="rail-section">
          <div className="section-heading"><span>参考文档</span><small>可选</small></div>
          <div className="doc-mini-list">
            {workspace.documents.slice(0, 6).map((document) => (
              <article key={document.id} className="doc-mini-card">
                <strong>{document.name}</strong>
                <span>{statusLabel(document)}</span>
              </article>
            ))}
            {!workspace.documents.length ? <p className="empty-note">未上传参考文档，仍可创建报告。</p> : null}
          </div>
          <form className="upload-card" onSubmit={uploadFile}>
            <label className="file-picker">
              <span>{fileDraft ? fileDraft.name : "选择本地文件"}</span>
              <small>.txt / .md / .pdf / .docx / .xlsx</small>
              <input key={fileInputKey} type="file" accept=".txt,.md,.pdf,.docx,.xlsx" onChange={(event) => setFileDraft(event.target.files?.[0] ?? null)} />
            </label>
            <div className="upload-controls">
              <select value={fileType} onChange={(event) => setFileType(event.target.value)}>
                <option>研报</option><option>政策</option><option>数据</option><option>年报</option><option>资料</option>
              </select>
              <button type="submit" disabled={busy || !fileDraft}>上传入库</button>
            </div>
          </form>
        </section>
      </aside>
    );
  }

  function renderCreateMain() {
    return (
      <article className="create-paper">
        <header className="create-head">
          <div>
            <p className="crumb">创建报告 / 当前章节结构</p>
            <h1>{projectDraft.industry || "当前行业"}报告模板</h1>
            <p>章节结构来自当前后端工作区，不引入新的模板数据库。</p>
          </div>
          <span className="status-pill">{workspace.sections.length} 个章节</span>
        </header>
        <div className="template-list">
          {workspace.sections.map((section) => (
            <button key={section.id} type="button" className="template-card" onClick={() => setActiveSectionId(section.id)}>
              <div className="template-card-head">
                <div>
                  <h2>{sectionLabel(section)}</h2>
                  <p>{section.prompt || "该章节暂无 Prompt。"}</p>
                </div>
                <span className={section.prompt ? "status-pill done" : "status-pill warn"}>{section.prompt ? "Prompt 已配" : "待配置"}</span>
              </div>
            </button>
          ))}
        </div>
      </article>
    );
  }

  function renderCreateRightPane() {
    const checks = [
      { ok: Boolean(projectDraft.name.trim()), message: projectDraft.name.trim() ? "报告名称已填写" : "请先填写报告名称" },
      { ok: Boolean(projectDraft.industry.trim()), message: projectDraft.industry.trim() ? "行业信息已填写" : "请填写分析行业" },
      { ok: workspace.sections.length > 0, message: workspace.sections.length > 0 ? "已有可用章节结构" : "章节结构尚未加载" },
    ];
    const canCreate = checks.every((check) => check.ok);
    return (
      <aside className="right-pane create-rail">
        <section className="panel-card">
          <header><h3>创建检查</h3><span className={canCreate ? "status-pill done" : "status-pill warn"}>{canCreate ? "可创建" : "待补充"}</span></header>
          <div className="panel-body check-stack">
            {checks.map((check) => <div key={check.message} className="check-line"><span className={check.ok ? "pulse" : "pulse running"}></span><span>{check.message}</span></div>)}
          </div>
        </section>
        <section className="panel-card">
          <header><h3>模板影响</h3><span className="small-note">进入编辑页后生效</span></header>
          <div className="panel-body summary-metrics compact">
            <div><strong>{workspace.sections.length}</strong><span>总章节</span></div>
            <div><strong>{workspace.sections.filter((section) => section.prompt.trim()).length}</strong><span>Prompt 已配</span></div>
            <div><strong>{workspace.documents.length}</strong><span>参考文档</span></div>
          </div>
        </section>
        <section className="panel-card">
          <header><h3>下一步</h3><span className="status-pill done">章节编辑</span></header>
          <div className="panel-body">
            <div className="next-step-list"><div>新建会清空章节版本和导出记录。</div><div>参考文档会保留，可继续勾选使用。</div><div>创建后直接进入当前报告工作台。</div></div>
            <button type="button" className="primary-action" disabled={busy || !canCreate} onClick={createNewReport}>确认创建报告</button>
          </div>
        </section>
      </aside>
    );
  }

  function renderAiMap() {
    return (
      <article className="document ai-map-paper">
        <div className="doc-header">
          <p className="doc-kicker">模型调用与检索证据</p>
          <h1>模型调用日志</h1>
          <span className="version-state">{workspace.ai_call_logs.length} 次调用</span>
        </div>
        <div className="ai-map-grid">
          {workspace.ai_call_logs.map((log) => (
            <section key={log.id} className="ai-map-card">
              <div>
                <strong>{log.operation}</strong>
                <small>{log.model} · {formatTime(log.created_at)}</small>
              </div>
              <dl>
                <div><dt>输入</dt><dd>{tokenText(log.prompt_tokens)}</dd></div>
                <div><dt>输出</dt><dd>{tokenText(log.completion_tokens)}</dd></div>
                <div><dt>合计</dt><dd>{tokenText(log.total_tokens)}</dd></div>
              </dl>
              <button type="button" onClick={() => setSelectedLog(log)}>查看完整日志</button>
            </section>
          ))}
          {!workspace.ai_call_logs.length ? <div className="empty-document"><h2>还没有模型调用记录</h2><p>生成章节后，这里会展示 Prompt、token 和检索引用。</p></div> : null}
        </div>
      </article>
    );
  }

  function renderPortalScreen() {
    if (screen === "report-center") return <section className="portal-pane">{renderReportCenter()}</section>;
    if (screen === "management") return <section className="portal-pane">{renderManagement()}</section>;
    return <section className="portal-pane">{renderHome()}</section>;
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
    <main className="app-root">
      {renderTopbar()}
      {screen === "workspace" ? (
        <section className="app-shell workspace-shell">
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
        <div className="workspace-toolbar">
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

        {workspaceMode === "editor" && activeSection ? (
          <article className="document">
            <div className="doc-header">
              <p className="doc-kicker">投行内部行业风险研究</p>
              <h1>{sectionLabel(activeSection)}</h1>
              <span className={activeSection.confirmed ? "version-state confirmed" : "version-state"}>
                {activeSection.confirmed ? "当前版本已确认" : "当前版本待确认"}
              </span>
            </div>

            {generating && generationSteps.length > 0 && (
              <div className="generation-trace" aria-live="polite">
                <p className="generation-trace-title">执行过程</p>
                <ol className="generation-trace-list">
                  {generationSteps.map((step, index) => {
                    const isLatest = index === generationSteps.length - 1;
                    return (
                      <li
                        key={`${step.phase}-${index}`}
                        className={
                          step.phase === "error"
                            ? "generation-trace-item error"
                            : isLatest
                              ? "generation-trace-item active"
                              : "generation-trace-item done"
                        }
                      >
                        <span className="generation-trace-mark" aria-hidden="true" />
                        <span>{step.message}</span>
                      </li>
                    );
                  })}
                </ol>
              </div>
            )}

            {generating || activeSection.current_version ? (
              <div className={generating ? "content-preview streaming" : "content-preview"}>
                {generating && !liveContent ? (
                  <p className="generation-waiting">正在准备正文…</p>
                ) : (
                  renderContentWithCitations(
                    generating ? liveContent : (activeSection.current_version?.content ?? ""),
                  )
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
        ) : workspaceMode === "preview" ? (
          <article className="document report-preview">
            <div className="doc-header">
              <p className="doc-kicker">完整交付预览</p>
              <h1>{workspace.project.name}</h1>
              <span className="version-state">{previewIssues.length ? "导出前需复核" : "可以导出"}</span>
            </div>

            <div className="check-strip">
              {previewIssues.length ? previewIssues.map((issue) => <span key={issue}>{issue}</span>) : <span>所有一级章节已生成并确认</span>}
            </div>

            {workspace.sections.map((section) => (
              <section key={section.id} className={section.level === 1 ? "preview-section major" : "preview-section"}>
                <h2>{sectionLabel(section)}</h2>
                {section.current_version ? (
                  <div className="preview-body">{renderContentWithCitations(section.current_version.content)}</div>
                ) : (
                  <p className="missing-copy">本章节尚未生成内容。</p>
                )}
              </section>
            ))}
          </article>
        ) : (
          renderAiMap()
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
                <button type="button" onClick={() => { setAssistContent(""); setAssistOpen(true); }}>AI 辅助生成</button>
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
              <details className="settings-panel table-config-panel">
                <summary>表格配置</summary>
                <label className="switch-row">
                  <input
                    type="checkbox"
                    checked={includeTable}
                    onChange={(event) => {
                      const checked = event.target.checked;
                      setIncludeTable(checked);
                      if (checked && !tableDrafts.length) {
                        setTableDrafts([createTableDraft()]);
                      }
                    }}
                  />
                  需要添加表格
                </label>
                {includeTable ? (
                  <div className="table-draft-list">
                    {tableDrafts.map((table, index) => (
                      <article key={table.id} className="table-draft-card">
                        <div className="table-draft-header">
                          <strong>表格 {index + 1}</strong>
                          <button
                            type="button"
                            disabled={tableDrafts.length <= 1}
                            onClick={() => removeTableDraft(table.id)}
                          >
                            删除
                          </button>
                        </div>
                        <div className="tab-switch" role="tablist" aria-label="表格来源">
                          <button
                            type="button"
                            role="tab"
                            aria-selected={table.mode === "synthesize"}
                            className={table.mode === "synthesize" ? "active" : ""}
                            onClick={() => updateTableDraft(table.id, { mode: "synthesize", confirmedChunkId: "", candidates: [] })}
                          >
                            AI 汇总
                          </button>
                          <button
                            type="button"
                            role="tab"
                            aria-selected={table.mode === "verbatim"}
                            className={table.mode === "verbatim" ? "active" : ""}
                            onClick={() => updateTableDraft(table.id, { mode: "verbatim" })}
                          >
                            引用原表
                          </button>
                        </div>
                        {table.mode === "synthesize" ? (
                          <>
                            <label>
                              表格标题
                              <input
                                type="text"
                                value={table.title}
                                placeholder="如：主要区域市场指标对比"
                                onChange={(event) => updateTableDraft(table.id, { title: event.target.value })}
                              />
                            </label>
                            <div className="table-column-list">
                              {table.columns.map((column, columnIndex) => (
                                <div key={`${table.id}-col-${columnIndex}`} className="table-column-row">
                                  <input
                                    type="text"
                                    value={column.name}
                                    placeholder="列名"
                                    onChange={(event) => {
                                      const columns = table.columns.map((item, idx) =>
                                        idx === columnIndex ? { ...item, name: event.target.value } : item,
                                      );
                                      updateTableDraft(table.id, { columns });
                                    }}
                                  />
                                  <input
                                    type="text"
                                    value={column.description}
                                    placeholder="列说明"
                                    onChange={(event) => {
                                      const columns = table.columns.map((item, idx) =>
                                        idx === columnIndex ? { ...item, description: event.target.value } : item,
                                      );
                                      updateTableDraft(table.id, { columns });
                                    }}
                                  />
                                  <button
                                    type="button"
                                    disabled={table.columns.length <= 1}
                                    onClick={() => {
                                      const columns = table.columns.filter((_, idx) => idx !== columnIndex);
                                      updateTableDraft(table.id, { columns });
                                    }}
                                  >
                                    删列
                                  </button>
                                </div>
                              ))}
                            </div>
                            <button
                              type="button"
                              onClick={() =>
                                updateTableDraft(table.id, {
                                  columns: [...table.columns, { name: "", description: "" }],
                                })
                              }
                            >
                              + 添加列
                            </button>
                            <label>
                              备注（可选）
                              <input
                                type="text"
                                value={table.notes}
                                onChange={(event) => updateTableDraft(table.id, { notes: event.target.value })}
                              />
                            </label>
                          </>
                        ) : (
                          <>
                            <label>
                              表格标题
                              <input
                                type="text"
                                value={table.title}
                                placeholder="如：主要区域市场指标对比"
                                onChange={(event) => updateTableDraft(table.id, { title: event.target.value })}
                              />
                            </label>
                            <label>
                              原表描述（检索用）
                              <input
                                type="text"
                                value={table.description}
                                placeholder="如：2025 收入分部数据表"
                                onChange={(event) =>
                                  updateTableDraft(table.id, {
                                    description: event.target.value,
                                    confirmedChunkId: "",
                                    candidates: [],
                                  })
                                }
                              />
                            </label>
                            <button
                              type="button"
                              disabled={busy || table.searching || !table.description.trim()}
                              onClick={() => void searchTableCandidates(table.id)}
                            >
                              {table.searching ? "检索中…" : "检索候选"}
                            </button>
                            {table.candidates.length ? (
                              <div className="table-candidate-list">
                                {table.candidates.map((candidate) => (
                                  <label key={candidate.chunk_id} className="table-candidate-item">
                                    <input
                                      type="radio"
                                      name={`candidate-${table.id}`}
                                      checked={table.confirmedChunkId === candidate.chunk_id}
                                      onChange={() => updateTableDraft(table.id, { confirmedChunkId: candidate.chunk_id })}
                                    />
                                    <span>
                                      <strong>{candidate.document_name}</strong>
                                      <small>
                                        {blockTypeLabel(candidate.block_type)} · {candidate.source_locator}
                                      </small>
                                      {renderTableCandidatePreview(candidate)}
                                    </span>
                                  </label>
                                ))}
                              </div>
                            ) : null}
                          </>
                        )}
                      </article>
                    ))}
                    <button
                      type="button"
                      disabled={tableDrafts.length >= 5}
                      onClick={() => setTableDrafts((current) => [...current, createTableDraft()])}
                    >
                      + 添加表格
                    </button>
                    {tableValidationError ? <p className="panel-copy table-validation">{tableValidationError}</p> : null}
                  </div>
                ) : null}
              </details>
              <button
                type="button"
                className="primary-action"
                disabled={busy || selectedIndexedCount === 0 || Boolean(tableValidationError)}
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
                  "已记录 Word 导出",
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
                  "已记录 PDF 导出",
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
      </aside>
        </section>
      ) : screen === "create" ? (
        <section className="app-shell create-shell">
          {renderCreateLeftPane()}
          <section className="center-pane create-center">{renderCreateMain()}</section>
          {renderCreateRightPane()}
        </section>
      ) : (
        renderPortalScreen()
      )}

      {assistOpen && activeSection && (
        <div className="modal-backdrop" role="presentation">
          <form className="modal" onSubmit={(event) => void submitPromptAssist(event)}>
            <div className="modal-header">
              <div>
                <p className="eyebrow">AI 辅助生成 Prompt</p>
                <h2>{activeSection.title}</h2>
              </div>
              <button type="button" disabled={assistGenerating} onClick={() => setAssistOpen(false)}>关闭</button>
            </div>
            <label>
              内容要求
              <textarea
                className="prompt-editor"
                value={assistContent}
                onChange={(event) => setAssistContent(event.target.value)}
                placeholder="描述本章节希望覆盖的重点、分析角度或写作要求（必填）。"
              />
            </label>
            <button
              type="submit"
              className="primary-action"
              disabled={busy || assistGenerating || !assistContent.trim()}
            >
              {assistGenerating ? "生成中…" : "生成 Prompt"}
            </button>
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
                      <small>{blockTypeLabel(chunk.metadata.block_type)} · {chunk.source_locator}</small>
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
