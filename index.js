// Qdrant Memory Extension for SillyTavern
// 從 Qdrant 取回相關記憶並注入對話情境
// Version 3.3.0 - 真 BM25 hybrid 搜尋（client 端 tokenize + Qdrant IDF modifier + RRF fusion）

const extensionName = "qdrant-memory"

// 預設設定
const defaultSettings = {
  enabled: true,
  qdrantUrl: "http://localhost:6333",
  qdrantApiKey: "",
  collectionName: "mem",
  embeddingProvider: "openai",
  openaiApiKey: "",
  openRouterApiKey: "",
  googleApiKey: "", // v3.2.0 新增
  localEmbeddingUrl: "",
  localEmbeddingApiKey: "",
  embeddingModel: "text-embedding-3-large",
  customEmbeddingDimensions: null,
  memoryLimit: 5,
  scoreThreshold: 0.3,
  memoryPosition: 2,
  debugMode: false,
  // v3.0 新增
  usePerCharacterCollections: true,
  autoSaveMemories: true,
  saveUserMessages: true,
  saveCharacterMessages: true,
  minMessageLength: 5,
  showMemoryNotifications: true,
  retainRecentMessages: 5,
  chunkMinSize: 1200,
  chunkMaxSize: 1500,
  chunkTimeout: 30000,
  // v3.1.2 新增
  dedupeThreshold: 0.92,
  preventDuplicateInjection: true,
  streamFinalizePollMs: 250,
  streamFinalizeStableMs: 1200,
  streamFinalizeMaxWaitMs: 300000,
  flushAfterAssistant: true,
  // v3.3.0 真 BM25 hybrid 搜尋
  useBM25Hybrid: false,        // 啟用後：新建集合用 hybrid schema，搜尋走 query+RRF
  bm25K1: 1.2,                 // BM25 TF saturation 參數
  bm25DenseTopK: 20,           // hybrid 搜尋時 dense prefetch 的 limit
  bm25SparseTopK: 20,          // hybrid 搜尋時 sparse prefetch 的 limit
  bm25CjkBigram: true,         // 中文除了逐字外是否加入相鄰二字 bigram（提升精準度但 sparse 變大）
}

let settings = { ...defaultSettings }
const saveQueue = []
let processingSaveQueue = false

let messageBuffer = []
let lastMessageTime = 0
let chunkTimer = null
let pendingAssistantFinalize = null

const memoryInjectionTracker = new Set()

function getChatHash(chat) {
  const lastMessages = chat.slice(-5).map(msg => {
    return `${msg.is_user ? 'U' : 'A'}_${msg.mes?.substring(0, 50) || ''}_${msg.send_date || ''}`
  }).join('|')

  return lastMessages
}

// v3.2.0：{{qdrant}} 巨集常數
const QDRANT_MACRO = "{{qdrant}}"

const EMBEDDING_MODEL_OPTIONS = {
  openai: [
    { value: "text-embedding-3-large", label: "text-embedding-3-large（最高品質）" },
    { value: "text-embedding-3-small", label: "text-embedding-3-small（速度較快）" },
    { value: "text-embedding-ada-002", label: "text-embedding-ada-002（舊版）" },
  ],
  openrouter: [
    { value: "openai/text-embedding-3-large", label: "OpenAI: Text Embedding 3 Large" },
    { value: "openai/text-embedding-3-small", label: "OpenAI: Text Embedding 3 Small" },
    { value: "openai/text-embedding-ada-002", label: "OpenAI: Text Embedding Ada 002" },
    { value: "qwen/qwen3-embedding-8b", label: "Qwen: Qwen3 Embedding 8B" },
    { value: "mistralai/mistral-embed-2312", label: "Mistral: Mistral Embed 2312" },
    { value: "google/gemini-embedding-001", label: "Google: Gemini Embedding 001" },
  ],
  // v3.2.0：Google AI Studio 模型清單動態載入
  google: [],
}

const DEFAULT_MODEL_BY_PROVIDER = {
  openai: "text-embedding-3-large",
  openrouter: EMBEDDING_MODEL_OPTIONS.openrouter[0].value,
  google: "models/text-embedding-004", // v3.2.0
}

const OPENROUTER_MODEL_ALIASES = {
  "text-embedding-3-large": "openai/text-embedding-3-large",
  "text-embedding-3-small": "openai/text-embedding-3-small",
  "text-embedding-ada-002": "openai/text-embedding-ada-002",
}

const OPENAI_MODEL_ALIASES = {
  "openai/text-embedding-3-large": "text-embedding-3-large",
  "openai/text-embedding-3-small": "text-embedding-3-small",
  "openai/text-embedding-ada-002": "text-embedding-ada-002",
}

// ============================================================================
// 日期／時間戳正規化
// ============================================================================

function normalizeTimestamp(date) {
  if (typeof date === 'number' && date > 1000000000000) {
    return date;
  }

  if (typeof date === 'number' && date > 1000000000 && date < 1000000000000) {
    return date * 1000;
  }

  if (date instanceof Date) {
    const timestamp = date.getTime();
    if (!isNaN(timestamp)) {
      return timestamp;
    }
  }

  if (typeof date === 'string' && date.trim()) {
    const parsed = new Date(date);
    const timestamp = parsed.getTime();
    if (!isNaN(timestamp)) {
      return timestamp;
    }
  }

  if (settings.debugMode) {
    console.warn('[Qdrant Memory] Could not normalize timestamp, using current time. Input:', date);
  }
  return Date.now();
}

function formatDateForChunk(timestamp) {
  try {
    const dateObj = new Date(timestamp);
    if (isNaN(dateObj.getTime())) {
      throw new Error('Invalid date');
    }
    return dateObj.toISOString().split('T')[0];
  } catch (e) {
    console.warn('[Qdrant Memory] Error formatting date:', e, 'timestamp:', timestamp);
    return new Date().toISOString().split('T')[0];
  }
}

// ============================================================================
// 設定管理（存 ST 伺服器端 extension_settings）
// ============================================================================

function getExtensionSettingsStore() {
  if (typeof window !== "undefined" && window.extension_settings) {
    return window.extension_settings
  }

  try {
    const ctx = window.SillyTavern?.getContext?.()
    if (ctx?.extensionSettings) return ctx.extensionSettings
    if (ctx?.extension_settings) return ctx.extension_settings
  } catch (e) {
    // 取不到就走 fallback
  }

  return null
}

function triggerSTSave() {
  if (typeof window.saveSettingsDebounced === "function") {
    try {
      window.saveSettingsDebounced()
      return true
    } catch (e) {
      console.warn("[Qdrant Memory] saveSettingsDebounced failed:", e)
    }
  }

  try {
    const ctx = window.SillyTavern?.getContext?.()
    if (typeof ctx?.saveSettingsDebounced === "function") {
      ctx.saveSettingsDebounced()
      return true
    }
    if (typeof ctx?.saveSettings === "function") {
      ctx.saveSettings()
      return true
    }
  } catch (e) {
    console.warn("[Qdrant Memory] context save failed:", e)
  }

  return false
}

function loadSettings() {
  const store = getExtensionSettingsStore()

  if (store) {
    if (!store[extensionName] || typeof store[extensionName] !== "object") {
      store[extensionName] = {}
    }

    const hasServerData = Object.keys(store[extensionName]).length > 0
    if (!hasServerData) {
      try {
        const legacy = localStorage.getItem(extensionName)
        if (legacy) {
          const parsed = JSON.parse(legacy)
          if (parsed && typeof parsed === "object") {
            Object.assign(store[extensionName], parsed)
            console.log("[Qdrant Memory] 已將舊有 localStorage 設定遷移到伺服器端")
            triggerSTSave()
          }
        }
      } catch (e) {
        console.warn("[Qdrant Memory] localStorage 遷移失敗：", e)
      }
    }

    settings = { ...defaultSettings, ...store[extensionName] }
    Object.assign(store[extensionName], settings)
  } else {
    console.warn("[Qdrant Memory] 找不到 extension_settings，退回 localStorage 模式")
    const saved = localStorage.getItem(extensionName)
    if (saved) {
      try {
        settings = { ...defaultSettings, ...JSON.parse(saved) }
      } catch (e) {
        console.error("[Qdrant Memory] 載入設定失敗：", e)
      }
    }
  }

  console.log("[Qdrant Memory] 設定已載入：", settings)
}

function saveSettings() {
  const store = getExtensionSettingsStore()

  if (store) {
    store[extensionName] = { ...settings }
    const ok = triggerSTSave()
    if (settings.debugMode) {
      console.log(`[Qdrant Memory] 設定已寫入 extension_settings（伺服器持久化：${ok ? "成功" : "未觸發"}）`)
    }
  } else {
    try {
      localStorage.setItem(extensionName, JSON.stringify(settings))
    } catch (e) {
      console.error("[Qdrant Memory] localStorage 寫入失敗：", e)
    }
    console.log("[Qdrant Memory] 設定已儲存（localStorage fallback）")
  }
}

function getCollectionName(characterName) {
  if (!settings.usePerCharacterCollections) {
    return settings.collectionName
  }

  const sanitized = characterName
    .toLowerCase()
    .replace(/[^a-z0-9_\-\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "")

  return `${settings.collectionName}_${sanitized}`
}

function encodeCollectionName(collectionName) {
  return encodeURIComponent(collectionName)
}

// 取得目前選用模型的嵌入維度
function getEmbeddingDimensions() {
  const dimensions = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
    "openai/text-embedding-3-large": 3072,
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-ada-002": 1536,
    "qwen/qwen3-embedding-8b": 4096,
    "mistralai/mistral-embed-2312": 1024,
    "google/gemini-embedding-001": 3072,
    // v3.2.0：Google AI Studio 原生模型
    "models/text-embedding-004": 768,
    "models/embedding-001": 768,
    "models/gemini-embedding-001": 3072,
    "models/gemini-embedding-exp-03-07": 3072,
    // 備援：未加 models/ 前綴的形式
    "text-embedding-004": 768,
    "embedding-001": 768,
    "gemini-embedding-001": 3072,
    "gemini-embedding-exp-03-07": 3072,
  }

  const customDimensions = Number.parseInt(settings.customEmbeddingDimensions, 10)
  const isCustomValid = Number.isFinite(customDimensions) && customDimensions > 0

  if (dimensions[settings.embeddingModel]) {
    return dimensions[settings.embeddingModel]
  }

  // local 與 google 允許自訂維度／自動偵測
  if (settings.embeddingProvider === "local" || settings.embeddingProvider === "google") {
    if (isCustomValid) {
      return customDimensions
    }
    return null
  }

  if (isCustomValid) {
    return customDimensions
  }

  return 1536
}

/**
 * v3.2.0：擴充為同時處理 local 與 google provider
 */
function updateLocalEmbeddingDimensions(vector) {
  if (settings.embeddingProvider !== "local" && settings.embeddingProvider !== "google") {
    return
  }

  if (!Array.isArray(vector)) {
    return
  }

  const vectorSize = vector.length
  if (!Number.isFinite(vectorSize) || vectorSize <= 0) {
    return
  }

  const currentDimensions = Number.parseInt(settings.customEmbeddingDimensions, 10)
  if (Number.isFinite(currentDimensions) && currentDimensions === vectorSize) {
    return
  }

  settings.customEmbeddingDimensions = vectorSize

  try {
    const $ = window.$
    if ($) {
      const $input = $("#qdrant_local_dimensions")
      if ($input && $input.length) {
        $input.val(vectorSize)
      }
    }
  } catch (error) {
    if (settings.debugMode) {
      console.warn("[Qdrant Memory] Unable to update local dimensions input:", error)
    }
  }

  saveSettings()

  if (settings.debugMode) {
    console.log(`[Qdrant Memory] 自動偵測到嵌入維度：${vectorSize}`)
  }
}

function getEmbeddingProviderError() {
  const provider = settings.embeddingProvider || "openai"

  const validProviders = ["openai", "openrouter", "google", "local"]
  if (!validProviders.includes(provider)) {
    return `不支援的嵌入服務提供者：${provider}`
  }

  if (provider === "openai") {
    if (!settings.openaiApiKey || !settings.openaiApiKey.trim()) {
      return "尚未設定 OpenAI API 金鑰"
    }
  }

  if (provider === "openrouter") {
    if (!settings.openRouterApiKey || !settings.openRouterApiKey.trim()) {
      return "尚未設定 OpenRouter API 金鑰"
    }
  }

  // v3.2.0：Google AI Studio
  if (provider === "google") {
    if (!settings.googleApiKey || !settings.googleApiKey.trim()) {
      return "尚未設定 Google API 金鑰"
    }
    if (!settings.embeddingModel || !settings.embeddingModel.trim()) {
      return "尚未選擇 Google 嵌入模型，請點選「抓取模型列表」"
    }
  }

  if (provider === "local") {
    if (!settings.localEmbeddingUrl || !settings.localEmbeddingUrl.trim()) {
      return "尚未設定本地嵌入端點網址"
    }

    if (settings.customEmbeddingDimensions != null && settings.customEmbeddingDimensions !== "") {
      const customDimensions = Number.parseInt(settings.customEmbeddingDimensions, 10)
      if (!Number.isFinite(customDimensions) || customDimensions <= 0) {
        return "嵌入維度必須是正整數"
      }
    }
  }

  if (!provider) {
    return "尚未設定嵌入服務提供者"
  }

  return null
}

// ============================================================================
// v3.2.0：Google AI Studio 模型清單抓取
// ============================================================================

/**
 * 從 Google AI Studio API 抓取可用嵌入模型清單
 * 篩選 supportedGenerationMethods 包含 "embedContent" 的模型
 */
async function fetchGoogleEmbeddingModels(apiKey) {
  if (!apiKey || !apiKey.trim()) {
    throw new Error("API 金鑰為空")
  }

  const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${encodeURIComponent(apiKey.trim())}`

  const response = await fetch(url, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  })

  if (!response.ok) {
    let errMsg = `HTTP ${response.status} ${response.statusText}`
    try {
      const errData = await response.json()
      if (errData?.error?.message) {
        errMsg += ` - ${errData.error.message}`
      }
    } catch (e) {
      // ignore
    }
    throw new Error(errMsg)
  }

  const data = await response.json()
  const models = Array.isArray(data?.models) ? data.models : []

  const embeddingModels = models.filter((m) => {
    const methods = m?.supportedGenerationMethods
    return Array.isArray(methods) && methods.includes("embedContent")
  })

  return embeddingModels.map((m) => {
    const name = m.name || ""
    const displayName = m.displayName || name
    return {
      value: name,
      label: displayName !== name ? `${displayName}（${name}）` : name,
    }
  })
}

// ============================================================================
// HTTP HEADERS 與 CSRF Token 處理
// ============================================================================

function tryGetCSRFTokenFromHelpers() {
  const helperCandidates = [
    () => (typeof window.getCSRFToken === "function" ? window.getCSRFToken() : null),
    () => (typeof window.getCsrfToken === "function" ? window.getCsrfToken() : null),
    () =>
      typeof window.SillyTavern?.getCSRFToken === "function"
        ? window.SillyTavern.getCSRFToken()
        : null,
    () =>
      typeof window.SillyTavern?.getCsrfToken === "function"
        ? window.SillyTavern.getCsrfToken()
        : null,
    () =>
      typeof window.SillyTavern?.extensions?.webui?.getCSRFToken === "function"
        ? window.SillyTavern.extensions.webui.getCSRFToken()
        : null,
    () =>
      typeof window.SillyTavern?.extensions?.webui?.getCsrfToken === "function"
        ? window.SillyTavern.extensions.webui.getCsrfToken()
        : null,
  ]

  for (const helper of helperCandidates) {
    try {
      const token = helper()
      if (typeof token === "string" && token.trim().length > 0) {
        return token.trim()
      }
    } catch (error) {
      console.warn("[Qdrant Memory] Failed to read CSRF token from helper:", error)
    }
  }

  return null
}

function getCookie(name) {
  const cookies = document.cookie ? document.cookie.split(";") : []

  for (const cookie of cookies) {
    const [cookieName, ...rest] = cookie.trim().split("=")
    if (cookieName === name) {
      return decodeURIComponent(rest.join("="))
    }
  }

  return null
}

function pickFirstCSRFToken() {
  const tokenCandidates = [
    document.querySelector('meta[name="csrf-token"]')?.content,
    document.querySelector('meta[name="csrfToken"]')?.content,
    window.CSRF_TOKEN,
    window.CSRFToken,
    window.csrfToken,
    window.csrf_token,
    tryGetCSRFTokenFromHelpers(),
    getCookie("csrftoken"),
    getCookie("csrf_token"),
    getCookie("XSRF-TOKEN"),
    getCookie("XSRF_TOKEN"),
  ]

  for (const token of tokenCandidates) {
    if (typeof token === "string" && token.trim().length > 0) {
      return token.trim()
    }
  }

  return null
}

function getHeadersFromSillyTavernContext() {
  try {
    const context = window.SillyTavern?.getContext?.()
    const getRequestHeaders = context?.getRequestHeaders

    if (typeof getRequestHeaders === "function") {
      const headers = getRequestHeaders.call(context)

      if (headers && typeof headers === "object") {
        return headers
      }
    }
  } catch (error) {
    console.warn("[Qdrant Memory] Failed to read headers from SillyTavern context:", error)
  }

  return null
}

function getSillyTavernHeaders() {
  if (settings.debugMode) {
    console.log("[Qdrant Memory] === Checking available ST methods ===")
    console.log("[Qdrant Memory] window.SillyTavern exists?", typeof SillyTavern !== "undefined")
    if (typeof SillyTavern !== "undefined") {
      console.log("[Qdrant Memory] SillyTavern keys:", Object.keys(SillyTavern))
      console.log("[Qdrant Memory] SillyTavern.getContext exists?", typeof SillyTavern.getContext === "function")
      if (typeof SillyTavern.getContext === "function") {
        const ctx = SillyTavern.getContext()
        console.log("[Qdrant Memory] Context keys:", Object.keys(ctx))
        console.log("[Qdrant Memory] getRequestHeaders exists?", typeof ctx.getRequestHeaders === "function")
      }
    }
  }

  const headerBuilders = [
    () => SillyTavern?.getContext?.()?.getRequestHeaders?.(),
    () => SillyTavern?.getRequestHeaders?.(),
    () => window.getRequestHeaders?.(),
    () => getContext()?.getRequestHeaders?.(),
  ]

  for (const builder of headerBuilders) {
    try {
      const headers = builder()
      if (headers && typeof headers === "object") {
        if (settings.debugMode) {
          console.log("[Qdrant Memory] ✓ Found working header builder!")
          console.log("[Qdrant Memory] Headers:", headers)
        }
        return headers
      }
    } catch (error) {
      // 換下一個
    }
  }

  if (settings.debugMode) {
    console.warn("[Qdrant Memory] No built-in header builder found, using manual fallback")
  }

  const headers = {
    "Content-Type": "application/json",
    Accept: "application/json",
    Origin: window.location.origin,
    "X-Requested-With": "XMLHttpRequest",
  }

  const csrfToken = pickFirstCSRFToken()

  if (csrfToken) {
    if (settings.debugMode) {
      console.log("[Qdrant Memory] Using manual CSRF token:", csrfToken.substring(0, 10) + "...")
    }
    headers["X-CSRF-Token"] = csrfToken
    headers["X-CSRFToken"] = csrfToken
    headers["csrf-token"] = csrfToken
  } else {
    console.warn("[Qdrant Memory] No CSRF token found - requests may fail")
  }

  return headers
}

function getQdrantHeaders() {
  const headers = {
    "Content-Type": "application/json",
  }

  if (settings.qdrantApiKey) {
    headers["api-key"] = settings.qdrantApiKey
  }

  return headers
}

// ============================================================================
// v3.3.0：BM25 sparse vector 工具
// ============================================================================
// 設計說明：
// - Qdrant 的 BM25 hybrid 模式是 sparse vector + 集合端 modifier:"idf"。
//   IDF 由 Qdrant 依集合統計自動計算；client 只負責產生 token frequency。
// - tokenize：CJK 逐字（必要）+ 相鄰 2-gram（可關）+ ASCII 詞（lowercase 切詞）。
// - token id：FNV-1a 32-bit hash，避免維護全域 vocabulary（手機環境友善）。
// - sparse value：BM25 TF saturation，doc 端 = tf*(k1+1)/(tf+k1)（b=0 簡化版，
//   不需 avgdl）；query 端為了把 query 重複詞的影響也飽和化，採同樣公式。

const BM25_DENSE_VECTOR_NAME = "dense"   // hybrid 集合的 dense vector 名
const BM25_SPARSE_VECTOR_NAME = "bm25"   // hybrid 集合的 sparse vector 名

function _isCjkChar(ch) {
  const code = ch.charCodeAt(0)
  // CJK Unified Ideographs + Extension A + 常用標點外的中日韓區塊
  return (
    (code >= 0x4e00 && code <= 0x9fff) ||
    (code >= 0x3400 && code <= 0x4dbf) ||
    (code >= 0xf900 && code <= 0xfaff) ||  // CJK Compatibility Ideographs
    (code >= 0x3040 && code <= 0x309f) ||  // Hiragana
    (code >= 0x30a0 && code <= 0x30ff) ||  // Katakana
    (code >= 0xac00 && code <= 0xd7af)     // Hangul Syllables
  )
}

function tokenizeForBM25(text, opts = {}) {
  if (!text || typeof text !== "string") return []
  const useBigram = opts.cjkBigram !== false
  const lower = text.toLowerCase()
  const tokens = []

  // ASCII 詞（含數字）：直接切
  const asciiMatches = lower.match(/[a-z0-9_]{2,}/g)
  if (asciiMatches) {
    for (const w of asciiMatches) tokens.push(w)
  }

  // CJK：scan 連續 CJK 段，逐字 + 相鄰 bigram
  let run = ""
  for (let i = 0; i < lower.length; i++) {
    const ch = lower[i]
    if (_isCjkChar(ch)) {
      run += ch
    } else {
      if (run) {
        for (let j = 0; j < run.length; j++) {
          tokens.push(run[j])
          if (useBigram && j + 1 < run.length) tokens.push(run.slice(j, j + 2))
        }
        run = ""
      }
    }
  }
  if (run) {
    for (let j = 0; j < run.length; j++) {
      tokens.push(run[j])
      if (useBigram && j + 1 < run.length) tokens.push(run.slice(j, j + 2))
    }
  }

  return tokens
}

// FNV-1a 32-bit hash（uint32）
function fnv1a32(str) {
  let h = 0x811c9dc5
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

// 產生 BM25 sparse vector：{ indices: uint32[], values: float[] }
// k1：TF saturation 參數（預設 1.2）
function makeBM25SparseVector(text, opts = {}) {
  const k1 = Number.isFinite(opts.k1) ? opts.k1 : (settings.bm25K1 || 1.2)
  const cjkBigram = opts.cjkBigram !== undefined ? opts.cjkBigram : (settings.bm25CjkBigram !== false)

  const tokens = tokenizeForBM25(text, { cjkBigram })
  if (tokens.length === 0) return { indices: [], values: [] }

  const tfMap = new Map()
  for (const t of tokens) {
    const id = fnv1a32(t)
    tfMap.set(id, (tfMap.get(id) || 0) + 1)
  }

  const indices = []
  const values = []
  for (const [id, tf] of tfMap) {
    indices.push(id)
    // BM25 TF saturation（b=0 簡化版，不需 avgdl）
    const v = (tf * (k1 + 1)) / (tf + k1)
    values.push(v)
  }
  return { indices, values }
}

// ============================================================================
// QDRANT 集合管理
// ============================================================================

// v3.3.0：集合格式 cache，避免每次 search/upsert 都重抓 collection info
// 結構：Map<collectionName, { exists, hasBM25, vectorSize, denseVectorName }>
// denseVectorName: hybrid 是 "dense"，legacy 是 "" 表示 unnamed
const collectionInfoCache = new Map()

function invalidateCollectionInfo(collectionName) {
  collectionInfoCache.delete(collectionName)
}

async function collectionExists(collectionName) {
  // 先看 cache
  if (collectionInfoCache.has(collectionName)) {
    const cached = collectionInfoCache.get(collectionName)
    return { ...cached }
  }

  try {
    const response = await fetch(`${settings.qdrantUrl}/collections/${encodeCollectionName(collectionName)}`, {
      headers: getQdrantHeaders(),
    })

    if (response.status === 404) {
      const info = { exists: false, vectorSize: null, hasBM25: false, denseVectorName: "" }
      collectionInfoCache.set(collectionName, info)
      return { ...info }
    }

    if (!response.ok) {
      console.error(
        `[Qdrant Memory] Failed to fetch collection info: ${collectionName} (${response.status} ${response.statusText})`,
      )
      return { exists: false, vectorSize: null, hasBM25: false, denseVectorName: "" }
    }

    const data = await response.json().catch(() => null)
    const params = data?.result?.config?.params || {}
    const vectorsConfig = params.vectors || data?.result?.vectors || null
    const sparseConfig = params.sparse_vectors || null

    // 判斷 dense vector 是 unnamed 還是 named
    let vectorSize = null
    let denseVectorName = ""  // 空字串代表 unnamed
    if (vectorsConfig && typeof vectorsConfig === "object") {
      if (Number.isFinite(vectorsConfig.size)) {
        // unnamed：{ size, distance }
        vectorSize = vectorsConfig.size
        denseVectorName = ""
      } else {
        // named：{ <name>: { size, distance }, ... }
        // 偏好 BM25_DENSE_VECTOR_NAME，否則取第一個
        const names = Object.keys(vectorsConfig)
        if (names.includes(BM25_DENSE_VECTOR_NAME)) {
          denseVectorName = BM25_DENSE_VECTOR_NAME
          vectorSize = vectorsConfig[BM25_DENSE_VECTOR_NAME]?.size ?? null
        } else if (names.length > 0) {
          denseVectorName = names[0]
          vectorSize = vectorsConfig[names[0]]?.size ?? null
        }
      }
    }

    const hasBM25 = !!(sparseConfig && sparseConfig[BM25_SPARSE_VECTOR_NAME])

    const info = { exists: true, vectorSize, hasBM25, denseVectorName }
    collectionInfoCache.set(collectionName, info)
    return { ...info }
  } catch (error) {
    console.error("[Qdrant Memory] Error checking collection:", error)
    return { exists: false, vectorSize: null, hasBM25: false, denseVectorName: "" }
  }
}

async function createCollection(collectionName, vectorSize, useHybrid) {
  try {
    const dimensions = Number.isFinite(vectorSize) && vectorSize > 0 ? vectorSize : getEmbeddingDimensions()

    if (!Number.isFinite(dimensions) || dimensions <= 0) {
      console.error(`[Qdrant Memory] Cannot create collection ${collectionName} - invalid embedding dimensions`)
      return false
    }

    // useHybrid 沒指定時，看 settings
    const hybrid = useHybrid !== undefined ? !!useHybrid : !!settings.useBM25Hybrid

    const body = hybrid
      ? {
          // hybrid schema：named dense + sparse bm25 with IDF modifier
          vectors: {
            [BM25_DENSE_VECTOR_NAME]: { size: dimensions, distance: "Cosine" },
          },
          sparse_vectors: {
            [BM25_SPARSE_VECTOR_NAME]: { modifier: "idf" },
          },
        }
      : {
          // legacy schema：unnamed dense
          vectors: { size: dimensions, distance: "Cosine" },
        }

    const response = await fetch(`${settings.qdrantUrl}/collections/${encodeCollectionName(collectionName)}`, {
      method: "PUT",
      headers: getQdrantHeaders(),
      body: JSON.stringify(body),
    })

    if (response.ok) {
      invalidateCollectionInfo(collectionName)
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Created collection: ${collectionName} (${hybrid ? "BM25 hybrid" : "legacy dense"})`)
      }
      return true
    } else {
      const errText = await response.text().catch(() => "")
      console.error(`[Qdrant Memory] Failed to create collection: ${collectionName} - ${response.status} ${errText}`)
      return false
    }
  } catch (error) {
    console.error("[Qdrant Memory] Error creating collection:", error)
    return false
  }
}

async function ensureCollection(characterName, vectorSize) {
  const collectionName = getCollectionName(characterName)
  const info = await collectionExists(collectionName)
  const { exists, vectorSize: existingSize } = info

  if (exists) {
    if (
      Number.isFinite(existingSize) &&
      Number.isFinite(vectorSize) &&
      existingSize > 0 &&
      vectorSize > 0 &&
      existingSize !== vectorSize
    ) {
      console.error(
        `[Qdrant Memory] Collection ${collectionName} has dimension ${existingSize}, but embedding returned ${vectorSize}. Please recreate the collection to match the model.`,
      )
      return false
    }

    return true
  }

  if (settings.debugMode) {
    console.log(`[Qdrant Memory] Collection doesn't exist, creating: ${collectionName} (hybrid=${!!settings.useBM25Hybrid})`)
  }

  // 新建集合：依 setting 決定 hybrid
  return await createCollection(collectionName, vectorSize, !!settings.useBM25Hybrid)
}

// ============================================================================
// 嵌入向量生成
// ============================================================================

async function generateEmbedding(text) {
  const providerError = getEmbeddingProviderError()
  if (providerError) {
    console.error(`[Qdrant Memory] ${providerError}`)
    return null
  }

  try {
    const provider = settings.embeddingProvider || "openai"
    let url = "https://api.openai.com/v1/embeddings"
    const headers = {
      "Content-Type": "application/json",
    }
    let body = {
      model: settings.embeddingModel,
      input: text,
    }

    if (provider === "openai") {
      headers.Authorization = `Bearer ${settings.openaiApiKey}`
    } else if (provider === "openrouter") {
      url = "https://openrouter.ai/api/v1/embeddings"
      headers.Authorization = `Bearer ${settings.openRouterApiKey}`
      if (window?.location?.origin) {
        headers["HTTP-Referer"] = window.location.origin
      }
      if (document?.title) {
        headers["X-Title"] = document.title
      }
    } else if (provider === "google") {
      // v3.2.0：Google AI Studio embedContent API
      const apiKey = settings.googleApiKey.trim()
      // 模型路徑要含 "models/" 前綴
      const modelPath = settings.embeddingModel.startsWith("models/")
        ? settings.embeddingModel
        : `models/${settings.embeddingModel}`
      url = `https://generativelanguage.googleapis.com/v1beta/${modelPath}:embedContent?key=${encodeURIComponent(apiKey)}`
      // Google 用獨特的 body 格式
      body = {
        model: modelPath,
        content: {
          parts: [{ text }],
        },
      }
      // Google 不用 Authorization header（key 在 URL 中）
    } else if (provider === "local") {
      url = settings.localEmbeddingUrl.trim()
      if (settings.localEmbeddingApiKey && settings.localEmbeddingApiKey.trim()) {
        headers.Authorization = `Bearer ${settings.localEmbeddingApiKey.trim()}`
      }
    } else {
      console.error(`[Qdrant Memory] Unsupported embedding provider: ${provider}`)
      return null
    }

    const response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      console.error(
        `[Qdrant Memory] ${provider} embedding API error:`,
        response.statusText,
        errorData
      )
      return null
    }

    const data = await response.json()
    let embeddingVector = null

    if (Array.isArray(data?.data) && Array.isArray(data.data[0]?.embedding)) {
      embeddingVector = data.data[0].embedding
    } else if (Array.isArray(data?.data) && Array.isArray(data.data[0]?.vector)) {
      embeddingVector = data.data[0].vector
    } else if (Array.isArray(data?.embedding?.values)) {
      // v3.2.0：Google AI Studio 格式
      embeddingVector = data.embedding.values
    } else if (Array.isArray(data?.embedding)) {
      embeddingVector = data.embedding
    } else if (Array.isArray(data?.embeddings)) {
      embeddingVector = data.embeddings[0]
    }

    if (!Array.isArray(embeddingVector)) {
      console.error("[Qdrant Memory] Unable to parse embedding response", data)
      return null
    }

    updateLocalEmbeddingDimensions(embeddingVector)

    return embeddingVector
  } catch (error) {
    console.error("[Qdrant Memory] Error generating embedding:", error)
    return null
  }
}

// ============================================================================
// 記憶搜尋與檢索
// ============================================================================

async function chunkExistsInCollection(collectionName, embedding, text, dedupeThreshold) {
  try {
    // v3.3.0：根據集合格式決定 vector 寫法
    const info = await collectionExists(collectionName)
    const isHybrid = info.exists && info.hasBM25 && info.denseVectorName

    const searchPayload = {
      // 純向量去重檢查：hybrid 集合用 named vector，legacy 用 unnamed
      vector: isHybrid
        ? { name: info.denseVectorName, vector: embedding }
        : embedding,
      limit: 1,
      score_threshold: dedupeThreshold,
      with_payload: true,
    }

    const response = await fetch(`${settings.qdrantUrl}/collections/${encodeCollectionName(collectionName)}/points/search`, {
      method: "POST",
      headers: getQdrantHeaders(),
      body: JSON.stringify(searchPayload),
    })

    if (!response.ok) {
      return false
    }

    const data = await response.json()
    const results = data.result || []

    if (results.length > 0) {
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Found similar chunk with score: ${results[0].score.toFixed(4)}`)
        console.log(`[Qdrant Memory] Existing: "${results[0].payload?.text?.substring(0, 80)}..."`)
        console.log(`[Qdrant Memory] New: "${text.substring(0, 80)}..."`)
      }
      return true
    }

    return false
  } catch (error) {
    console.warn('[Qdrant Memory] Deduplication check failed:', error)
    return false
  }
}

async function searchMemories(query, characterName) {
  if (!settings.enabled) return []

  try {
    const collectionName = getCollectionName(characterName)

    const embedding = await generateEmbedding(query)
    if (!embedding) return []

    const collectionReady = await ensureCollection(characterName, embedding.length)
    if (!collectionReady) {
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Collection not ready: ${collectionName}`)
      }
      return []
    }

    // v3.3.0：偵測集合格式（從 cache）
    const info = await collectionExists(collectionName)
    const useHybridSearch = !!(settings.useBM25Hybrid && info.hasBM25 && info.denseVectorName)

    const context = getContext()
    const chat = context.chat || []
    const excludedMessageIds = new Set()

    if (settings.retainRecentMessages > 0 && chat.length > settings.retainRecentMessages) {
      const recentMessages = chat.slice(-settings.retainRecentMessages)

      recentMessages.forEach(msg => {
        const normalizedDate = normalizeTimestamp(msg.send_date || Date.now())
        const msgIndex = chat.indexOf(msg)

        excludedMessageIds.add(`${characterName}_${normalizedDate}_${msgIndex}`)
        excludedMessageIds.add(`${characterName}_${msg.send_date}_${msgIndex}`)

        if (settings.debugMode && excludedMessageIds.size <= 5) {
          console.log(`[Qdrant Memory] Excluding message ID: ${characterName}_${normalizedDate}_${msgIndex}`)
        }
      })

      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Excluding ${excludedMessageIds.size} recent message IDs from search`)
      }
    }

    const filterConditions = []

    if (!settings.usePerCharacterCollections) {
      filterConditions.push({
        key: "character",
        match: { value: characterName },
      })
    }

    let results = []

    if (useHybridSearch) {
      // ============ Hybrid 路徑：query API + RRF fusion ============
      const sparse = makeBM25SparseVector(query)

      // 若 query 無法產出任何 token（極少見），退回純 dense
      const hasSparse = sparse.indices.length > 0

      const denseLimit = Math.max(1, settings.bm25DenseTopK || 20)
      const sparseLimit = Math.max(1, settings.bm25SparseTopK || 20)

      const prefetch = [
        {
          query: embedding,
          using: info.denseVectorName,
          limit: denseLimit,
        },
      ]
      if (hasSparse) {
        prefetch.push({
          query: { indices: sparse.indices, values: sparse.values },
          using: BM25_SPARSE_VECTOR_NAME,
          limit: sparseLimit,
        })
      }

      const queryPayload = {
        prefetch,
        // 兩路都成立才做 RRF fusion；只剩一路（query 無 token）就直接拿 dense 結果
        query: hasSparse ? { fusion: "rrf" } : embedding,
        limit: settings.memoryLimit * 2,
        with_payload: true,
      }
      // hybrid 用 dense-only 退化路徑時，要指定 using
      if (!hasSparse) queryPayload.using = info.denseVectorName

      if (filterConditions.length > 0) {
        queryPayload.filter = { must: filterConditions }
      }

      const response = await fetch(`${settings.qdrantUrl}/collections/${encodeCollectionName(collectionName)}/points/query`, {
        method: "POST",
        headers: getQdrantHeaders(),
        body: JSON.stringify(queryPayload),
      })

      if (!response.ok) {
        const errText = await response.text().catch(() => "")
        console.error("[Qdrant Memory] Hybrid query failed:", response.status, errText)
        // 回退：嘗試純 dense 搜尋（named vector）
        const fallbackPayload = {
          vector: { name: info.denseVectorName, vector: embedding },
          limit: settings.memoryLimit * 2,
          score_threshold: settings.scoreThreshold,
          with_payload: true,
        }
        if (filterConditions.length > 0) fallbackPayload.filter = { must: filterConditions }

        const fbRes = await fetch(`${settings.qdrantUrl}/collections/${encodeCollectionName(collectionName)}/points/search`, {
          method: "POST",
          headers: getQdrantHeaders(),
          body: JSON.stringify(fallbackPayload),
        })
        if (!fbRes.ok) return []
        const fbData = await fbRes.json()
        results = fbData.result || fbData.points || []
      } else {
        const data = await response.json()
        // query API 的回傳結構：result.points 為主，部分版本也用 result 直接是陣列
        results = data.result?.points || data.result || []
        if (settings.debugMode) {
          console.log(`[Qdrant Memory] Hybrid search: dense+sparse RRF, raw=${results.length}`)
        }
      }
    } else {
      // ============ Legacy 路徑：純向量 search ============
      const searchPayload = {
        // 集合可能是 hybrid 但 settings 關掉，仍要走 named vector
        vector: info.exists && info.denseVectorName
          ? { name: info.denseVectorName, vector: embedding }
          : embedding,
        limit: settings.memoryLimit * 2,
        score_threshold: settings.scoreThreshold,
        with_payload: true,
      }

      if (filterConditions.length > 0) {
        searchPayload.filter = { must: filterConditions }
      }

      const response = await fetch(`${settings.qdrantUrl}/collections/${encodeCollectionName(collectionName)}/points/search`, {
        method: "POST",
        headers: getQdrantHeaders(),
        body: JSON.stringify(searchPayload),
      })

      if (!response.ok) {
        console.error("[Qdrant Memory] Search failed:", response.statusText)
        return []
      }

      const data = await response.json()
      results = data.result || []
    }

    if (excludedMessageIds.size > 0) {
      const beforeFilterCount = results.length

      results = results.filter(memory => {
        const messageIds = memory.payload?.messageIds || ""
        const chunkMessageIds = messageIds.split(",")

        const hasExcludedMessage = chunkMessageIds.some(id => excludedMessageIds.has(id.trim()))

        if (hasExcludedMessage && settings.debugMode) {
          console.log(`[Qdrant Memory] Filtered out chunk containing recent message: ${messageIds}`)
        }

        return !hasExcludedMessage
      })

      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Filtered ${beforeFilterCount - results.length} chunks with recent messages`)
      }
    }

    const uniqueResults = []
    const seenTexts = new Set()

    for (const result of results) {
      const text = result.payload?.text || ""

      const normalizedText = text
        .replace(/\[[\d-]+\]/g, '')
        .replace(/\s+/g, ' ')
        .trim()
        .substring(0, 200)

      if (!seenTexts.has(normalizedText)) {
        seenTexts.add(normalizedText)
        uniqueResults.push(result)
      } else if (settings.debugMode) {
        console.log(`[Qdrant Memory] Filtered duplicate search result: "${normalizedText.substring(0, 50)}..."`)
      }

      if (uniqueResults.length >= settings.memoryLimit) {
        break
      }
    }

    results = uniqueResults

    if (settings.debugMode) {
      console.log(`[Qdrant Memory] Found ${results.length} unique memories (after deduplication, hybrid=${useHybridSearch})`)
    }

    return results
  } catch (error) {
    console.error("[Qdrant Memory] Error searching memories:", error)
    return []
  }
}

function formatMemories(memories) {
  if (!memories || memories.length === 0) return ""

  let formatted = "\n[過往對話記憶]\n\n"

  const personaName = getPersonaName()

  memories.forEach((memory) => {
    const payload = memory.payload

    let speakerLabel
    if (payload.isChunk) {
      speakerLabel = `對話（${payload.speakers}）`
    } else {
      speakerLabel = payload.speaker === "user"
        ? `${personaName} 說`
        : "角色說"
    }

    let text = payload.text.replace(/\n/g, " ")

    const score = (memory.score * 100).toFixed(0)

    formatted += `• ${speakerLabel}：「${text}」（相關度：${score}%）\n\n`
  })

  return formatted
}

// ============================================================================
// v3.2.0：{{qdrant}} 巨集處理
// ============================================================================

/**
 * 將記憶內容套用到 chat。
 *
 * 流程：
 * 1. 掃描所有訊息，若含 {{qdrant}} 巨集就替換成記憶文字（即使記憶為空也要清掉巨集）。
 * 2. 沒有巨集 + 有記憶 → 退回 memoryPosition 設定，用 system 訊息插入。
 * 3. 沒有巨集 + 沒有記憶 → 不做任何事。
 *
 * @param {Array} chat - SillyTavern chat 陣列
 * @param {string} memoryText - formatMemories() 的輸出，可能為空字串
 * @returns {"macro" | "position" | "none"}
 */
function applyMemoryInjection(chat, memoryText) {
  // 巨集替換時去掉前後空白行，避免破壞訊息排版
  const inlineText = (memoryText || "").replace(/^\n+|\n+$/g, "")

  let macroFound = false

  for (let i = 0; i < chat.length; i++) {
    const msg = chat[i]
    if (msg && typeof msg.mes === "string" && msg.mes.includes(QDRANT_MACRO)) {
      // split + join 比 replaceAll 相容性高
      msg.mes = msg.mes.split(QDRANT_MACRO).join(inlineText)
      macroFound = true

      if (settings.debugMode) {
        console.log(`[Qdrant Memory] {{qdrant}} 巨集已替換於訊息 #${i}`)
      }
    }
  }

  if (macroFound) {
    return "macro"
  }

  // 沒有巨集且沒有記憶 → 不動作
  if (!memoryText) {
    return "none"
  }

  // 退回 memoryPosition 注入
  const memoryEntry = {
    name: "System",
    is_user: false,
    is_system: true,
    mes: memoryText,
    send_date: Date.now(),
  }

  const insertIndex = Math.max(0, chat.length - settings.memoryPosition)
  chat.splice(insertIndex, 0, memoryEntry)

  if (settings.debugMode) {
    console.log(`[Qdrant Memory] 以 memoryPosition=${settings.memoryPosition} 注入記憶（位置 ${insertIndex}）`)
  }

  return "position"
}

// ============================================================================
// 訊息切塊與緩衝
// ============================================================================

function getChatParticipants() {
  const context = getContext()
  const characterName = context.name2

  const characters = context.characters || []
  const chat = context.chat || []

  if (characters.length > 1) {
    const participants = new Set()

    if (characterName) {
      participants.add(characterName)
    }

    chat.slice(-50).forEach((msg) => {
      if (!msg.is_user && msg.name && msg.name !== "System") {
        participants.add(msg.name)
      }
    })

    return Array.from(participants)
  }

  return characterName ? [characterName] : []
}

function createChunkFromBuffer() {
  if (messageBuffer.length === 0) return null

  let chunkText = ""
  const speakers = new Set()
  const messageIds = []
  let totalLength = 0
  const currentTimestamp = Date.now()

  const personaName = getPersonaName()

  messageBuffer.forEach((msg) => {
    const speaker = msg.isUser ? personaName : msg.characterName
    speakers.add(speaker)
    messageIds.push(msg.messageId)

    const line = `${speaker}: ${msg.text}\n`
    chunkText += line
    totalLength += line.length
  })

  let finalText = chunkText.trim()
  const dateStr = formatDateForChunk(currentTimestamp)
  finalText = `[${dateStr}]\n${finalText}`

  return {
    text: finalText,
    speakers: Array.from(speakers),
    messageIds: messageIds,
    messageCount: messageBuffer.length,
    timestamp: currentTimestamp,
  }
}

async function saveChunkToQdrant(chunk, participants) {
  if (!settings.enabled) return false
  if (!chunk || !participants || participants.length === 0) return false

  try {
    const embedding = await generateEmbedding(chunk.text)
    if (!embedding) {
      console.error("[Qdrant Memory] Cannot save chunk - embedding generation failed")
      return false
    }

    let alreadyExists = false

    for (const characterName of participants) {
      const collectionName = getCollectionName(characterName)
      const collectionReady = await ensureCollection(characterName, embedding.length)

      if (!collectionReady) {
        console.error(`[Qdrant Memory] Cannot check duplicates - collection creation failed for ${characterName}`)
        continue
      }

      const exists = await chunkExistsInCollection(
        collectionName,
        embedding,
        chunk.text,
        settings.dedupeThreshold
      )

      if (exists) {
        alreadyExists = true
        if (settings.debugMode) {
          console.log(`[Qdrant Memory] Duplicate chunk detected in ${characterName}'s collection, skipping save`)
        }
        break
      }
    }

    if (alreadyExists) {
      if (settings.showMemoryNotifications) {
        const toastr = window.toastr
        toastr.info("已儲存過類似的對話", "Qdrant Memory", { timeOut: 1500 })
      }
      return false
    }

    const pointId = generateUUID()

    const payload = {
      text: chunk.text,
      speakers: chunk.speakers.join(", "),
      messageCount: chunk.messageCount,
      timestamp: chunk.timestamp,
      messageIds: chunk.messageIds.join(","),
      isChunk: true,
    }

    const savePromises = participants.map(async (characterName) => {
      const collectionName = getCollectionName(characterName)

      const collectionReady = await ensureCollection(characterName, embedding.length)
      if (!collectionReady) {
        console.error(`[Qdrant Memory] Cannot save chunk - collection creation failed for ${characterName}`)
        return false
      }

      const characterPayload = settings.usePerCharacterCollections
        ? payload
        : { ...payload, character: characterName }

      // v3.3.0：根據集合格式決定 vector 結構
      const colInfo = await collectionExists(collectionName)
      let vectorField
      if (colInfo.exists && colInfo.hasBM25 && colInfo.denseVectorName) {
        // hybrid 集合：dense + sparse 一起 upsert
        const sparse = makeBM25SparseVector(chunk.text)
        vectorField = {
          [colInfo.denseVectorName]: embedding,
          [BM25_SPARSE_VECTOR_NAME]: { indices: sparse.indices, values: sparse.values },
        }
      } else if (colInfo.exists && colInfo.denseVectorName) {
        // 罕見：named dense 但沒 sparse（例如手動建立的）
        vectorField = { [colInfo.denseVectorName]: embedding }
      } else {
        // legacy unnamed dense
        vectorField = embedding
      }

      const response = await fetch(`${settings.qdrantUrl}/collections/${encodeCollectionName(collectionName)}/points`, {
        method: "PUT",
        headers: getQdrantHeaders(),
        body: JSON.stringify({
          points: [
            {
              id: pointId,
              vector: vectorField,
              payload: characterPayload,
            },
          ],
        }),
      })

      if (!response.ok) {
        const errText = await response.text().catch(() => "")
        console.error(
          `[Qdrant Memory] Failed to save chunk to ${characterName}: ${response.status} ${errText}`,
        )
        return false
      }

      if (settings.debugMode) {
        console.log(
          `[Qdrant Memory] Saved chunk to ${characterName}'s collection (${chunk.messageCount} messages, ${chunk.text.length} chars, hybrid=${!!colInfo.hasBM25})`,
        )
      }

      return true
    })

    const results = await Promise.all(savePromises)
    const successCount = results.filter((r) => r).length

    if (settings.debugMode) {
      console.log(`[Qdrant Memory] Chunk saved to ${successCount}/${participants.length} collections`)
    }

    return successCount > 0
  } catch (err) {
    console.error("[Qdrant Memory] Error saving chunk:", err)
    return false
  }
}

async function processMessageBuffer() {
  if (!settings.enabled) return
  if (messageBuffer.length === 0) return

  const chunk = createChunkFromBuffer()
  if (!chunk) return

  const participants = getChatParticipants()

  if (participants.length === 0) {
    console.error("[Qdrant Memory] No participants found for chunk")
    messageBuffer = []
    return
  }

  await saveChunkToQdrant(chunk, participants)

  messageBuffer = []
}

function bufferMessage(text, characterName, isUser, messageId) {
  if (!settings.enabled) return
  if (!settings.autoSaveMemories) return
  if (getEmbeddingProviderError()) return
  if (text.length < settings.minMessageLength) return

  if (isUser && !settings.saveUserMessages) return
  if (!isUser && !settings.saveCharacterMessages) return

  messageBuffer.push({ text, characterName, isUser, messageId })
  lastMessageTime = Date.now()

  let bufferSize = 0
  messageBuffer.forEach((msg) => {
    bufferSize += msg.text.length + msg.characterName.length + 4
  })

  if (settings.debugMode) {
    console.log(`[Qdrant Memory] Buffer: ${messageBuffer.length} messages, ${bufferSize} chars`)
  }

  if (chunkTimer) {
    clearTimeout(chunkTimer)
  }

  if (bufferSize >= settings.chunkMaxSize) {
    if (settings.debugMode) {
      console.log(`[Qdrant Memory] Buffer reached max size (${bufferSize}), processing chunk`)
    }
    processMessageBuffer()
  }
  else if (bufferSize >= settings.chunkMinSize) {
    chunkTimer = setTimeout(() => {
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Buffer reached min size and timeout, processing chunk`)
      }
      processMessageBuffer()
    }, 5000)
  }
  else {
    chunkTimer = setTimeout(() => {
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Buffer timeout reached, processing chunk`)
      }
      processMessageBuffer()
    }, settings.chunkTimeout)
  }
}

// ============================================================================
// 聊天紀錄索引功能
// ============================================================================

async function getCharacterChats(characterName) {
  try {
    const context = getContext()

    if (settings.debugMode) {
      console.log("[Qdrant Memory] Getting chats for character:", characterName)
    }

    let avatar_url = `${characterName}.png`
    if (context.characters && Array.isArray(context.characters)) {
      const char = context.characters.find((c) => c.name === characterName)
      if (char && char.avatar) {
        avatar_url = char.avatar
      }
    }

    const response = await fetch("/api/characters/chats", {
      method: "POST",
      headers: getSillyTavernHeaders(),
      credentials: "include",
      body: JSON.stringify({
        avatar_url: avatar_url,
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error("[Qdrant Memory] Failed to get chat list:", response.status, response.statusText)
      console.error("[Qdrant Memory] Error response:", errorText)
      return []
    }

    const data = await response.json()

    let chatFiles = []

    if (Array.isArray(data)) {
      if (typeof data[0] === 'string') {
        chatFiles = data
      } else if (data[0] && data[0].file_name) {
        chatFiles = data.map(item => item.file_name)
      } else {
        chatFiles = data.map(item => {
          if (typeof item === 'string') return item
          if (item.file_name) return item.file_name
          if (item.filename) return item.filename
          return null
        }).filter(f => f !== null)
      }
    } else if (data && Array.isArray(data.files)) {
      chatFiles = data.files.map(item => {
        if (typeof item === 'string') return item
        if (item.file_name) return item.file_name
        if (item.filename) return item.filename
        return null
      }).filter(f => f !== null)
    } else if (data && Array.isArray(data.chats)) {
      chatFiles = data.chats.map(item => {
        if (typeof item === 'string') return item
        if (item.file_name) return item.file_name
        if (item.filename) return item.filename
        return null
      }).filter(f => f !== null)
    }

    if (settings.debugMode) {
      console.log("[Qdrant Memory] Extracted filenames:", chatFiles)
    }

    return chatFiles
  } catch (error) {
    console.error("[Qdrant Memory] Error getting character chats:", error)
    return []
  }
}

async function loadChatFile(characterName, chatFile) {
  try {
    if (settings.debugMode) {
      console.log("[Qdrant Memory] Loading chat file:", chatFile, "for character:", characterName)
    }

    if (typeof chatFile !== 'string') {
      console.error("[Qdrant Memory] chatFile is not a string:", chatFile)
      if (chatFile && chatFile.file_name) {
        chatFile = chatFile.file_name
      } else {
        return null
      }
    }

    const fileNameWithoutExt = chatFile.replace(/\.jsonl$/, '')

    const context = getContext()

    let avatar_url = `${characterName}.png`
    if (context.characters && Array.isArray(context.characters)) {
      const char = context.characters.find((c) => c.name === characterName)
      if (char && char.avatar) {
        avatar_url = char.avatar
      }
    }

    const response = await fetch("/api/chats/get", {
      method: "POST",
      headers: getSillyTavernHeaders(),
      credentials: "include",
      body: JSON.stringify({
        ch_name: characterName,
        file_name: fileNameWithoutExt,
        avatar_url: avatar_url,
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error("[Qdrant Memory] Failed to load chat file:", response.status, response.statusText)
      console.error("[Qdrant Memory] Error response:", errorText)
      return null
    }

    const chatData = await response.json()

    let messages = null
    if (Array.isArray(chatData)) {
      messages = chatData
    } else if (chatData && Array.isArray(chatData.chat)) {
      messages = chatData.chat
    } else if (chatData && Array.isArray(chatData.messages)) {
      messages = chatData.messages
    } else if (chatData && typeof chatData === 'object') {
      messages = [chatData]
    }

    if (settings.debugMode) {
      console.log("[Qdrant Memory] Loaded chat with", messages?.length || 0, "messages")
    }

    return messages
  } catch (error) {
    console.error("[Qdrant Memory] Error loading chat file:", error)
    return null
  }
}

async function chunkExists(collectionName, messageIds) {
  try {
    const response = await fetch(`${settings.qdrantUrl}/collections/${encodeCollectionName(collectionName)}/points/scroll`, {
      method: "POST",
      headers: getQdrantHeaders(),
      body: JSON.stringify({
        filter: {
          should: messageIds.map((id) => ({
            key: "messageIds",
            match: { text: id },
          })),
        },
        limit: 1,
        with_payload: false,
      }),
    })

    if (!response.ok) return false

    const data = await response.json()
    return data.result?.points?.length > 0
  } catch (error) {
    console.error("[Qdrant Memory] Error checking chunk existence:", error)
    return false
  }
}

function createChunksFromChat(messages, characterName) {
  const chunks = []
  let currentChunk = []
  let currentSize = 0

  for (const msg of messages) {
    if (msg.is_system) continue

    const text = msg.mes?.trim()
    if (!text || text.length < settings.minMessageLength) continue

    const isUser = msg.is_user || false
    if (isUser && !settings.saveUserMessages) continue
    if (!isUser && !settings.saveCharacterMessages) continue

    const normalizedDate = normalizeTimestamp(msg.send_date || Date.now())

    if (settings.debugMode) {
      console.log("[Qdrant Memory] Message date - raw:", msg.send_date, "normalized:", normalizedDate, "formatted:", formatDateForChunk(normalizedDate))
    }

    const messageObj = {
      text: text,
      characterName: characterName,
      isUser: isUser,
      messageId: `${characterName}_${normalizedDate}_${messages.indexOf(msg)}`,
      timestamp: normalizedDate,
    }

    const messageSize = text.length + characterName.length + 4

    if (currentSize + messageSize > settings.chunkMaxSize && currentChunk.length > 0) {
      chunks.push(createChunkFromMessages(currentChunk))
      currentChunk = []
      currentSize = 0
    }

    currentChunk.push(messageObj)
    currentSize += messageSize

    if (currentSize >= settings.chunkMinSize && currentChunk.length >= 3) {
      chunks.push(createChunkFromMessages(currentChunk))
      currentChunk = []
      currentSize = 0
    }
  }

  if (currentChunk.length > 0) {
    chunks.push(createChunkFromMessages(currentChunk))
  }

  return chunks
}

function createChunkFromMessages(messages) {
  let chunkText = ""
  const speakers = new Set()
  const messageIds = []
  let oldestTimestamp = Number.POSITIVE_INFINITY

  const personaName = getPersonaName()

  messages.forEach((msg) => {
    const speaker = msg.isUser ? personaName : msg.characterName
    speakers.add(speaker)
    messageIds.push(msg.messageId)

    const line = `${speaker}: ${msg.text}\n`
    chunkText += line

    if (msg.timestamp < oldestTimestamp) {
      oldestTimestamp = msg.timestamp
    }
  })

  let finalText = chunkText.trim()
  if (oldestTimestamp !== Number.POSITIVE_INFINITY) {
    const dateStr = formatDateForChunk(oldestTimestamp)
    finalText = `[${dateStr}]\n${finalText}`
  }

  return {
    text: finalText,
    speakers: Array.from(speakers),
    messageIds: messageIds,
    messageCount: messages.length,
    timestamp: oldestTimestamp !== Number.POSITIVE_INFINITY ? oldestTimestamp : Date.now(),
  }
}

async function indexCharacterChats() {
  const context = getContext()
  const characterName = context.name2
  const toastr = window.toastr
  const $ = window.$

  if (!characterName) {
    toastr.warning("尚未選擇角色", "Qdrant Memory")
    return
  }

  const providerError = getEmbeddingProviderError()
  if (providerError) {
    toastr.error(providerError, "Qdrant Memory")
    return
  }

  const modalHtml = `
    <div id="qdrant_index_modal" style="
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: white;
      padding: 30px;
      border-radius: 10px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
      z-index: 10000;
      max-width: 500px;
      width: 90%;
    ">
      <div style="color: #333;">
        <h3 style="margin-top: 0;">索引聊天紀錄 - ${characterName}</h3>
        <p id="qdrant_index_status">正在掃描聊天檔案...</p>
        <div style="background: #f0f0f0; border-radius: 5px; height: 20px; margin: 15px 0; overflow: hidden;">
          <div id="qdrant_index_progress" style="background: #4CAF50; height: 100%; width: 0%; transition: width 0.3s;"></div>
        </div>
        <p id="qdrant_index_details" style="font-size: 0.9em; color: #666;"></p>
        <button id="qdrant_index_cancel" class="menu_button" style="margin-top: 15px;">取消</button>
      </div>
    </div>
    <div id="qdrant_index_overlay" style="
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0,0,0,0.5);
      z-index: 9999;
    "></div>
  `

  $("body").append(modalHtml)

  const closeModal = () => {
    $("#qdrant_index_modal").remove()
    $("#qdrant_index_overlay").remove()
  }

  const setCancelButtonToClose = () => {
    $("#qdrant_index_cancel")
      .prop("disabled", false)
      .text("關閉")
      .off("click")
      .on("click", closeModal)
  }

  let cancelled = false
  $("#qdrant_index_cancel").on("click", () => {
    cancelled = true
    $("#qdrant_index_cancel").text("取消中...").prop("disabled", true)
  })

  try {
    const chatFiles = await getCharacterChats(characterName)

    if (chatFiles.length === 0) {
      $("#qdrant_index_status").text("找不到任何聊天檔案")
      setCancelButtonToClose()
      setTimeout(() => {
        closeModal()
      }, 2000)
      return
    }

    $("#qdrant_index_status").text(`共找到 ${chatFiles.length} 個聊天檔案`)

    const collectionName = getCollectionName(characterName)

    let totalChunks = 0
    let savedChunks = 0
    let skippedChunks = 0

    for (let i = 0; i < chatFiles.length; i++) {
      if (cancelled) break

      const chatFile = chatFiles[i]
      const progress = ((i / chatFiles.length) * 100).toFixed(0)

      $("#qdrant_index_progress").css("width", `${progress}%`)
      $("#qdrant_index_status").text(`處理中：第 ${i + 1}/${chatFiles.length} 個聊天`)
      $("#qdrant_index_details").text(`檔案：${chatFile}`)

      const chatData = await loadChatFile(characterName, chatFile)
      if (!chatData || !Array.isArray(chatData)) continue

      const chunks = createChunksFromChat(chatData, characterName)
      totalChunks += chunks.length

      for (const chunk of chunks) {
        if (cancelled) break

        const exists = await chunkExists(collectionName, chunk.messageIds)
        if (exists) {
          skippedChunks++
          continue
        }

        const participants = [characterName]

        const success = await saveChunkToQdrant(chunk, participants)
        if (success) {
          savedChunks++
        }

        $("#qdrant_index_details").text(`已儲存：${savedChunks}｜略過：${skippedChunks}｜總計：${totalChunks}`)
      }
    }

    $("#qdrant_index_progress").css("width", "100%")

    if (cancelled) {
      $("#qdrant_index_status").text("已取消索引")
      toastr.info(`取消前已索引 ${savedChunks} 個區塊`, "Qdrant Memory")
    } else {
      $("#qdrant_index_status").text("索引完成！")
      toastr.success(`新增索引 ${savedChunks} 個區塊，略過已存在的 ${skippedChunks} 個`, "Qdrant Memory")
    }

    setCancelButtonToClose()
  } catch (error) {
    console.error("[Qdrant Memory] Error indexing chats:", error)
    $("#qdrant_index_status").text("索引過程發生錯誤")
    $("#qdrant_index_details").text(error.message)
    toastr.error("聊天索引失敗", "Qdrant Memory")
    setCancelButtonToClose()
  }
}

// ============================================================================
// v3.3.0：BM25 集合 migration（從 legacy 升級到 hybrid schema）
// ============================================================================

// 把舊集合（unnamed dense）升級為 hybrid（named dense + sparse bm25）
// 流程：scroll 全部點位 → localStorage 備份 → 刪舊 → 建新 → 批次重 upsert
// 失敗時備份仍在 localStorage，可手動恢復
async function migrateCollectionToBM25(characterName, progressCb) {
  const collectionName = getCollectionName(characterName)
  const log = (msg) => {
    console.log(`[Qdrant Memory][migrate] ${msg}`)
    if (typeof progressCb === "function") progressCb(msg)
  }

  // 1) 確認集合存在 + 檢查格式
  invalidateCollectionInfo(collectionName)
  const info = await collectionExists(collectionName)
  if (!info.exists) {
    log(`集合不存在：${collectionName}`)
    return { ok: false, reason: "collection-missing" }
  }
  if (info.hasBM25) {
    log(`集合 ${collectionName} 已是 hybrid 格式，無需 migration`)
    return { ok: true, alreadyHybrid: true, collection: collectionName }
  }
  if (!Number.isFinite(info.vectorSize) || info.vectorSize <= 0) {
    log(`無法取得 ${collectionName} 的 vector 維度`)
    return { ok: false, reason: "no-vector-size" }
  }

  // 2) Scroll 全部點位（with vector + payload）
  log(`開始 scroll 點位...`)
  const allPoints = []
  let nextOffset = null
  const batchSize = 200
  let safety = 1000  // 最多 200 * 1000 = 200k 點位
  while (safety-- > 0) {
    const body = {
      limit: batchSize,
      with_payload: true,
      with_vector: true,
    }
    if (nextOffset !== null && nextOffset !== undefined) body.offset = nextOffset

    const res = await fetch(`${settings.qdrantUrl}/collections/${encodeCollectionName(collectionName)}/points/scroll`, {
      method: "POST",
      headers: getQdrantHeaders(),
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      const errText = await res.text().catch(() => "")
      log(`scroll 失敗：${res.status} ${errText}`)
      return { ok: false, reason: "scroll-failed", error: errText }
    }
    const data = await res.json()
    const points = data.result?.points || []
    for (const p of points) allPoints.push(p)
    nextOffset = data.result?.next_page_offset
    log(`已讀 ${allPoints.length} 點位${nextOffset ? "（繼續）" : ""}`)
    if (!nextOffset) break
  }

  if (allPoints.length === 0) {
    log(`集合無點位，直接重建為 hybrid 格式`)
    // 即使無點位也要重建集合 schema
  }

  // 3) localStorage 備份（限制 5MB；超過只保留 console）
  const backupKey = `qdrant_bm25_backup_${collectionName}_${Date.now()}`
  try {
    const backupPayload = {
      collection: collectionName,
      timestamp: Date.now(),
      vectorSize: info.vectorSize,
      pointCount: allPoints.length,
      points: allPoints,
    }
    const json = JSON.stringify(backupPayload)
    if (json.length < 5 * 1024 * 1024) {
      try {
        localStorage.setItem(backupKey, json)
        log(`備份寫入 localStorage：${backupKey}（${(json.length / 1024).toFixed(0)} KB）`)
      } catch (e) {
        log(`localStorage 寫入失敗（可能配額不足）：${e?.message || e}`)
        console.log("[Qdrant Memory][migrate] 備份完整資料（請手動保存）：", backupPayload)
      }
    } else {
      log(`備份太大（${(json.length / 1024 / 1024).toFixed(1)} MB），跳過 localStorage`)
      console.log("[Qdrant Memory][migrate] 備份完整資料（請手動保存）：", backupPayload)
    }
  } catch (e) {
    log(`備份序列化失敗：${e?.message || e}`)
    console.log("[Qdrant Memory][migrate] 備份完整資料（請手動保存）：", { collection: collectionName, points: allPoints })
  }

  // 4) 刪除舊集合
  log(`刪除舊集合 ${collectionName}...`)
  const delRes = await fetch(`${settings.qdrantUrl}/collections/${encodeCollectionName(collectionName)}`, {
    method: "DELETE",
    headers: getQdrantHeaders(),
  })
  if (!delRes.ok) {
    const errText = await delRes.text().catch(() => "")
    log(`刪除失敗：${delRes.status} ${errText}（備份在 ${backupKey}）`)
    return { ok: false, reason: "delete-failed", error: errText, backupKey }
  }
  invalidateCollectionInfo(collectionName)

  // 5) 建立新 hybrid 集合
  log(`建立 hybrid 集合（dimensions=${info.vectorSize}）...`)
  const created = await createCollection(collectionName, info.vectorSize, true)
  if (!created) {
    log(`建立失敗（備份在 ${backupKey}，可重試）`)
    return { ok: false, reason: "create-failed", backupKey }
  }

  // 6) 批次重 upsert（每批 32 點，避免 payload 太大）
  const upsertBatchSize = 32
  let upserted = 0
  for (let i = 0; i < allPoints.length; i += upsertBatchSize) {
    const slice = allPoints.slice(i, i + upsertBatchSize)
    const newPoints = slice.map((p) => {
      // 舊 vector 可能是陣列（unnamed dense），也可能是 { "": [...] } 格式
      let denseVec = p.vector
      if (denseVec && typeof denseVec === "object" && !Array.isArray(denseVec)) {
        // 取第一個值
        const keys = Object.keys(denseVec)
        denseVec = keys.length > 0 ? denseVec[keys[0]] : null
      }
      const text = p.payload?.text || ""
      const sparse = makeBM25SparseVector(text)
      return {
        id: p.id,
        vector: {
          [BM25_DENSE_VECTOR_NAME]: denseVec,
          [BM25_SPARSE_VECTOR_NAME]: { indices: sparse.indices, values: sparse.values },
        },
        payload: p.payload || {},
      }
    })

    const upsertRes = await fetch(`${settings.qdrantUrl}/collections/${encodeCollectionName(collectionName)}/points`, {
      method: "PUT",
      headers: getQdrantHeaders(),
      body: JSON.stringify({ points: newPoints }),
    })
    if (!upsertRes.ok) {
      const errText = await upsertRes.text().catch(() => "")
      log(`第 ${i}-${i + slice.length} 批 upsert 失敗：${errText}（備份在 ${backupKey}）`)
      return { ok: false, reason: "upsert-failed", error: errText, backupKey, upserted }
    }
    upserted += slice.length
    log(`已 upsert ${upserted}/${allPoints.length}`)
  }

  invalidateCollectionInfo(collectionName)
  log(`✓ migration 完成：${collectionName}（${upserted} 點位，備份 key=${backupKey}）`)
  return { ok: true, collection: collectionName, upserted, backupKey }
}

// ============================================================================
// 生成攔截器（v3.2.0：支援 {{qdrant}} 巨集）
// ============================================================================

globalThis.qdrantMemoryInterceptor = async (chat, contextSize, abort, type) => {
  if (!settings.enabled) {
    if (settings.debugMode) {
      console.log("[Qdrant Memory] Extension disabled, skipping")
    }
    return
  }

  if (settings.preventDuplicateInjection) {
    const chatHash = getChatHash(chat)

    if (memoryInjectionTracker.has(chatHash)) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Memories already injected for this chat state, skipping")
      }
      return
    }

    memoryInjectionTracker.add(chatHash)

    if (memoryInjectionTracker.size > 50) {
      const oldestHash = memoryInjectionTracker.values().next().value
      memoryInjectionTracker.delete(oldestHash)
    }
  }

  try {
    const context = getContext()
    const characterName = context.name2

    if (!characterName) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] No character selected, skipping")
      }
      return
    }

    const lastUserMsg = chat
      .slice()
      .reverse()
      .find((msg) => msg.is_user)
    if (!lastUserMsg || !lastUserMsg.mes) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] No user message found, skipping")
      }
      return
    }

    const query = lastUserMsg.mes

    if (settings.debugMode) {
      console.log("[Qdrant Memory] Generation interceptor triggered")
      console.log("[Qdrant Memory] Type:", type)
      console.log("[Qdrant Memory] Context size:", contextSize)
      console.log("[Qdrant Memory] Searching for:", query)
      console.log("[Qdrant Memory] Character:", characterName)
    }

    const memories = await searchMemories(query, characterName)
    const memoryText = memories.length > 0 ? formatMemories(memories) : ""

    if (settings.debugMode && memories.length > 0) {
      console.log("[Qdrant Memory] Retrieved memories:", memoryText)
    }

    // v3.2.0：把記憶套用到 chat（巨集優先，否則 memoryPosition）
    const injectionMode = applyMemoryInjection(chat, memoryText)

    if (memories.length > 0) {
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] 已注入 ${memories.length} 則記憶（模式：${injectionMode}）`)
      }

      const toastr = window.toastr
      if (settings.showMemoryNotifications) {
        const modeLabel = injectionMode === "macro" ? "（{{qdrant}}）" : ""
        toastr.info(`已取得 ${memories.length} 則相關記憶${modeLabel}`, "Qdrant Memory", { timeOut: 2000 })
      }
    } else {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] No relevant memories found")
      }
    }
  } catch (error) {
    console.error("[Qdrant Memory] Error in generation interceptor:", error)
  }
}

// ============================================================================
// 自動記憶建立
// ============================================================================

function clearPendingAssistantFinalize() {
  if (pendingAssistantFinalize?.pollTimerId) {
    clearInterval(pendingAssistantFinalize.pollTimerId)
  }
  pendingAssistantFinalize = null
}

function scheduleFinalizeLastAssistantMessage(messageId, characterName) {
  const context = getContext()
  const chat = context.chat || []

  if (chat.length === 0) {
    if (settings.debugMode) {
      console.log("[Qdrant Memory] No chat messages to finalize")
    }
    return
  }

  const lastMessage = chat[chat.length - 1]

  if (lastMessage.is_user) {
    if (settings.debugMode) {
      console.log("[Qdrant Memory] Last message is user message, skipping finalize")
    }
    return
  }

  const initialText = lastMessage?.mes || ""

  clearPendingAssistantFinalize()

  const pollInterval = settings.streamFinalizePollMs || 250
  const stableMs = settings.streamFinalizeStableMs || 1200
  const maxWaitMs = settings.streamFinalizeMaxWaitMs || 300000

  if (settings.debugMode) {
    console.log("[Qdrant Memory] Starting stream finalize poll for character message")
    console.log("[Qdrant Memory] Initial text length:", initialText.length)
  }

  pendingAssistantFinalize = {
    messageId,
    characterName,
    startedAt: Date.now(),
    lastText: initialText,
    lastChangeAt: Date.now(),
    pollTimerId: null,
  }

  const finalizeAssistant = (text, reason) => {
    if (!text || text.trim().length === 0) {
      if (settings.debugMode) {
        console.warn("[Qdrant Memory] Attempted to finalize empty message, skipping")
      }
      clearPendingAssistantFinalize()
      return
    }

    if (settings.debugMode) {
      console.log(`[Qdrant Memory] Finalizing character message (${reason})`)
      console.log(`[Qdrant Memory] Final text length: ${text.length}`)
      console.log(`[Qdrant Memory] Text preview: "${text.substring(0, 100)}..."`)
    }

    bufferMessage(text, characterName, false, messageId)

    if (settings.flushAfterAssistant && messageBuffer.length >= 2) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Flushing buffer after assistant message")
      }
      processMessageBuffer()
    }

    clearPendingAssistantFinalize()
  }

  let pollCount = 0

  pendingAssistantFinalize.pollTimerId = setInterval(() => {
    pollCount++

    const currentContext = getContext()
    const currentChat = currentContext.chat || []

    if (currentChat.length === 0) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Chat is empty, cancelling finalize")
      }
      clearPendingAssistantFinalize()
      return
    }

    const currentLastMessage = currentChat[currentChat.length - 1] || {}
    const currentText = currentLastMessage.mes || ""
    const now = Date.now()

    if (settings.debugMode && pollCount % 16 === 0) {
      console.log(`[Qdrant Memory] Stream poll check #${pollCount}: length=${currentText.length}`)
    }

    if (currentText !== pendingAssistantFinalize.lastText) {
      if (settings.debugMode && pollCount % 4 === 0) {
        console.log(`[Qdrant Memory] Text changed: ${pendingAssistantFinalize.lastText.length} → ${currentText.length}`)
      }
      pendingAssistantFinalize.lastText = currentText
      pendingAssistantFinalize.lastChangeAt = now
    }

    const stableDuration = now - pendingAssistantFinalize.lastChangeAt
    const totalDuration = now - pendingAssistantFinalize.startedAt

    if (currentLastMessage.is_user) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Detected new user message, finalizing previous assistant message")
      }
      finalizeAssistant(pendingAssistantFinalize.lastText, "new user message detected")
      return
    }

    if (stableDuration >= stableMs) {
      finalizeAssistant(pendingAssistantFinalize.lastText, `stable for ${stableDuration}ms`)
      return
    }

    if (totalDuration >= maxWaitMs) {
      if (settings.debugMode) {
        console.warn(`[Qdrant Memory] Max wait (${maxWaitMs}ms) reached while finalizing assistant message`)
      }
      finalizeAssistant(pendingAssistantFinalize.lastText, "max wait reached")
    }
  }, pollInterval)
}

function onMessageSent() {
  if (!settings.enabled) return
  if (!settings.autoSaveMemories) return

  try {
    const context = getContext()
    const chat = context.chat || []
    const characterName = context.name2

    if (!characterName || chat.length === 0) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] No character or empty chat, skipping save")
      }
      return
    }

    const lastMessage = chat[chat.length - 1]

    const normalizedDate = normalizeTimestamp(lastMessage.send_date || Date.now())

    const messageId = `${characterName}_${normalizedDate}_${chat.length - 1}`

    if (!lastMessage.mes || lastMessage.mes.trim().length === 0) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Empty message, skipping")
      }
      return
    }

    const isUser = lastMessage.is_user || false

    if (settings.debugMode) {
      console.log(`[Qdrant Memory] onMessageSent - isUser: ${isUser}, length: ${lastMessage.mes.length}`)
    }

    if (isUser) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Buffering user message immediately")
      }
      bufferMessage(lastMessage.mes, characterName, true, messageId)
    } else {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Character message detected, scheduling finalize poll")
      }
      scheduleFinalizeLastAssistantMessage(messageId, characterName)
    }
  } catch (error) {
    console.error("[Qdrant Memory] Error in onMessageSent:", error)
  }
}

// ============================================================================
// 記憶檢視器
// ============================================================================

async function getCollectionInfo(collectionName) {
  try {
    const response = await fetch(`${settings.qdrantUrl}/collections/${encodeCollectionName(collectionName)}`, {
      headers: getQdrantHeaders(),
    })
    if (response.ok) {
      const data = await response.json()
      return data.result
    }
    return null
  } catch (error) {
    console.error("[Qdrant Memory] Error getting collection info:", error)
    return null
  }
}

async function deleteCollection(collectionName) {
  try {
    const response = await fetch(`${settings.qdrantUrl}/collections/${encodeCollectionName(collectionName)}`, {
      method: "DELETE",
      headers: getQdrantHeaders(),
    })
    return response.ok
  } catch (error) {
    console.error("[Qdrant Memory] Error deleting collection:", error)
    return false
  }
}

async function showMemoryViewer() {
  const context = getContext()
  const characterName = context.name2

  if (!characterName) {
    const toastr = window.toastr
    toastr.warning("尚未選擇角色", "Qdrant Memory")
    return
  }

  const collectionName = getCollectionName(characterName)
  const info = await getCollectionInfo(collectionName)

  if (!info) {
    const toastr = window.toastr
    toastr.warning(`找不到 ${characterName} 的記憶`, "Qdrant Memory")
    return
  }

  const count = info.points_count || 0

  const modalHtml = `
        <div id="qdrant_modal" style="
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 10000;
            max-width: 500px;
            width: 90%;
        ">
            <div style="color: #333;">
                <h3 style="margin-top: 0;">記憶檢視器 - ${characterName}</h3>
                <p><strong>集合：</strong>${collectionName}</p>
                <p><strong>記憶總數：</strong>${count}</p>
                <div style="margin-top: 20px; display: flex; gap: 10px;">
                    <button id="qdrant_delete_collection_btn" class="menu_button" style="background-color: #dc3545; color: white;">
                        刪除全部記憶
                    </button>
                    <button id="qdrant_close_modal" class="menu_button">
                        關閉
                    </button>
                </div>
            </div>
        </div>
        <div id="qdrant_overlay" style="
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 9999;
        "></div>
    `

  const $ = window.$
  $("body").append(modalHtml)

  $("#qdrant_close_modal, #qdrant_overlay").on("click", () => {
    $("#qdrant_modal").remove()
    $("#qdrant_overlay").remove()
  })

  $("#qdrant_delete_collection_btn").on("click", async function () {
    const confirmed = confirm(
      `確定要刪除 ${characterName} 的所有記憶嗎？此操作無法復原！`,
    )
    if (confirmed) {
      $(this).prop("disabled", true).text("刪除中...")
      const success = await deleteCollection(collectionName)
      if (success) {
        const toastr = window.toastr
        toastr.success(`已刪除 ${characterName} 的所有記憶`, "Qdrant Memory")
        $("#qdrant_modal").remove()
        $("#qdrant_overlay").remove()
      } else {
        const toastr = window.toastr
        toastr.error("刪除記憶失敗", "Qdrant Memory")
        $(this).prop("disabled", false).text("刪除全部記憶")
      }
    }
  })
}

// ============================================================================
// 工具函式
// ============================================================================

function getContext() {
  const SillyTavern = window.SillyTavern

  if (typeof SillyTavern !== "undefined" && SillyTavern.getContext) {
    return SillyTavern.getContext()
  }
  return {
    chat: window.chat || [],
    name2: window.name2 || "",
    characters: window.characters || [],
  }
}

function getPersonaName() {
  const context = getContext()

  const personaName =
    context.name1 ||
    context.persona?.name ||
    window.name1 ||
    window.SillyTavern?.getContext?.()?.name1 ||
    "User"

  if (settings.debugMode && personaName !== "User") {
    console.log(`[Qdrant Memory] Using persona name: ${personaName}`)
  }

  return personaName
}

function generateUUID() {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    var r = (Math.random() * 16) | 0,
      v = c == "x" ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

async function processSaveQueue() {
  if (processingSaveQueue || saveQueue.length === 0) return
  processingSaveQueue = true
  while (saveQueue.length > 0) {
    saveQueue.shift()
  }
  processingSaveQueue = false
}

// ============================================================================
// 設定 UI
// ============================================================================

function createSettingsUI() {
  const settingsHtml = `
        <div class="qdrant-memory-settings">
            <div class="inline-drawer">
                <div class="inline-drawer-toggle inline-drawer-header">
                    <b>Qdrant Memory</b>
                    <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
                </div>
                <div class="inline-drawer-content">
                    <p style="margin: 10px 0; color: #666; font-size: 0.9em;">
                        具備時間情境的自動記憶建立
                    </p>

                    <div style="margin: 15px 0;">
                        <label style="display: flex; align-items: center; gap: 10px;">
                            <input type="checkbox" id="qdrant_enabled" ${settings.enabled ? "checked" : ""} />
                            <strong>啟用 Qdrant Memory</strong>
                        </label>
                    </div>

            <hr style="margin: 15px 0;" />

            <h4>連線設定</h4>

            <div style="margin: 10px 0;">
                <label><strong>Qdrant 網址：</strong></label>
                <input type="text" id="qdrant_url" class="text_pole" value="${settings.qdrantUrl}"
                       style="width: 100%; margin-top: 5px;"
                       placeholder="http://localhost:6333" />
                <small style="color: #666;">你的 Qdrant 實例網址</small>
            </div>

             <div style="margin: 10px 0;">
                <label><strong>Qdrant API 金鑰：</strong></label>
                <input type="password" id="qdrant_api_key" class="text_pole" value="${settings.qdrantApiKey || ""}"
                       style="width: 100%; margin-top: 5px;"
                       placeholder="非必填，如不需要可留空" />
                <small style="color: #666;">Qdrant 驗證用 API 金鑰（選填）</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>基礎集合名稱：</strong></label>
                <input type="text" id="qdrant_collection" class="text_pole" value="${settings.collectionName}"
                       style="width: 100%; margin-top: 5px;"
                       placeholder="sillytavern_memories" />
                <small style="color: #666;">集合的基礎名稱（角色名會自動附加在後，支援中文）</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>嵌入服務提供者：</strong></label>
                <select id="qdrant_embedding_provider" class="text_pole" style="width: 100%; margin-top: 5px;">
                    <option value="openai" ${settings.embeddingProvider === "openai" ? "selected" : ""}>OpenAI</option>
                    <option value="openrouter" ${settings.embeddingProvider === "openrouter" ? "selected" : ""}>OpenRouter</option>
                    <option value="google" ${settings.embeddingProvider === "google" ? "selected" : ""}>Google AI Studio</option>
                    <option value="local" ${settings.embeddingProvider === "local" ? "selected" : ""}>本地／自訂端點</option>
                </select>
                <small style="color: #666;">選擇用於產生嵌入向量的 API</small>
            </div>

            <div id="qdrant_openai_key_group" style="margin: 10px 0;">
                <label><strong>OpenAI API 金鑰：</strong></label>
                <input type="password" id="qdrant_openai_key" class="text_pole" value="${settings.openaiApiKey}"
                       placeholder="sk-..." style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">使用 OpenAI 時必填</small>
            </div>

            <div id="qdrant_openrouter_key_group" style="margin: 10px 0; display: none;">
                <label><strong>OpenRouter API 金鑰：</strong></label>
                <input type="password" id="qdrant_openrouter_key" class="text_pole" value="${settings.openRouterApiKey}"
                       placeholder="or-..." style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">使用 OpenRouter 時必填</small>
            </div>

            <!-- v3.2.0：Google AI Studio 設定區 -->
            <div id="qdrant_google_key_group" style="margin: 10px 0; display: none;">
                <label><strong>Google API 金鑰：</strong></label>
                <input type="password" id="qdrant_google_key" class="text_pole" value="${settings.googleApiKey || ""}"
                       placeholder="AIza..." style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">在 <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener">Google AI Studio</a> 取得（免費額度）</small>
                <div style="margin-top: 8px; display: flex; gap: 8px; align-items: center;">
                    <button id="qdrant_google_fetch_models" class="menu_button" type="button">抓取模型列表</button>
                    <span id="qdrant_google_fetch_status" style="font-size: 0.85em; color: #666;"></span>
                </div>
            </div>

            <div id="qdrant_local_url_group" style="margin: 10px 0; display: none;">
                <label><strong>嵌入端點網址：</strong></label>
                <input type="text" id="qdrant_local_url" class="text_pole" value="${settings.localEmbeddingUrl}"
                       placeholder="http://localhost:11434/api/embeddings"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">接受 OpenAI 相容嵌入請求的端點</small>
            </div>

            <div id="qdrant_local_api_key_group" style="margin: 10px 0; display: none;">
                <label><strong>嵌入 API 金鑰（選填）：</strong></label>
                <input type="password" id="qdrant_local_api_key" class="text_pole" value="${settings.localEmbeddingApiKey}"
                       placeholder="本地端點所需的 Bearer 權杖"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">若你的本地／自訂端點需要驗證才填寫</small>
            </div>

            <div id="qdrant_local_dimensions_group" style="margin: 10px 0; display: none;">
                <label><strong>嵌入向量維度：</strong></label>
                <input type="number" id="qdrant_local_dimensions" class="text_pole"
                       value="${settings.customEmbeddingDimensions ?? ""}"
                       min="1" step="1" style="width: 100%; margin-top: 5px;" placeholder="首次呼叫後自動偵測" />
                <small style="color: #666;">自訂嵌入模型回傳的向量維度（留空則自動偵測）</small>
            </div>

<div id="qdrant_local_model_group" style="margin: 10px 0; display: none;">
                <label><strong>嵌入模型名稱：</strong></label>
                <input type="text" id="qdrant_local_model" class="text_pole" value="${settings.embeddingModel}"
                       placeholder="text-embedding-004"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">API 請求所使用的模型名稱（如 text-embedding-004、gemini-embedding-001）</small>
            </div>

            <div id="qdrant_embedding_model_group" style="margin: 10px 0;">
                <label><strong>嵌入模型：</strong></label>
                <select id="qdrant_embedding_model" class="text_pole" style="width: 100%; margin-top: 5px;">
                    <option value="text-embedding-3-large" ${settings.embeddingModel === "text-embedding-3-large" ? "selected" : ""}>text-embedding-3-large（最高品質）</option>
                    <option value="text-embedding-3-small" ${settings.embeddingModel === "text-embedding-3-small" ? "selected" : ""}>text-embedding-3-small（速度較快）</option>
                    <option value="text-embedding-ada-002" ${settings.embeddingModel === "text-embedding-ada-002" ? "selected" : ""}>text-embedding-ada-002（舊版）</option>
                </select>
            </div>

            <hr style="margin: 15px 0;" />

            <h4>記憶檢索設定</h4>

            <!-- v3.2.0：{{qdrant}} 巨集說明 -->
            <div style="margin: 10px 0; padding: 10px; background: #fff8e1; border-left: 4px solid #ffc107; border-radius: 4px;">
                <p style="margin: 0 0 6px 0; font-size: 0.9em;"><strong>💡 進階：用 <code>{{qdrant}}</code> 巨集自訂注入位置</strong></p>
                <p style="margin: 0; font-size: 0.85em; color: #555; line-height: 1.5;">
                    在角色卡的描述、世界書條目、Author's Note 或任何訊息中放入 <code>{{qdrant}}</code>，記憶就會被替換到那個確切位置。<br>
                    沒有放 <code>{{qdrant}}</code> 時，會用下方「記憶插入位置」設定退回預設行為。<br>
                    範例：在 Author's Note 寫「<code>過去的記憶：{{qdrant}}</code>」就能讓記憶以你想要的格式呈現。
                </p>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>檢索記憶數量：</strong> <span id="memory_limit_display">${settings.memoryLimit}</span></label>
                <input type="range" id="qdrant_memory_limit" min="1" max="50" value="${settings.memoryLimit}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">每次生成最多檢索的記憶數</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>相關度門檻：</strong> <span id="score_threshold_display">${settings.scoreThreshold}</span></label>
                <input type="range" id="qdrant_score_threshold" min="0" max="1" step="0.05" value="${settings.scoreThreshold}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">最低相似度分數（0.0 - 1.0）</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>記憶插入位置：</strong> <span id="memory_position_display">${settings.memoryPosition}</span></label>
                <input type="range" id="qdrant_memory_position" min="1" max="30" value="${settings.memoryPosition}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">從最後一則往前數，插入記憶的位置（沒用 {{qdrant}} 巨集時生效）</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>保留近期訊息：</strong> <span id="retain_recent_display">${settings.retainRecentMessages}</span></label>
                <input type="range" id="qdrant_retain_recent" min="0" max="50" value="${settings.retainRecentMessages}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">從檢索中排除最後 N 則訊息（0 表示不排除）</small>
            </div>

            <hr style="margin: 15px 0;" />

            <h4>自動記憶建立</h4>

            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_per_character" ${settings.usePerCharacterCollections ? "checked" : ""} />
                    <strong>每個角色使用獨立集合</strong>
                </label>
                <small style="color: #666; display: block; margin-left: 30px;">每個角色擁有自己的專屬集合（建議啟用）</small>
            </div>

            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_auto_save" ${settings.autoSaveMemories ? "checked" : ""} />
                    <strong>自動儲存記憶</strong>
                </label>
                <small style="color: #666; display: block; margin-left: 30px;">對話進行時自動將訊息儲存到 Qdrant</small>
            </div>

            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_save_user" ${settings.saveUserMessages ? "checked" : ""} />
                    儲存使用者訊息
                </label>
            </div>

            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_save_character" ${settings.saveCharacterMessages ? "checked" : ""} />
                    儲存角色訊息
                </label>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>最小訊息長度：</strong> <span id="min_message_length_display">${settings.minMessageLength}</span></label>
                <input type="range" id="qdrant_min_length" min="5" max="50" value="${settings.minMessageLength}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">儲存訊息所需的最少字元數</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>去重閾值：</strong> <span id="dedupe_threshold_display">${settings.dedupeThreshold}</span></label>
                <input type="range" id="qdrant_dedupe_threshold" min="0.80" max="1.00" step="0.01" value="${settings.dedupeThreshold}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">避免儲存重複的記憶區塊（越高越嚴格）</small>
            </div>

            <hr style="margin: 15px 0;" />

            <!-- v3.3.0：BM25 Hybrid 搜尋 -->
            <h4>BM25 混合搜尋（v3.3.0）</h4>

            <div style="margin: 10px 0; padding: 10px; background: #e7f3ff; border-left: 3px solid #2196F3; border-radius: 4px; font-size: 0.9em; color: #333;">
                <strong>真 BM25 hybrid 搜尋</strong>：dense 向量 + sparse BM25（client tokenize + Qdrant IDF modifier）+ RRF fusion。<br />
                需要 <strong>Qdrant 1.10+</strong>。打開後新建立的集合會用 hybrid schema；舊集合需要按下方按鈕做 migration（會重建集合，有 localStorage 備份）。
            </div>

            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_bm25_hybrid" ${settings.useBM25Hybrid ? "checked" : ""} />
                    <strong>啟用 BM25 hybrid 搜尋</strong>
                </label>
                <small style="color: #666; display: block; margin-left: 30px;">關閉時退回純向量搜尋（v3.2.0 行為），舊集合不受影響</small>
            </div>

            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_bm25_cjk_bigram" ${settings.bm25CjkBigram ? "checked" : ""} />
                    中文加入 bigram（相鄰二字）
                </label>
                <small style="color: #666; display: block; margin-left: 30px;">提升中文短語精準度，sparse vector 會變大但仍很省（建議開）</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>Dense prefetch 數：</strong> <span id="bm25_dense_topk_display">${settings.bm25DenseTopK}</span></label>
                <input type="range" id="qdrant_bm25_dense_topk" min="5" max="50" step="1" value="${settings.bm25DenseTopK}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">RRF 前 dense 通道取多少候選</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>Sparse prefetch 數：</strong> <span id="bm25_sparse_topk_display">${settings.bm25SparseTopK}</span></label>
                <input type="range" id="qdrant_bm25_sparse_topk" min="5" max="50" step="1" value="${settings.bm25SparseTopK}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">RRF 前 BM25 通道取多少候選</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>BM25 k1（TF saturation）：</strong> <span id="bm25_k1_display">${settings.bm25K1}</span></label>
                <input type="range" id="qdrant_bm25_k1" min="0.5" max="2.5" step="0.1" value="${settings.bm25K1}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">BM25 詞頻飽和參數（標準範圍 1.2 ~ 2.0）</small>
            </div>

            <div style="margin: 10px 0; padding: 10px; background: #fff3cd; border-left: 3px solid #ffc107; border-radius: 4px; font-size: 0.9em; color: #333;">
                <strong>升級此角色集合至 hybrid schema</strong><br />
                此操作會：① scroll 全部點位 → ② 寫入 localStorage 備份 → ③ 刪除舊集合 → ④ 建立新格式集合 → ⑤ 重新 upsert 並補上 BM25 sparse vector。<br />
                失敗時備份保留在 localStorage（key 形如 <code>qdrant_bm25_backup_&lt;collection&gt;_&lt;ts&gt;</code>），打 console 可看到。
            </div>

            <div style="margin: 10px 0; display: flex; gap: 10px; flex-wrap: wrap;">
                <button id="qdrant_bm25_migrate" class="menu_button" style="background: #ff9800; color: white;">升級此角色集合</button>
                <button id="qdrant_bm25_check" class="menu_button">檢查目前集合格式</button>
            </div>

            <hr style="margin: 15px 0;" />

            <h4>其他設定</h4>

            <div style="margin: 15px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_prevent_duplicate" ${settings.preventDuplicateInjection ? "checked" : ""} />
                    防止重複注入記憶
                </label>
                <small style="color: #666; display: block; margin-left: 30px;">防止同一段記憶被多次加入情境</small>
            </div>

            <div style="margin: 15px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_notifications" ${settings.showMemoryNotifications ? "checked" : ""} />
                    顯示記憶通知
                </label>
            </div>

            <div style="margin: 15px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_debug" ${settings.debugMode ? "checked" : ""} />
                    除錯模式（請查看主控台）
                </label>
            </div>

            <hr style="margin: 15px 0;" />

            <div style="margin: 15px 0; display: flex; gap: 10px; flex-wrap: wrap;">
                <button id="qdrant_test" class="menu_button">測試連線</button>
                <button id="qdrant_save" class="menu_button">儲存設定</button>
                <button id="qdrant_view_memories" class="menu_button">檢視記憶</button>
                <button id="qdrant_index_chats" class="menu_button" style="background-color: #28a745; color: white;">索引角色聊天紀錄</button>
            </div>

            <div id="qdrant_status" style="margin-top: 10px; padding: 10px; border-radius: 5px;"></div>
                </div>
            </div>
        </div>
    `

  const $ = window.$
  $("#extensions_settings2").append(settingsHtml)

  if (typeof window.applyInlineDrawerListeners === "function") {
    window.applyInlineDrawerListeners()
  }

  function updateEmbeddingModelOptions(provider) {
    const models = EMBEDDING_MODEL_OPTIONS[provider] || EMBEDDING_MODEL_OPTIONS.openai
    const $modelSelect = $("#qdrant_embedding_model")
    let previousValue = settings.embeddingModel
    if (provider === "openrouter" && OPENROUTER_MODEL_ALIASES[previousValue]) {
      previousValue = OPENROUTER_MODEL_ALIASES[previousValue]
      settings.embeddingModel = previousValue
    } else if (provider === "openai" && OPENAI_MODEL_ALIASES[previousValue]) {
      previousValue = OPENAI_MODEL_ALIASES[previousValue]
      settings.embeddingModel = previousValue
    }
    let matched = false

    $modelSelect.empty()

    // v3.2.0：Google 還沒抓模型時顯示提示
    if (provider === "google" && models.length === 0) {
      $modelSelect.append(`<option value="">（請先點選「抓取模型列表」）</option>`)
      return
    }

    models.forEach((model) => {
      const isSelected = model.value === previousValue
      if (isSelected) {
        matched = true
      }

      const optionHtml = `<option value="${model.value}"${
        isSelected ? " selected" : ""
      }>${model.label}</option>`
      $modelSelect.append(optionHtml)
    })

    if (!matched && models.length > 0) {
      const fallback = DEFAULT_MODEL_BY_PROVIDER[provider] || models[0].value
      // 如果 fallback 在清單中就用它，否則用第一個
      const inList = models.some(m => m.value === fallback)
      settings.embeddingModel = inList ? fallback : models[0].value
      $modelSelect.val(settings.embeddingModel)
    }
  }

  function updateEmbeddingProviderUI() {
    const provider = settings.embeddingProvider || "openai"
    const $openAIGroup = $("#qdrant_openai_key_group")
    const $openRouterGroup = $("#qdrant_openrouter_key_group")
    const $googleGroup = $("#qdrant_google_key_group") // v3.2.0
    const $localGroup = $("#qdrant_local_url_group")
    const $localApiKeyGroup = $("#qdrant_local_api_key_group")
    const $localDimensionsGroup = $("#qdrant_local_dimensions_group")
    const $localDimensionsInput = $("#qdrant_local_dimensions")
    const $modelGroup = $("#qdrant_embedding_model_group")

    $openAIGroup.toggle(provider === "openai")
    $openRouterGroup.toggle(provider === "openrouter")
    $googleGroup.toggle(provider === "google") // v3.2.0
    $localGroup.toggle(provider === "local")
    $localApiKeyGroup.toggle(provider === "local")
    $localDimensionsGroup.toggle(provider === "local")

    $("#qdrant_local_model_group").toggle(provider === "local")

    if (provider === "local") {
      $localDimensionsInput.val(settings.customEmbeddingDimensions ?? "")
    }

    // v3.2.0：Google 也要顯示模型下拉選單
    const showModelSelect = provider !== "local"
    $modelGroup.toggle(showModelSelect)

    if (showModelSelect) {
      updateEmbeddingModelOptions(provider)
    }
  }

  // 事件處理
  $("#qdrant_enabled").on("change", function () {
    settings.enabled = $(this).is(":checked")
  })

  $("#qdrant_url").on("input", function () {
    settings.qdrantUrl = $(this).val()
  })

  $("#qdrant_api_key").on("input", function () {
    settings.qdrantApiKey = $(this).val()
  })

  $("#qdrant_collection").on("input", function () {
    settings.collectionName = $(this).val()
  })

  $("#qdrant_embedding_provider").on("change", function () {
    settings.embeddingProvider = $(this).val()
    updateEmbeddingProviderUI()
  })

  $("#qdrant_openai_key").on("input", function () {
    settings.openaiApiKey = $(this).val()
  })

  $("#qdrant_openrouter_key").on("input", function () {
    settings.openRouterApiKey = $(this).val()
  })

  // v3.2.0：Google 金鑰輸入
  $("#qdrant_google_key").on("input", function () {
    settings.googleApiKey = $(this).val()
  })

  // v3.2.0：抓取 Google 模型列表
  $("#qdrant_google_fetch_models").on("click", async function () {
    const $btn = $(this)
    const $status = $("#qdrant_google_fetch_status")
    const apiKey = settings.googleApiKey

    if (!apiKey || !apiKey.trim()) {
      $status.text("請先填入 API 金鑰").css("color", "#c62828")
      return
    }

    $btn.prop("disabled", true).text("抓取中...")
    $status.text("正在連線到 Google AI Studio...").css("color", "#1565c0")

    try {
      const models = await fetchGoogleEmbeddingModels(apiKey)

      if (!models || models.length === 0) {
        $status.text("未找到任何支援 embedContent 的模型").css("color", "#c62828")
        return
      }

      // 寫入快取，重新渲染下拉選單
      EMBEDDING_MODEL_OPTIONS.google = models
      updateEmbeddingModelOptions("google")

      // 同步 settings.embeddingModel 與目前選到的值
      const currentVal = $("#qdrant_embedding_model").val()
      if (currentVal) {
        settings.embeddingModel = currentVal
      }

      $status.text(`✓ 找到 ${models.length} 個模型`).css("color", "#2e7d32")

      const toastr = window.toastr
      if (toastr) {
        toastr.success(`已載入 ${models.length} 個 Google 嵌入模型`, "Qdrant Memory")
      }
    } catch (error) {
      console.error("[Qdrant Memory] 抓取 Google 模型失敗：", error)
      $status.text(`✗ 抓取失敗：${error.message}`).css("color", "#c62828")
    } finally {
      $btn.prop("disabled", false).text("抓取模型列表")
    }
  })

  $("#qdrant_local_url").on("input", function () {
    settings.localEmbeddingUrl = $(this).val()
  })

  $("#qdrant_local_api_key").on("input", function () {
    settings.localEmbeddingApiKey = $(this).val()
  })

  $("#qdrant_local_dimensions").on("input", function () {
    const value = Number.parseInt($(this).val(), 10)
    settings.customEmbeddingDimensions = Number.isFinite(value) && value > 0 ? value : null
  })

  $("#qdrant_local_model").on("input", function () {
    settings.embeddingModel = $(this).val()
  })

  $("#qdrant_embedding_model").on("change", function () {
    settings.embeddingModel = $(this).val()
  })

  $("#qdrant_memory_limit").on("input", function () {
    settings.memoryLimit = Number.parseInt($(this).val())
    $("#memory_limit_display").text(settings.memoryLimit)
  })

  $("#qdrant_score_threshold").on("input", function () {
    settings.scoreThreshold = Number.parseFloat($(this).val())
    $("#score_threshold_display").text(settings.scoreThreshold)
  })

  $("#qdrant_memory_position").on("input", function () {
    settings.memoryPosition = Number.parseInt($(this).val())
    $("#memory_position_display").text(settings.memoryPosition)
  })

  $("#qdrant_retain_recent").on("input", function () {
    settings.retainRecentMessages = Number.parseInt($(this).val())
    $("#retain_recent_display").text(settings.retainRecentMessages)
  })

  $("#qdrant_per_character").on("change", function () {
    settings.usePerCharacterCollections = $(this).is(":checked")
  })

  $("#qdrant_auto_save").on("change", function () {
    settings.autoSaveMemories = $(this).is(":checked")
  })

  $("#qdrant_save_user").on("change", function () {
    settings.saveUserMessages = $(this).is(":checked")
  })

  $("#qdrant_save_character").on("change", function () {
    settings.saveCharacterMessages = $(this).is(":checked")
  })

  $("#qdrant_min_length").on("input", function () {
    settings.minMessageLength = Number.parseInt($(this).val())
    $("#min_message_length_display").text(settings.minMessageLength)
  })

  $("#qdrant_dedupe_threshold").on("input", function () {
    settings.dedupeThreshold = Number.parseFloat($(this).val())
    $("#dedupe_threshold_display").text(settings.dedupeThreshold.toFixed(2))
  })

  // v3.3.0：BM25 hybrid 事件處理
  $("#qdrant_bm25_hybrid").on("change", function () {
    settings.useBM25Hybrid = $(this).is(":checked")
    if (settings.useBM25Hybrid) {
      $("#qdrant_status")
        .text("已啟用 BM25 hybrid。新建集合會用新格式；舊集合請按「升級此角色集合」。")
        .css({ color: "#004085", background: "#cce5ff", border: "1px solid #004085" })
    }
  })

  $("#qdrant_bm25_cjk_bigram").on("change", function () {
    settings.bm25CjkBigram = $(this).is(":checked")
  })

  $("#qdrant_bm25_dense_topk").on("input", function () {
    settings.bm25DenseTopK = Number.parseInt($(this).val())
    $("#bm25_dense_topk_display").text(settings.bm25DenseTopK)
  })

  $("#qdrant_bm25_sparse_topk").on("input", function () {
    settings.bm25SparseTopK = Number.parseInt($(this).val())
    $("#bm25_sparse_topk_display").text(settings.bm25SparseTopK)
  })

  $("#qdrant_bm25_k1").on("input", function () {
    settings.bm25K1 = Number.parseFloat($(this).val())
    $("#bm25_k1_display").text(settings.bm25K1.toFixed(1))
  })

  $("#qdrant_bm25_check").on("click", async () => {
    const ctx = getContext()
    const characterName = ctx.name2
    if (!characterName) {
      $("#qdrant_status")
        .text("請先選擇一個角色")
        .css({ color: "#856404", background: "#fff3cd", border: "1px solid #ffc107" })
      return
    }
    const collectionName = getCollectionName(characterName)
    invalidateCollectionInfo(collectionName)
    const info = await collectionExists(collectionName)
    let msg
    if (!info.exists) {
      msg = `集合 ${collectionName} 不存在（尚未建立）`
    } else if (info.hasBM25) {
      msg = `✓ ${collectionName}：hybrid 格式（dense="${info.denseVectorName}" + bm25 sparse），維度 ${info.vectorSize}`
    } else {
      msg = `${collectionName}：legacy 純向量格式（${info.denseVectorName ? `named "${info.denseVectorName}"` : "unnamed"}），維度 ${info.vectorSize}。可按「升級」鈕轉成 hybrid。`
    }
    $("#qdrant_status")
      .text(msg)
      .css({ color: "#004085", background: "#cce5ff", border: "1px solid #004085" })
  })

  $("#qdrant_bm25_migrate").on("click", async () => {
    const ctx = getContext()
    const characterName = ctx.name2
    if (!characterName) {
      $("#qdrant_status")
        .text("請先選擇一個角色")
        .css({ color: "#856404", background: "#fff3cd", border: "1px solid #ffc107" })
      return
    }
    const collectionName = getCollectionName(characterName)

    const confirmed = window.confirm(
      `將升級集合：${collectionName}\n\n` +
      `會 scroll 全部點位 → 寫入 localStorage 備份 → 刪除舊集合 → 建立 hybrid 集合 → 重新 upsert 並補 BM25 sparse。\n\n` +
      `失敗時備份在 localStorage（key 形如 qdrant_bm25_backup_*），可從 console 救回。\n\n` +
      `確定要繼續嗎？`
    )
    if (!confirmed) return

    $("#qdrant_status")
      .text("升級中... 請看 console 進度")
      .css({ color: "#004085", background: "#cce5ff", border: "1px solid #004085" })

    const result = await migrateCollectionToBM25(characterName, (msg) => {
      $("#qdrant_status").text(`升級中：${msg}`)
    })

    if (result.ok) {
      if (result.alreadyHybrid) {
        $("#qdrant_status")
          .text(`集合 ${collectionName} 已是 hybrid 格式，無需升級。`)
          .css({ color: "green", background: "#d4edda", border: "1px solid green" })
      } else {
        $("#qdrant_status")
          .text(`✓ 升級完成：${collectionName}（${result.upserted} 點位）。備份 key：${result.backupKey}`)
          .css({ color: "green", background: "#d4edda", border: "1px solid green" })
        // 順便打開 hybrid 開關（如果還沒開）
        if (!settings.useBM25Hybrid) {
          settings.useBM25Hybrid = true
          $("#qdrant_bm25_hybrid").prop("checked", true)
        }
      }
    } else {
      $("#qdrant_status")
        .text(`✗ 升級失敗（${result.reason}）。備份 key：${result.backupKey || "(無)"}。詳細請看 console。`)
        .css({ color: "#721c24", background: "#f8d7da", border: "1px solid #721c24" })
    }
  })

  $("#qdrant_prevent_duplicate").on("change", function () {
    settings.preventDuplicateInjection = $(this).is(":checked")
  })

  $("#qdrant_notifications").on("change", function () {
    settings.showMemoryNotifications = $(this).is(":checked")
  })

  $("#qdrant_debug").on("change", function () {
    settings.debugMode = $(this).is(":checked")
  })

  updateEmbeddingProviderUI()

  // v3.2.0：若已存有 Google API Key 但模型清單空，啟動時自動抓一次
  if (
    settings.embeddingProvider === "google" &&
    settings.googleApiKey &&
    settings.googleApiKey.trim() &&
    EMBEDDING_MODEL_OPTIONS.google.length === 0
  ) {
    setTimeout(() => {
      $("#qdrant_google_fetch_models").trigger("click")
    }, 500)
  }

  $("#qdrant_save").on("click", () => {
    saveSettings()
    $("#qdrant_status")
      .text("✓ 設定已儲存！")
      .css({ color: "green", background: "#d4edda", border: "1px solid green" })
    setTimeout(() => $("#qdrant_status").text("").css({ background: "", border: "" }), 3000)
  })

  $("#qdrant_test").on("click", async () => {
    $("#qdrant_status")
      .text("正在測試連線...")
      .css({ color: "#004085", background: "#cce5ff", border: "1px solid #004085" })

    try {
      const response = await fetch(`${settings.qdrantUrl}/collections`, {
        headers: getQdrantHeaders(),
      })

      if (response.ok) {
        const data = await response.json()
        const collections = data.result?.collections || []
        $("#qdrant_status")
          .text(`✓ 連線成功！共找到 ${collections.length} 個集合。`)
          .css({ color: "green", background: "#d4edda", border: "1px solid green" })
      } else {
        $("#qdrant_status")
          .text("✗ 連線失敗，請檢查網址。")
          .css({ color: "#721c24", background: "#f8d7da", border: "1px solid #721c24" })
      }
    } catch (error) {
      $("#qdrant_status")
        .text(`✗ 錯誤：${error.message}`)
        .css({ color: "#721c24", background: "#f8d7da", border: "1px solid #721c24" })
    }
  })

  $("#qdrant_view_memories").on("click", () => {
    showMemoryViewer()
  })

  $("#qdrant_index_chats").on("click", () => {
    indexCharacterChats()
  })
}

// ============================================================================
// 擴充功能初始化
// ============================================================================

window.jQuery(async () => {
  loadSettings()
  createSettingsUI()

  const eventSource = window.eventSource
  if (typeof eventSource !== "undefined" && eventSource.on) {
    const handleMessageEvent = () => {
      if (!settings.enabled || !settings.autoSaveMemories) return
      onMessageSent()
    }

    eventSource.on("MESSAGE_RECEIVED", handleMessageEvent)
    eventSource.on("USER_MESSAGE_RENDERED", handleMessageEvent)
    console.log("[Qdrant Memory] Using eventSource hooks")
  } else {
    console.log("[Qdrant Memory] Using polling fallback for auto-save")
    let lastChatLength = 0
    setInterval(() => {
      const context = getContext()
      const chat = context.chat || []

      if (!settings.enabled || !settings.autoSaveMemories) {
        lastChatLength = chat.length
        return
      }

      if (chat.length > lastChatLength) {
        onMessageSent()
      }
      lastChatLength = chat.length
    }, 2000)
  }

  console.log("[Qdrant Memory] 擴充功能載入完成（v3.3.0 - 真 BM25 hybrid 搜尋：dense + sparse BM25 + RRF）")
})
