// Qdrant Memory Extension for SillyTavern
// This extension retrieves relevant memories from Qdrant and injects them into conversations
// Version 4.0.0 - 繁中介面、ST 伺服器端設定儲存、混合搜尋（dense + 全文 multilingual）、Cohere Rerank

const extensionName = "qdrant-memory"

// Default settings
const defaultSettings = {
  enabled: true,
  qdrantUrl: "http://localhost:6333",
  qdrantApiKey: "",
  collectionName: "mem",
  embeddingProvider: "openai",
  openaiApiKey: "",
  openRouterApiKey: "",
  localEmbeddingUrl: "",
  localEmbeddingApiKey: "",
  embeddingModel: "text-embedding-3-large",
  customEmbeddingDimensions: null,
  memoryLimit: 5,
  scoreThreshold: 0.3,
  memoryPosition: 2,
  debugMode: false,
  // New v3.0 settings
  usePerCharacterCollections: true,
  autoSaveMemories: true,
  saveUserMessages: true,
  saveCharacterMessages: true,
  minMessageLength: 5,
  showMemoryNotifications: true,
  retainRecentMessages: 5,
  chunkMinSize: 1200,
  chunkMaxSize: 1500,
  chunkTimeout: 30000, // 30 seconds - save chunk if no new messages
  // NEW v3.1.2 settings
  dedupeThreshold: 0.92, // Similarity threshold for chunk deduplication
  preventDuplicateInjection: true, // Prevent inserting memories multiple times
  streamFinalizePollMs: 250,
  streamFinalizeStableMs: 1200,
  streamFinalizeMaxWaitMs: 300000,
  flushAfterAssistant: true,
  // v4.0 全文索引與混合搜尋設定（Qdrant 1.15+）
  enableTextIndex: true,            // 建立 collection 時自動建立 multilingual 文字索引
  textIndexTokenizer: "multilingual", // multilingual | word | whitespace | prefix
  enableHybridSearch: true,         // 啟用 dense + 全文 混合搜尋
  hybridCandidateMultiplier: 4,     // 候選池大小 = memoryLimit × 此倍數
  rrfK: 60,                         // RRF 融合參數（標準值 60）
  // v4.0 Cohere Rerank 設定
  enableRerank: false,              // 啟用 Cohere Rerank
  cohereApiKey: "",
  cohereRerankModel: "rerank-multilingual-v3.0",
  rerankCandidates: 20,             // 送入 Rerank 的候選數量
}

let settings = { ...defaultSettings }
const saveQueue = []
let processingSaveQueue = false

let messageBuffer = []
let lastMessageTime = 0
let chunkTimer = null
let pendingAssistantFinalize = null

// NEW: Track which chats have had memories injected to prevent duplicates
const memoryInjectionTracker = new Set()

// Helper to create a unique hash for a chat state
function getChatHash(chat) {
  // Create a hash based on the last few messages to identify unique chat states
  const lastMessages = chat.slice(-5).map(msg => {
    return `${msg.is_user ? 'U' : 'A'}_${msg.mes?.substring(0, 50) || ''}_${msg.send_date || ''}`
  }).join('|')
  
  return lastMessages
}

const EMBEDDING_MODEL_OPTIONS = {
  openai: [
    {
      value: "text-embedding-3-large",
      label: "text-embedding-3-large (best quality)",
    },
    {
      value: "text-embedding-3-small",
      label: "text-embedding-3-small (faster)",
    },
    {
      value: "text-embedding-ada-002",
      label: "text-embedding-ada-002 (legacy)",
    },
  ],
  openrouter: [
    {
      value: "openai/text-embedding-3-large",
      label: "OpenAI: Text Embedding 3 Large",
    },
    {
      value: "openai/text-embedding-3-small",
      label: "OpenAI: Text Embedding 3 Small",
    },
    {
      value: "openai/text-embedding-ada-002",
      label: "OpenAI: Text Embedding Ada 002",
    },
    {
      value: "qwen/qwen3-embedding-8b",
      label: "Qwen: Qwen3 Embedding 8B",
    },
    {
      value: "mistralai/mistral-embed-2312",
      label: "Mistral: Mistral Embed 2312",
    },
    {
      value: "google/gemini-embedding-001",
      label: "Google: Gemini Embedding 001",
    },
  ],
}

const DEFAULT_MODEL_BY_PROVIDER = {
  openai: "text-embedding-3-large",
  openrouter: EMBEDDING_MODEL_OPTIONS.openrouter[0].value,
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
// DATE/TIMESTAMP NORMALIZATION
// ============================================================================

/**
 * Normalizes various date formats to Unix timestamp in milliseconds
 */
function normalizeTimestamp(date) {
  // Already a valid millisecond timestamp
  if (typeof date === 'number' && date > 1000000000000) {
    return date;
  }
  
  // Timestamp in seconds - convert to milliseconds
  if (typeof date === 'number' && date > 1000000000 && date < 1000000000000) {
    return date * 1000;
  }
  
  // Date object
  if (date instanceof Date) {
    const timestamp = date.getTime();
    if (!isNaN(timestamp)) {
      return timestamp;
    }
  }
  
  // String date - try to parse it
  if (typeof date === 'string' && date.trim()) {
    const parsed = new Date(date);
    const timestamp = parsed.getTime();
    if (!isNaN(timestamp)) {
      return timestamp;
    }
  }
  
  // Fallback to current time
  if (settings.debugMode) {
    console.warn('[Qdrant Memory] Could not normalize timestamp, using current time. Input:', date);
  }
  return Date.now();
}

/**
 * Formats a timestamp as YYYY-MM-DD for display in memory chunks
 */
function formatDateForChunk(timestamp) {
  try {
    const dateObj = new Date(timestamp);
    if (isNaN(dateObj.getTime())) {
      throw new Error('Invalid date');
    }
    return dateObj.toISOString().split('T')[0]; // YYYY-MM-DD format
  } catch (e) {
    console.warn('[Qdrant Memory] Error formatting date:', e, 'timestamp:', timestamp);
    return new Date().toISOString().split('T')[0]; // Fallback to today
  }
}

// ============================================================================
// SETTINGS MANAGEMENT
// ============================================================================

// 取得 SillyTavern 的 extension_settings 物件（伺服器端儲存）
function getExtensionSettingsStore() {
  // 多重後備：優先用 window 全域，再退回 context
  if (typeof window !== "undefined" && window.extension_settings && typeof window.extension_settings === "object") {
    return window.extension_settings
  }
  try {
    const ctx = window.SillyTavern?.getContext?.()
    if (ctx && ctx.extensionSettings && typeof ctx.extensionSettings === "object") {
      return ctx.extensionSettings
    }
  } catch (e) {
    // ignore
  }
  return null
}

// 取得 SillyTavern 的 saveSettingsDebounced 函式
function getSaveSettingsDebounced() {
  if (typeof window !== "undefined" && typeof window.saveSettingsDebounced === "function") {
    return window.saveSettingsDebounced
  }
  try {
    const ctx = window.SillyTavern?.getContext?.()
    if (ctx && typeof ctx.saveSettingsDebounced === "function") {
      return ctx.saveSettingsDebounced
    }
  } catch (e) {
    // ignore
  }
  return null
}

// 從 ST 伺服器端載入設定；若伺服器端沒有則嘗試從舊版 localStorage 遷移
function loadSettings() {
  const store = getExtensionSettingsStore()

  // 路徑 1：ST 伺服器端
  if (store) {
    if (store[extensionName] && typeof store[extensionName] === "object") {
      settings = { ...defaultSettings, ...store[extensionName] }
      console.log("[Qdrant Memory] 已從 ST 伺服器端載入設定")
      // 確保 store 有最新預設值（避免新增欄位空缺）
      store[extensionName] = settings
      return
    }

    // 路徑 2：伺服器端沒有 → 嘗試遷移舊版 localStorage
    try {
      const saved = localStorage.getItem(extensionName)
      if (saved) {
        const parsed = JSON.parse(saved)
        settings = { ...defaultSettings, ...parsed }
        store[extensionName] = settings
        const save = getSaveSettingsDebounced()
        if (typeof save === "function") {
          save()
        }
        // 遷移完成後清掉 localStorage
        try {
          localStorage.removeItem(extensionName)
          console.log("[Qdrant Memory] 已從 localStorage 遷移設定到 ST 伺服器端，並清除舊資料")
        } catch (e) {
          // ignore
        }
        return
      }
    } catch (e) {
      console.error("[Qdrant Memory] 遷移舊版 localStorage 設定失敗:", e)
    }

    // 路徑 3：完全沒有設定 → 使用預設值並寫入伺服器端
    settings = { ...defaultSettings }
    store[extensionName] = settings
    console.log("[Qdrant Memory] 使用預設設定（首次載入）")
    return
  }

  // 路徑 4：完全取不到 ST 伺服器端 store（極少見）→ 退回 localStorage
  console.warn("[Qdrant Memory] 無法存取 ST extension_settings，暫時使用 localStorage")
  try {
    const saved = localStorage.getItem(extensionName)
    if (saved) {
      settings = { ...defaultSettings, ...JSON.parse(saved) }
    }
  } catch (e) {
    console.error("[Qdrant Memory] 載入設定失敗:", e)
  }
}

// 儲存設定到 ST 伺服器端
function saveSettings() {
  const store = getExtensionSettingsStore()

  if (store) {
    store[extensionName] = settings
    const save = getSaveSettingsDebounced()
    if (typeof save === "function") {
      save()
      console.log("[Qdrant Memory] 設定已寫入 ST 伺服器端")
      return
    }
    // 找不到 saveSettingsDebounced → 至少 store 已更新，後續 ST 觸發儲存時會帶上
    console.warn("[Qdrant Memory] 找不到 saveSettingsDebounced，設定已更新但可能未即時寫入伺服器")
    return
  }

  // 後備：localStorage
  try {
    localStorage.setItem(extensionName, JSON.stringify(settings))
    console.warn("[Qdrant Memory] 退回 localStorage 儲存（ST extension_settings 不可用）")
  } catch (e) {
    console.error("[Qdrant Memory] 儲存設定失敗:", e)
  }
}

// Get collection name for a character
function getCollectionName(characterName) {
  if (!settings.usePerCharacterCollections) {
    return settings.collectionName
  }

  // Sanitize character name for collection name (lowercase, replace spaces/special chars)
  const sanitized = characterName
    .toLowerCase()
    .replace(/[^a-z0-9_-]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "")

  return `${settings.collectionName}_${sanitized}`
}

// Get embedding dimensions for the selected model
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
  }

  const customDimensions = Number.parseInt(settings.customEmbeddingDimensions, 10)
  const isCustomValid = Number.isFinite(customDimensions) && customDimensions > 0

  if (settings.embeddingProvider === "local") {
    if (isCustomValid) {
      return customDimensions
    }
    return null
  }

  if (dimensions[settings.embeddingModel]) {
    return dimensions[settings.embeddingModel]
  }

  if (isCustomValid) {
    return customDimensions
  }

  return 1536
}

function updateLocalEmbeddingDimensions(vector) {
  if (settings.embeddingProvider !== "local") {
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
    console.log(`[Qdrant Memory] Auto-detected local embedding dimensions: ${vectorSize}`)
  }
}

function getEmbeddingProviderError() {
  const provider = settings.embeddingProvider || "openai"

  const validProviders = ["openai", "openrouter", "local"]
  if (!validProviders.includes(provider)) {
    return `Unsupported embedding provider: ${provider}`
  }

  if (provider === "openai") {
    if (!settings.openaiApiKey || !settings.openaiApiKey.trim()) {
      return "OpenAI API key not set"
    }
  }

  if (provider === "openrouter") {
    if (!settings.openRouterApiKey || !settings.openRouterApiKey.trim()) {
      return "OpenRouter API key not set"
    }
  }

  if (provider === "local") {
    if (!settings.localEmbeddingUrl || !settings.localEmbeddingUrl.trim()) {
      return "Local embedding URL not set"
    }

    if (settings.customEmbeddingDimensions != null && settings.customEmbeddingDimensions !== "") {
      const customDimensions = Number.parseInt(settings.customEmbeddingDimensions, 10)
      if (!Number.isFinite(customDimensions) || customDimensions <= 0) {
        return "Embedding dimensions must be a positive number"
      }
    }
  }

  if (!provider) {
    return "Embedding provider not configured"
  }

  return null
}

// ============================================================================
// HTTP HEADERS AND CSRF TOKEN HANDLING
// ============================================================================

// Helper to safely call potential CSRF token providers
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

// Helper to read a cookie value by name
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

// Get headers for SillyTavern API requests (with CSRF token if available)
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

  // Try multiple possible locations for the header builder
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
      // Continue to next method
    }
  }
  
  // None of the built-in methods worked
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

// Get headers for Qdrant requests (with optional API key)
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
// QDRANT COLLECTION MANAGEMENT
// ============================================================================

// Check if collection exists
async function collectionExists(collectionName) {
  try {
    const response = await fetch(`${settings.qdrantUrl}/collections/${collectionName}`, {
      headers: getQdrantHeaders(),
    })

    if (response.status === 404) {
      return { exists: false, vectorSize: null }
    }

    if (!response.ok) {
      console.error(
        `[Qdrant Memory] Failed to fetch collection info: ${collectionName} (${response.status} ${response.statusText})`,
      )
      return { exists: false, vectorSize: null }
    }

    const data = await response.json().catch(() => null)
    const vectorSize =
      data?.result?.config?.params?.vectors?.size ??
      data?.result?.config?.params?.vectors?.default?.size ??
      data?.result?.vectors?.size ??
      null

    return { exists: true, vectorSize }
  } catch (error) {
    console.error("[Qdrant Memory] Error checking collection:", error)
    return { exists: false, vectorSize: null }
  }
}

// Create collection for a character
async function createCollection(collectionName, vectorSize) {
  try {
    const dimensions = Number.isFinite(vectorSize) && vectorSize > 0 ? vectorSize : getEmbeddingDimensions()

    if (!Number.isFinite(dimensions) || dimensions <= 0) {
      console.error(`[Qdrant Memory] Cannot create collection ${collectionName} - invalid embedding dimensions`)
      return false
    }

    const response = await fetch(`${settings.qdrantUrl}/collections/${collectionName}`, {
      method: "PUT",
      headers: getQdrantHeaders(),
      body: JSON.stringify({
        vectors: {
          size: dimensions,
          distance: "Cosine",
        },
      }),
    })

    if (response.ok) {
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Created collection: ${collectionName}`)
      }
      // 建立 collection 後立刻嘗試建立全文索引（multilingual 分詞）
      if (settings.enableTextIndex) {
        await ensureTextIndex(collectionName)
      }
      return true
    } else {
      console.error(`[Qdrant Memory] Failed to create collection: ${collectionName}`)
      return false
    }
  } catch (error) {
    console.error("[Qdrant Memory] Error creating collection:", error)
    return false
  }
}

// 為現有 collection 建立 / 確認 multilingual 文字 payload index（Qdrant 1.15+）
// 此函式對「已存在的索引」是冪等的：Qdrant 會回傳 200，無副作用
async function ensureTextIndex(collectionName) {
  if (!settings.enableTextIndex) return false

  try {
    const tokenizer = settings.textIndexTokenizer || "multilingual"
    const response = await fetch(`${settings.qdrantUrl}/collections/${collectionName}/index`, {
      method: "PUT",
      headers: getQdrantHeaders(),
      body: JSON.stringify({
        field_name: "text",
        field_schema: {
          type: "text",
          tokenizer: tokenizer,
          min_token_len: 1,
          max_token_len: 30,
          lowercase: true,
        },
      }),
    })

    if (response.ok) {
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Text payload index ready on ${collectionName} (tokenizer=${tokenizer})`)
      }
      return true
    }

    // Qdrant 對於已存在的相同 schema 通常會回傳成功；如果是 schema 衝突會 4xx
    const errText = await response.text().catch(() => "")
    console.warn(
      `[Qdrant Memory] Text index 建立回應 ${response.status} 於 ${collectionName}: ${errText.substring(0, 200)}`,
    )
    return false
  } catch (error) {
    console.warn("[Qdrant Memory] ensureTextIndex 發生錯誤:", error)
    return false
  }
}

// Ensure collection exists (create if needed)
async function ensureCollection(characterName, vectorSize) {
  const collectionName = getCollectionName(characterName)
  const { exists, vectorSize: existingSize } = await collectionExists(collectionName)

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
    console.log(`[Qdrant Memory] Collection doesn't exist, creating: ${collectionName}`)
  }

  return await createCollection(collectionName, vectorSize)
}

// ============================================================================
// EMBEDDING GENERATION
// ============================================================================

// Generate embedding using the configured provider
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
    const body = {
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
// MEMORY SEARCH AND RETRIEVAL
// ============================================================================

// NEW: Check if chunk already exists (deduplication)
async function chunkExistsInCollection(collectionName, embedding, text, dedupeThreshold) {
  try {
    const searchPayload = {
      vector: embedding,
      limit: 1,
      score_threshold: dedupeThreshold,
      with_payload: true,
    }

    const response = await fetch(`${settings.qdrantUrl}/collections/${collectionName}/points/search`, {
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

// ========== 混合搜尋輔助函式 ==========

// dense 向量搜尋（取得有 score 的候選）
async function denseSearch(collectionName, vector, limit, filterConditions) {
  const payload = {
    vector,
    limit,
    score_threshold: 0, // 候選池階段不過濾，最後一起判
    with_payload: true,
  }
  if (filterConditions && filterConditions.length > 0) {
    payload.filter = { must: filterConditions }
  }
  try {
    const response = await fetch(`${settings.qdrantUrl}/collections/${collectionName}/points/search`, {
      method: "POST",
      headers: getQdrantHeaders(),
      body: JSON.stringify(payload),
    })
    if (!response.ok) {
      if (settings.debugMode) console.warn("[Qdrant Memory] dense 搜尋失敗:", response.statusText)
      return []
    }
    const data = await response.json()
    return data.result || []
  } catch (e) {
    console.error("[Qdrant Memory] dense 搜尋錯誤:", e)
    return []
  }
}

// 文字過濾搜尋（透過 scroll + text match filter；Qdrant 1.15+ multilingual 分詞）
// 結果無相關度分數，依時間倒序作為排名
async function textSearch(collectionName, queryText, limit, filterConditions) {
  if (!queryText || !queryText.trim()) return []

  const mustConditions = [
    { key: "text", match: { text: queryText } },
  ]
  if (filterConditions && filterConditions.length > 0) {
    mustConditions.push(...filterConditions)
  }

  const payload = {
    filter: { must: mustConditions },
    limit,
    with_payload: true,
    with_vector: false,
  }

  try {
    const response = await fetch(`${settings.qdrantUrl}/collections/${collectionName}/points/scroll`, {
      method: "POST",
      headers: getQdrantHeaders(),
      body: JSON.stringify(payload),
    })
    if (!response.ok) {
      if (settings.debugMode) {
        const txt = await response.text().catch(() => "")
        console.warn("[Qdrant Memory] 文字搜尋失敗（可能是 text index 還沒建立）:", txt.substring(0, 200))
      }
      return []
    }
    const data = await response.json()
    const points = data.result?.points || []

    // 依 timestamp 倒序排序作為 ranking 依據
    points.sort((a, b) => {
      const ta = a.payload?.timestamp || 0
      const tb = b.payload?.timestamp || 0
      return tb - ta
    })

    return points
  } catch (e) {
    console.error("[Qdrant Memory] 文字搜尋錯誤:", e)
    return []
  }
}

// Reciprocal Rank Fusion：合併多組排名後的結果
function rrfFusion(rankings, k) {
  const kVal = Number.isFinite(k) && k > 0 ? k : 60
  const scoreMap = new Map() // id -> { score, item }

  for (const ranking of rankings) {
    if (!Array.isArray(ranking)) continue
    ranking.forEach((item, rank) => {
      if (!item || !item.id) return
      const rrfScore = 1 / (kVal + rank + 1)
      const existing = scoreMap.get(item.id)
      if (existing) {
        existing.score += rrfScore
      } else {
        scoreMap.set(item.id, { score: rrfScore, item })
      }
    })
  }

  return [...scoreMap.values()]
    .sort((a, b) => b.score - a.score)
    .map((entry) => ({ ...entry.item, _rrfScore: entry.score }))
}

// Cohere Rerank（v2 API）
async function cohereRerank(query, documents, topN) {
  if (!settings.enableRerank) return null
  if (!settings.cohereApiKey || !settings.cohereApiKey.trim()) {
    if (settings.debugMode) console.warn("[Qdrant Memory] 未設定 Cohere API Key，跳過 Rerank")
    return null
  }
  if (!Array.isArray(documents) || documents.length === 0) return null

  try {
    const model = settings.cohereRerankModel || "rerank-multilingual-v3.0"
    const docTexts = documents.map((d) => d.text || "")
    const response = await fetch("https://api.cohere.com/v2/rerank", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${settings.cohereApiKey.trim()}`,
      },
      body: JSON.stringify({
        model,
        query,
        documents: docTexts,
        top_n: Math.min(topN, docTexts.length),
      }),
    })

    if (!response.ok) {
      const errText = await response.text().catch(() => "")
      console.error("[Qdrant Memory] Cohere Rerank 失敗:", response.status, errText.substring(0, 200))
      return null
    }

    const data = await response.json()
    const results = data.results || []
    if (settings.debugMode) {
      console.log(`[Qdrant Memory] Cohere Rerank 完成，回傳 ${results.length} 個結果`)
    }
    return results // [{ index, relevance_score }, ...]
  } catch (e) {
    console.error("[Qdrant Memory] Cohere Rerank 錯誤:", e)
    return null
  }
}

// Search Qdrant for relevant memories
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

    // 確保 multilingual 文字索引存在（混合搜尋必要）
    if (settings.enableHybridSearch && settings.enableTextIndex) {
      await ensureTextIndex(collectionName)
    }

    // FIXED: Improved retain logic - get ALL message IDs that should be excluded
    const context = getContext()
    const chat = context.chat || []
    const excludedMessageIds = new Set()

    if (settings.retainRecentMessages > 0 && chat.length > settings.retainRecentMessages) {
      const recentMessages = chat.slice(-settings.retainRecentMessages)

      recentMessages.forEach((msg) => {
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

    // 計算候選池大小
    const candidatePoolSize = Math.max(
      settings.memoryLimit * Math.max(2, settings.hybridCandidateMultiplier || 4),
      settings.enableRerank ? settings.rerankCandidates || 20 : settings.memoryLimit * 2,
    )

    // === 並行執行：dense 搜尋 + 文字搜尋 ===
    let denseResults = []
    let textResults = []

    if (settings.enableHybridSearch) {
      ;[denseResults, textResults] = await Promise.all([
        denseSearch(collectionName, embedding, candidatePoolSize, filterConditions),
        textSearch(collectionName, query, candidatePoolSize, filterConditions),
      ])

      if (settings.debugMode) {
        console.log(
          `[Qdrant Memory] 混合搜尋：dense ${denseResults.length} 筆，文字 ${textResults.length} 筆`,
        )
      }
    } else {
      denseResults = await denseSearch(collectionName, embedding, candidatePoolSize, filterConditions)
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] 純向量搜尋：${denseResults.length} 筆`)
      }
    }

    // === 融合 ===
    let fusedResults
    if (settings.enableHybridSearch && textResults.length > 0) {
      fusedResults = rrfFusion([denseResults, textResults], settings.rrfK)
    } else {
      fusedResults = denseResults
    }

    // 過濾近期訊息
    if (excludedMessageIds.size > 0) {
      const beforeFilterCount = fusedResults.length
      fusedResults = fusedResults.filter((memory) => {
        const messageIds = memory.payload?.messageIds || ""
        const chunkMessageIds = messageIds.split(",")
        const hasExcluded = chunkMessageIds.some((id) => excludedMessageIds.has(id.trim()))
        return !hasExcluded
      })
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] 過濾掉 ${beforeFilterCount - fusedResults.length} 個含近期訊息的 chunk`)
      }
    }

    // 文字去重（用前 200 字元做為比較基準）
    const dedupedResults = []
    const seenTexts = new Set()
    for (const result of fusedResults) {
      const text = result.payload?.text || ""
      const normalizedText = text
        .replace(/\[[\d-]+\]/g, "")
        .replace(/\s+/g, " ")
        .trim()
        .substring(0, 200)

      if (!seenTexts.has(normalizedText)) {
        seenTexts.add(normalizedText)
        dedupedResults.push(result)
      }
    }

    // === Rerank（可選）===
    let finalResults = dedupedResults
    if (settings.enableRerank && dedupedResults.length > 0) {
      const rerankPool = dedupedResults.slice(0, settings.rerankCandidates || 20)
      const docs = rerankPool.map((r) => ({ text: r.payload?.text || "" }))
      const rerankResp = await cohereRerank(query, docs, settings.memoryLimit)

      if (rerankResp && rerankResp.length > 0) {
        // 用 rerank 的 relevance_score 取代原 score，並按 rerank 的順序回傳
        finalResults = rerankResp
          .map((item) => {
            const original = rerankPool[item.index]
            if (!original) return null
            return {
              ...original,
              score: item.relevance_score, // formatMemories 會用這個顯示百分比
              _rerankScore: item.relevance_score,
            }
          })
          .filter(Boolean)

        if (settings.debugMode) {
          console.log(`[Qdrant Memory] Rerank 後保留 ${finalResults.length} 筆`)
        }
      } else {
        // Rerank 失敗 → 退回原排序，但仍套用相關度門檻
        if (settings.debugMode) console.log("[Qdrant Memory] Rerank 失敗，退回融合結果")
      }
    }

    // 最終門檻過濾（僅對有 score 的結果）+ 取前 N 筆
    const thresholded = finalResults.filter((r) => {
      // RRF 融合分數不適用 0-1 的相似度門檻；只有當有原始 score 時才套用
      if (typeof r.score === "number") {
        return r.score >= settings.scoreThreshold
      }
      return true
    })

    const trimmed = thresholded.slice(0, settings.memoryLimit)

    if (settings.debugMode) {
      console.log(`[Qdrant Memory] 最終回傳 ${trimmed.length} 筆 memories（門檻 ${settings.scoreThreshold}）`)
    }

    return trimmed
  } catch (error) {
    console.error("[Qdrant Memory] Error searching memories:", error)
    return []
  }
}

// Format memories for display
function formatMemories(memories) {
  if (!memories || memories.length === 0) return ""

  let formatted = "\n[過往對話記憶]\n\n"

  // Get persona name for display
  const personaName = getPersonaName()

  memories.forEach((memory) => {
    const payload = memory.payload

    let speakerLabel
    if (payload.isChunk) {
      // For conversation chunks, show all speakers
      speakerLabel = `對話片段（${payload.speakers}）`
    } else {
      // For individual messages (legacy format), use persona name
      speakerLabel = payload.speaker === "user"
        ? `${personaName} 說`
        : "角色說"
    }

    let text = payload.text.replace(/\n/g, " ") // flatten newlines

    const score = (memory.score * 100).toFixed(0)

    formatted += `• ${speakerLabel}：「${text}」（相關度：${score}%）\n\n`
  })

  return formatted
}

// ============================================================================
// MESSAGE CHUNKING AND BUFFERING
// ============================================================================

function getChatParticipants() {
  const context = getContext()
  const characterName = context.name2

  // Check if this is a group chat
  const characters = context.characters || []
  const chat = context.chat || []

  // For group chats, get all unique character names from recent messages
  if (characters.length > 1) {
    const participants = new Set()

    // Add the main character
    if (characterName) {
      participants.add(characterName)
    }

    // Look through recent messages to find all participants
    chat.slice(-50).forEach((msg) => {
      if (!msg.is_user && msg.name && msg.name !== "System") {
        participants.add(msg.name)
      }
    })

    return Array.from(participants)
  }

  // Single character chat
  return characterName ? [characterName] : []
}

function createChunkFromBuffer() {
  if (messageBuffer.length === 0) return null

  let chunkText = ""
  const speakers = new Set()
  const messageIds = []
  let totalLength = 0
  const currentTimestamp = Date.now()
  
  // NEW: Get the persona name once for this chunk
  const personaName = getPersonaName()

  // Build chunk text with speaker labels
  messageBuffer.forEach((msg) => {
    const speaker = msg.isUser ? personaName : msg.characterName  // ← CHANGED: Use personaName
    speakers.add(speaker)
    messageIds.push(msg.messageId)

    const line = `${speaker}: ${msg.text}\n`
    chunkText += line
    totalLength += line.length
  })

  // Format date prefix
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
    // Generate embedding for the chunk text
    const embedding = await generateEmbedding(chunk.text)
    if (!embedding) {
      console.error("[Qdrant Memory] Cannot save chunk - embedding generation failed")
      return false
    }

    // NEW: Check for duplicates before saving
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
        toastr.info("已有相似對話片段，跳過儲存", "Qdrant Memory", { timeOut: 1500 })
      }
      return false
    }

    const pointId = generateUUID()

    // Prepare payload
    const payload = {
      text: chunk.text,
      speakers: chunk.speakers.join(", "),
      messageCount: chunk.messageCount,
      timestamp: chunk.timestamp,
      messageIds: chunk.messageIds.join(","),
      isChunk: true,
    }

    // Save to all participant collections
    const savePromises = participants.map(async (characterName) => {
      const collectionName = getCollectionName(characterName)

      // Ensure collection exists
      const collectionReady = await ensureCollection(characterName, embedding.length)
      if (!collectionReady) {
        console.error(`[Qdrant Memory] Cannot save chunk - collection creation failed for ${characterName}`)
        return false
      }

      // Add character name to payload only if using shared collection
      const characterPayload = settings.usePerCharacterCollections 
        ? payload 
        : { ...payload, character: characterName }

      // Save to Qdrant
      const response = await fetch(`${settings.qdrantUrl}/collections/${collectionName}/points`, {
        method: "PUT",
        headers: getQdrantHeaders(),
        body: JSON.stringify({
          points: [
            {
              id: pointId,
              vector: embedding,
              payload: characterPayload,
            },
          ],
        }),
      })

      if (!response.ok) {
        console.error(
          `[Qdrant Memory] Failed to save chunk to ${characterName}: ${response.status} ${response.statusText}`,
        )
        return false
      }

      if (settings.debugMode) {
        console.log(
          `[Qdrant Memory] Saved chunk to ${characterName}'s collection (${chunk.messageCount} messages, ${chunk.text.length} chars)`,
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

  // Get all participants (for group chats)
  const participants = getChatParticipants()

  if (participants.length === 0) {
    console.error("[Qdrant Memory] No participants found for chunk")
    messageBuffer = []
    return
  }

  // Save chunk to all participant collections
  await saveChunkToQdrant(chunk, participants)

  // Clear buffer after saving
  messageBuffer = []
}

function bufferMessage(text, characterName, isUser, messageId) {
  if (!settings.enabled) return
  if (!settings.autoSaveMemories) return
  if (getEmbeddingProviderError()) return
  if (text.length < settings.minMessageLength) return

  // Check if we should save this type of message
  if (isUser && !settings.saveUserMessages) return
  if (!isUser && !settings.saveCharacterMessages) return

  // Add to buffer
  messageBuffer.push({ text, characterName, isUser, messageId })
  lastMessageTime = Date.now()

  // Calculate current buffer size
  let bufferSize = 0
  messageBuffer.forEach((msg) => {
    bufferSize += msg.text.length + msg.characterName.length + 4 // +4 for ": " and "\n"
  })

  if (settings.debugMode) {
    console.log(`[Qdrant Memory] Buffer: ${messageBuffer.length} messages, ${bufferSize} chars`)
  }

  // Clear existing timer
  if (chunkTimer) {
    clearTimeout(chunkTimer)
  }

  // If buffer exceeds max size, process it now
  if (bufferSize >= settings.chunkMaxSize) {
    if (settings.debugMode) {
      console.log(`[Qdrant Memory] Buffer reached max size (${bufferSize}), processing chunk`)
    }
    processMessageBuffer()
  }
  // If buffer is at least min size, set a short timer
  else if (bufferSize >= settings.chunkMinSize) {
    chunkTimer = setTimeout(() => {
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Buffer reached min size and timeout, processing chunk`)
      }
      processMessageBuffer()
    }, 5000) // 5 seconds after reaching min size
  }
  // Otherwise, set a longer timer
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
// CHAT INDEXING FUNCTIONS
// ============================================================================

async function getCharacterChats(characterName) {
  try {
    const context = getContext()

    if (settings.debugMode) {
      console.log("[Qdrant Memory] Getting chats for character:", characterName)
    }

    // Try to get the character's avatar URL
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

    // Handle different response formats - extract just the filenames
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

    // Ensure chatFile is a string
    if (typeof chatFile !== 'string') {
      console.error("[Qdrant Memory] chatFile is not a string:", chatFile)
      if (chatFile && chatFile.file_name) {
        chatFile = chatFile.file_name
      } else {
        return null
      }
    }

    // Remove .jsonl extension as the API adds it back
    const fileNameWithoutExt = chatFile.replace(/\.jsonl$/, '')

    const context = getContext()

    

    // Try to get the character's avatar URL
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
    
    // Handle different response formats
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
    // Search for any of the message IDs in the chunk
    const response = await fetch(`${settings.qdrantUrl}/collections/${collectionName}/points/scroll`, {
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
    // Skip system messages
    if (msg.is_system) continue

    const text = msg.mes?.trim()
    if (!text || text.length < settings.minMessageLength) continue

    // Check if we should save this type of message
    const isUser = msg.is_user || false
    if (isUser && !settings.saveUserMessages) continue
    if (!isUser && !settings.saveCharacterMessages) continue

    // Normalize send_date before using it
    const normalizedDate = normalizeTimestamp(msg.send_date || Date.now())
    
    if (settings.debugMode) {
      console.log("[Qdrant Memory] Message date - raw:", msg.send_date, "normalized:", normalizedDate, "formatted:", formatDateForChunk(normalizedDate))
    }

    // Create message object
    const messageObj = {
      text: text,
      characterName: characterName,
      isUser: isUser,
      messageId: `${characterName}_${normalizedDate}_${messages.indexOf(msg)}`,
      timestamp: normalizedDate,
    }

    const messageSize = text.length + characterName.length + 4

    // If adding this message would exceed max size, save current chunk
    if (currentSize + messageSize > settings.chunkMaxSize && currentChunk.length > 0) {
      chunks.push(createChunkFromMessages(currentChunk))
      currentChunk = []
      currentSize = 0
    }

    currentChunk.push(messageObj)
    currentSize += messageSize

    // If we've reached min size and have a good number of messages, consider chunking
    if (currentSize >= settings.chunkMinSize && currentChunk.length >= 3) {
      chunks.push(createChunkFromMessages(currentChunk))
      currentChunk = []
      currentSize = 0
    }
  }

  // Save any remaining messages
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
  
  // NEW: Get the persona name once for all messages
  const personaName = getPersonaName()

  messages.forEach((msg) => {
    const speaker = msg.isUser ? personaName : msg.characterName  // ← CHANGED: Use personaName
    speakers.add(speaker)
    messageIds.push(msg.messageId)

    const line = `${speaker}: ${msg.text}\n`
    chunkText += line

    if (msg.timestamp < oldestTimestamp) {
      oldestTimestamp = msg.timestamp
    }
  })

  // Format date prefix for the chunk
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

  // Create progress modal
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
        <h3 style="margin-top: 0;">索引對話記錄 - ${characterName}</h3>
        <p id="qdrant_index_status">正在掃描對話檔案…</p>
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
    $("#qdrant_index_cancel").text("正在取消…").prop("disabled", true)
  })

  try {
    // Get all chat files for this character
    const chatFiles = await getCharacterChats(characterName)

    if (chatFiles.length === 0) {
      $("#qdrant_index_status").text("找不到對話檔案")
      setCancelButtonToClose()
      setTimeout(() => {
        closeModal()
      }, 2000)
      return
    }

    $("#qdrant_index_status").text(`找到 ${chatFiles.length} 個對話檔案`)

    const collectionName = getCollectionName(characterName)

    let totalChunks = 0
    let savedChunks = 0
    let skippedChunks = 0

    // Process each chat file
    for (let i = 0; i < chatFiles.length; i++) {
      if (cancelled) break

      const chatFile = chatFiles[i]
      const progress = ((i / chatFiles.length) * 100).toFixed(0)

      $("#qdrant_index_progress").css("width", `${progress}%`)
      $("#qdrant_index_status").text(`處理對話 ${i + 1}/${chatFiles.length}`)
      $("#qdrant_index_details").text(`檔案：${chatFile}`)

      // Load chat file
      const chatData = await loadChatFile(characterName, chatFile)
      if (!chatData || !Array.isArray(chatData)) continue

      // Create chunks from messages
      const chunks = createChunksFromChat(chatData, characterName)
      totalChunks += chunks.length

      // Save each chunk
      for (const chunk of chunks) {
        if (cancelled) break

        // Check if chunk already exists
        const exists = await chunkExists(collectionName, chunk.messageIds)
        if (exists) {
          skippedChunks++
          continue
        }

        // Get participants (for group chats)
        const participants = [characterName]

        // Save chunk
        const success = await saveChunkToQdrant(chunk, participants)
        if (success) {
          savedChunks++
        }

        $("#qdrant_index_details").text(`已存：${savedChunks} | 已跳過：${skippedChunks} | 共：${totalChunks}`)
      }
    }

    // Complete
    $("#qdrant_index_progress").css("width", "100%")

    if (cancelled) {
      $("#qdrant_index_status").text("已取消索引")
      toastr.info(`取消前已索引 ${savedChunks} 個片段`, "Qdrant Memory")
    } else {
      $("#qdrant_index_status").text("索引完成！")
      toastr.success(`新索引 ${savedChunks} 個片段，跳過 ${skippedChunks} 個既有片段`, "Qdrant Memory")
    }

    setCancelButtonToClose()
  } catch (error) {
    console.error("[Qdrant Memory] Error indexing chats:", error)
    $("#qdrant_index_status").text("索引時發生錯誤")
    $("#qdrant_index_details").text(error.message)
    toastr.error("索引對話失敗", "Qdrant Memory")
    setCancelButtonToClose()
  }
}

// ============================================================================
// GENERATION INTERCEPTOR
// ============================================================================

// FIXED: Prevent duplicate memory injection
globalThis.qdrantMemoryInterceptor = async (chat, contextSize, abort, type) => {
  if (!settings.enabled) {
    if (settings.debugMode) {
      console.log("[Qdrant Memory] Extension disabled, skipping")
    }
    return
  }

  // NEW: Use chat hash instead of WeakMap
  if (settings.preventDuplicateInjection) {
    const chatHash = getChatHash(chat)
    
    if (memoryInjectionTracker.has(chatHash)) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Memories already injected for this chat state, skipping")
      }
      return
    }
    
    // Mark this chat state as having memories injected
    memoryInjectionTracker.add(chatHash)
    
    // Clean up old hashes to prevent memory leaks (keep last 50)
    if (memoryInjectionTracker.size > 50) {
      const oldestHash = memoryInjectionTracker.values().next().value
      memoryInjectionTracker.delete(oldestHash)
    }
  }

  try {
    const context = getContext()
    const characterName = context.name2

    // Skip if no character is selected
    if (!characterName) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] No character selected, skipping")
      }
      return
    }

    // Find the last user message to use as the query
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

    // Search for relevant memories
    const memories = await searchMemories(query, characterName)

    if (memories.length > 0) {
      const memoryText = formatMemories(memories)

      if (settings.debugMode) {
        console.log("[Qdrant Memory] Retrieved memories:", memoryText)
      }

      // Create memory entry
      const memoryEntry = {
        name: "System",
        is_user: false,
        is_system: true,
        mes: memoryText,
        send_date: Date.now(),
      }

      // Insert memories at the specified position from the end
      const insertIndex = Math.max(0, chat.length - settings.memoryPosition)
      chat.splice(insertIndex, 0, memoryEntry)

      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Injected ${memories.length} memories at position ${insertIndex}`)
      }

      const toastr = window.toastr
      if (settings.showMemoryNotifications) {
        toastr.info(`已注入 ${memories.length} 條相關記憶`, "Qdrant Memory", { timeOut: 2000 })
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
// AUTOMATIC MEMORY CREATION
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
  
  // Safety check: make sure this is actually a character message
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

    // Buffer the complete message
    bufferMessage(text, characterName, false, messageId)

    // Optionally flush the buffer if we have enough messages
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

    // Log every 4 seconds (16 polls at 250ms) in debug mode
    if (settings.debugMode && pollCount % 16 === 0) {
      console.log(`[Qdrant Memory] Stream poll check #${pollCount}: length=${currentText.length}`)
    }

    // Detect if the text has changed
    if (currentText !== pendingAssistantFinalize.lastText) {
      if (settings.debugMode && pollCount % 4 === 0) {
        console.log(`[Qdrant Memory] Text changed: ${pendingAssistantFinalize.lastText.length} → ${currentText.length}`)
      }
      pendingAssistantFinalize.lastText = currentText
      pendingAssistantFinalize.lastChangeAt = now
    }

    const stableDuration = now - pendingAssistantFinalize.lastChangeAt
    const totalDuration = now - pendingAssistantFinalize.startedAt

    // Check if message switched to user (conversation continued)
    if (currentLastMessage.is_user) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Detected new user message, finalizing previous assistant message")
      }
      finalizeAssistant(pendingAssistantFinalize.lastText, "new user message detected")
      return
    }

    // Check if text has been stable for the required duration
    if (stableDuration >= stableMs) {
      finalizeAssistant(pendingAssistantFinalize.lastText, `stable for ${stableDuration}ms`)
      return
    }

    // Safety: max wait time exceeded
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

    // Get the last message
    const lastMessage = chat[chat.length - 1]

    // Normalize send_date for messageId
    const normalizedDate = normalizeTimestamp(lastMessage.send_date || Date.now())
    
    // Create a unique ID for this message
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
      // User messages are saved immediately
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Buffering user message immediately")
      }
      bufferMessage(lastMessage.mes, characterName, true, messageId)
    } else {
      // Character messages need to wait for streaming to complete
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
// MEMORY VIEWER FUNCTIONS
// ============================================================================

async function getCollectionInfo(collectionName) {
  try {
    const response = await fetch(`${settings.qdrantUrl}/collections/${collectionName}`, {
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
    const response = await fetch(`${settings.qdrantUrl}/collections/${collectionName}`, {
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
                <h3 style="margin-top: 0;">記憶檢視 - ${characterName}</h3>
                <p><strong>Collection：</strong>${collectionName}</p>
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
      `確定要刪除 ${characterName} 的所有記憶嗎？此動作無法復原！`,
    )
    if (confirmed) {
      $(this).prop("disabled", true).text("正在刪除…")
      const success = await deleteCollection(collectionName)
      if (success) {
        const toastr = window.toastr
        toastr.success(`已刪除 ${characterName} 的全部記憶`, "Qdrant Memory")
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
// UTILITY FUNCTIONS
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
  
  // Try multiple possible locations for persona name in order of preference
  const personaName = 
    context.name1 ||                              // Standard SillyTavern location
    context.persona?.name ||                      // Alternative location
    window.name1 ||                               // Direct window access
    window.SillyTavern?.getContext?.()?.name1 ||  // Through ST API
    "User"                                        // Fallback to generic "User"
  
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
// SETTINGS UI
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
                        自動建立並注入帶時間脈絡的對話記憶
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
                <label><strong>Qdrant URL：</strong></label>
                <input type="text" id="qdrant_url" class="text_pole" value="${settings.qdrantUrl}"
                       style="width: 100%; margin-top: 5px;"
                       placeholder="http://localhost:6333" />
                <small style="color: #666;">Qdrant 服務的網址</small>
            </div>

             <div style="margin: 10px 0;">
                <label><strong>Qdrant API Key：</strong></label>
                <input type="password" id="qdrant_api_key" class="text_pole" value="${settings.qdrantApiKey || ""}"
                       style="width: 100%; margin-top: 5px;"
                       placeholder="選填，若未設密碼可留空" />
                <small style="color: #666;">Qdrant 認證金鑰（選填）</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>Collection 基底名稱：</strong></label>
                <input type="text" id="qdrant_collection" class="text_pole" value="${settings.collectionName}"
                       style="width: 100%; margin-top: 5px;"
                       placeholder="sillytavern_memories" />
                <small style="color: #666;">每個角色會自動附加角色名作為後綴</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>Embedding 服務商：</strong></label>
                <select id="qdrant_embedding_provider" class="text_pole" style="width: 100%; margin-top: 5px;">
                    <option value="openai" ${settings.embeddingProvider === "openai" ? "selected" : ""}>OpenAI</option>
                    <option value="openrouter" ${settings.embeddingProvider === "openrouter" ? "selected" : ""}>OpenRouter</option>
                    <option value="local" ${settings.embeddingProvider === "local" ? "selected" : ""}>本機 / 自訂端點</option>
                </select>
                <small style="color: #666;">選擇用於產生向量的 API</small>
            </div>

            <div id="qdrant_openai_key_group" style="margin: 10px 0;">
                <label><strong>OpenAI API Key：</strong></label>
                <input type="password" id="qdrant_openai_key" class="text_pole" value="${settings.openaiApiKey}"
                       placeholder="sk-..." style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">使用 OpenAI 時必填</small>
            </div>

            <div id="qdrant_openrouter_key_group" style="margin: 10px 0; display: none;">
                <label><strong>OpenRouter API Key：</strong></label>
                <input type="password" id="qdrant_openrouter_key" class="text_pole" value="${settings.openRouterApiKey}"
                       placeholder="or-..." style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">使用 OpenRouter 時必填</small>
            </div>

            <div id="qdrant_local_url_group" style="margin: 10px 0; display: none;">
                <label><strong>Embedding 端點 URL：</strong></label>
                <input type="text" id="qdrant_local_url" class="text_pole" value="${settings.localEmbeddingUrl}"
                       placeholder="http://localhost:11434/api/embeddings"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">需相容 OpenAI embeddings 格式（例如 Google AI Studio 的 OpenAI 相容端點）</small>
            </div>

            <div id="qdrant_local_api_key_group" style="margin: 10px 0; display: none;">
                <label><strong>Embedding API Key（選填）：</strong></label>
                <input type="password" id="qdrant_local_api_key" class="text_pole" value="${settings.localEmbeddingApiKey}"
                       placeholder="本機端點的 Bearer token"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">若你的端點需要驗證請填入</small>
            </div>

            <div id="qdrant_local_dimensions_group" style="margin: 10px 0; display: none;">
                <label><strong>Embedding 維度：</strong></label>
                <input type="number" id="qdrant_local_dimensions" class="text_pole"
                       value="${settings.customEmbeddingDimensions ?? ""}"
                       min="1" step="1" style="width: 100%; margin-top: 5px;" placeholder="留空則首次呼叫後自動偵測" />
                <small style="color: #666;">自訂模型回傳的向量大小（留空可自動偵測）</small>
            </div>

<div id="qdrant_local_model_group" style="margin: 10px 0; display: none;">
                <label><strong>Embedding 模型名稱：</strong></label>
                <input type="text" id="qdrant_local_model" class="text_pole" value="${settings.embeddingModel}"
                       placeholder="text-embedding-004"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">送出 API 請求時用的 model 名稱（如 text-embedding-004、gemini-embedding-001）</small>
            </div>

            <div id="qdrant_embedding_model_group" style="margin: 10px 0;">
                <label><strong>Embedding 模型：</strong></label>
                <select id="qdrant_embedding_model" class="text_pole" style="width: 100%; margin-top: 5px;">
                    <option value="text-embedding-3-large" ${settings.embeddingModel === "text-embedding-3-large" ? "selected" : ""}>text-embedding-3-large（品質最佳）</option>
                    <option value="text-embedding-3-small" ${settings.embeddingModel === "text-embedding-3-small" ? "selected" : ""}>text-embedding-3-small（較快）</option>
                    <option value="text-embedding-ada-002" ${settings.embeddingModel === "text-embedding-ada-002" ? "selected" : ""}>text-embedding-ada-002（舊版）</option>
                </select>
            </div>

            <hr style="margin: 15px 0;" />

            <h4>記憶擷取設定</h4>

            <div style="margin: 10px 0;">
                <label><strong>記憶數量：</strong> <span id="memory_limit_display">${settings.memoryLimit}</span></label>
                <input type="range" id="qdrant_memory_limit" min="1" max="50" value="${settings.memoryLimit}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">每次生成最多注入幾條記憶</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>相關度門檻：</strong> <span id="score_threshold_display">${settings.scoreThreshold}</span></label>
                <input type="range" id="qdrant_score_threshold" min="0" max="1" step="0.05" value="${settings.scoreThreshold}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">最低相似度分數（0.0 ~ 1.0）。啟用 Rerank 後此門檻套用在 rerank 分數上</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>記憶插入位置：</strong> <span id="memory_position_display">${settings.memoryPosition}</span></label>
                <input type="range" id="qdrant_memory_position" min="1" max="30" value="${settings.memoryPosition}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">從對話末端往前數第幾則訊息插入記憶</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>保留近期訊息：</strong> <span id="retain_recent_display">${settings.retainRecentMessages}</span></label>
                <input type="range" id="qdrant_retain_recent" min="0" max="50" value="${settings.retainRecentMessages}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">將最近 N 則訊息排除在搜尋結果之外（0 = 不排除）</small>
            </div>

            <hr style="margin: 15px 0;" />

            <h4>混合搜尋（Qdrant 1.15+）</h4>

            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_text_index" ${settings.enableTextIndex ? "checked" : ""} />
                    <strong>建立全文索引</strong>
                </label>
                <small style="color: #666; display: block; margin-left: 30px;">建立 collection 時自動為 text 欄位建立 payload index（中日韓建議使用 multilingual 分詞）</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>分詞方式：</strong></label>
                <select id="qdrant_text_tokenizer" class="text_pole" style="width: 100%; margin-top: 5px;">
                    <option value="multilingual" ${settings.textIndexTokenizer === "multilingual" ? "selected" : ""}>multilingual（推薦，支援中日韓）</option>
                    <option value="word" ${settings.textIndexTokenizer === "word" ? "selected" : ""}>word（以單字切詞）</option>
                    <option value="whitespace" ${settings.textIndexTokenizer === "whitespace" ? "selected" : ""}>whitespace（以空白切詞）</option>
                    <option value="prefix" ${settings.textIndexTokenizer === "prefix" ? "selected" : ""}>prefix（前綴匹配）</option>
                </select>
                <small style="color: #666;">需要 Qdrant 1.15 以上才支援 multilingual</small>
            </div>

            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_hybrid_search" ${settings.enableHybridSearch ? "checked" : ""} />
                    <strong>啟用混合搜尋（dense + 全文）</strong>
                </label>
                <small style="color: #666; display: block; margin-left: 30px;">同時用向量相似度與關鍵字檢索，再以 RRF 融合</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>候選池倍數：</strong> <span id="hybrid_multiplier_display">${settings.hybridCandidateMultiplier}</span></label>
                <input type="range" id="qdrant_hybrid_multiplier" min="2" max="10" step="1" value="${settings.hybridCandidateMultiplier}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">每路（dense 與全文）取 memoryLimit × 此倍數筆候選</small>
            </div>

            <hr style="margin: 15px 0;" />

            <h4>Cohere Rerank（可選）</h4>

            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_rerank_enabled" ${settings.enableRerank ? "checked" : ""} />
                    <strong>啟用 Cohere Rerank</strong>
                </label>
                <small style="color: #666; display: block; margin-left: 30px;">在融合候選後再用 Cohere 重排序，大幅提升相關性</small>
            </div>

            <div id="qdrant_cohere_key_group" style="margin: 10px 0; ${settings.enableRerank ? "" : "display: none;"}">
                <label><strong>Cohere API Key：</strong></label>
                <input type="password" id="qdrant_cohere_key" class="text_pole" value="${settings.cohereApiKey || ""}"
                       placeholder="cohere api key" style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">前往 dashboard.cohere.com 取得（trial key 即可使用）</small>
            </div>

            <div id="qdrant_cohere_model_group" style="margin: 10px 0; ${settings.enableRerank ? "" : "display: none;"}">
                <label><strong>Rerank 模型：</strong></label>
                <select id="qdrant_cohere_model" class="text_pole" style="width: 100%; margin-top: 5px;">
                    <option value="rerank-multilingual-v3.0" ${settings.cohereRerankModel === "rerank-multilingual-v3.0" ? "selected" : ""}>rerank-multilingual-v3.0（推薦，支援中文）</option>
                    <option value="rerank-v3.5" ${settings.cohereRerankModel === "rerank-v3.5" ? "selected" : ""}>rerank-v3.5（最新）</option>
                    <option value="rerank-english-v3.0" ${settings.cohereRerankModel === "rerank-english-v3.0" ? "selected" : ""}>rerank-english-v3.0（英文專用）</option>
                </select>
            </div>

            <div id="qdrant_rerank_candidates_group" style="margin: 10px 0; ${settings.enableRerank ? "" : "display: none;"}">
                <label><strong>送入 Rerank 的候選數：</strong> <span id="rerank_candidates_display">${settings.rerankCandidates}</span></label>
                <input type="range" id="qdrant_rerank_candidates" min="5" max="100" step="5" value="${settings.rerankCandidates}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">越多越準但越慢、也越耗 API 額度</small>
            </div>

            <hr style="margin: 15px 0;" />

            <h4>自動建立記憶</h4>

            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_per_character" ${settings.usePerCharacterCollections ? "checked" : ""} />
                    <strong>每個角色獨立 Collection</strong>
                </label>
                <small style="color: #666; display: block; margin-left: 30px;">每個角色都用自己的 collection（建議啟用）</small>
            </div>

            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_auto_save" ${settings.autoSaveMemories ? "checked" : ""} />
                    <strong>自動儲存記憶</strong>
                </label>
                <small style="color: #666; display: block; margin-left: 30px;">對話進行中自動把訊息存入 Qdrant</small>
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
                <label><strong>最短訊息長度：</strong> <span id="min_message_length_display">${settings.minMessageLength}</span></label>
                <input type="range" id="qdrant_min_length" min="5" max="50" value="${settings.minMessageLength}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">少於此字數的訊息不儲存</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>去重門檻：</strong> <span id="dedupe_threshold_display">${settings.dedupeThreshold}</span></label>
                <input type="range" id="qdrant_dedupe_threshold" min="0.80" max="1.00" step="0.01" value="${settings.dedupeThreshold}"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">避免儲存重複片段（越高越嚴格）</small>
            </div>

            <hr style="margin: 15px 0;" />

            <h4>其他設定</h4>

            <div style="margin: 15px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_prevent_duplicate" ${settings.preventDuplicateInjection ? "checked" : ""} />
                    防止重複注入記憶
                </label>
                <small style="color: #666; display: block; margin-left: 30px;">同一個對話狀態不重複注入記憶</small>
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
                    Debug 模式（請查看主控台）
                </label>
            </div>

            <hr style="margin: 15px 0;" />

            <div style="margin: 15px 0; display: flex; gap: 10px; flex-wrap: wrap;">
                <button id="qdrant_test" class="menu_button">測試連線</button>
                <button id="qdrant_save" class="menu_button">儲存設定</button>
                <button id="qdrant_view_memories" class="menu_button">檢視記憶</button>
                <button id="qdrant_index_chats" class="menu_button" style="background-color: #28a745; color: white;">索引角色對話</button>
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
      settings.embeddingModel = fallback
      $modelSelect.val(settings.embeddingModel)
    }
  }

  function updateEmbeddingProviderUI() {
    const provider = settings.embeddingProvider || "openai"
    const $openAIGroup = $("#qdrant_openai_key_group")
    const $openRouterGroup = $("#qdrant_openrouter_key_group")
    const $localGroup = $("#qdrant_local_url_group")
    const $localApiKeyGroup = $("#qdrant_local_api_key_group")
    const $localDimensionsGroup = $("#qdrant_local_dimensions_group")
    const $localDimensionsInput = $("#qdrant_local_dimensions")
    const $modelGroup = $("#qdrant_embedding_model_group")

    $openAIGroup.toggle(provider === "openai")
    $openRouterGroup.toggle(provider === "openrouter")
    $localGroup.toggle(provider === "local")
    $localApiKeyGroup.toggle(provider === "local")
    $localDimensionsGroup.toggle(provider === "local")

    $("#qdrant_local_model_group").toggle(provider === "local")

    if (provider === "local") {
      $localDimensionsInput.val(settings.customEmbeddingDimensions ?? "")
    }

    const showModelSelect = provider !== "local"
    $modelGroup.toggle(showModelSelect)

    if (showModelSelect) {
      updateEmbeddingModelOptions(provider)
    }
  }

  // Event handlers
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

  // 混合搜尋 / 全文索引
  $("#qdrant_text_index").on("change", function () {
    settings.enableTextIndex = $(this).is(":checked")
  })

  $("#qdrant_text_tokenizer").on("change", function () {
    settings.textIndexTokenizer = $(this).val()
  })

  $("#qdrant_hybrid_search").on("change", function () {
    settings.enableHybridSearch = $(this).is(":checked")
  })

  $("#qdrant_hybrid_multiplier").on("input", function () {
    settings.hybridCandidateMultiplier = Number.parseInt($(this).val())
    $("#hybrid_multiplier_display").text(settings.hybridCandidateMultiplier)
  })

  // Cohere Rerank
  $("#qdrant_rerank_enabled").on("change", function () {
    settings.enableRerank = $(this).is(":checked")
    $("#qdrant_cohere_key_group").toggle(settings.enableRerank)
    $("#qdrant_cohere_model_group").toggle(settings.enableRerank)
    $("#qdrant_rerank_candidates_group").toggle(settings.enableRerank)
  })

  $("#qdrant_cohere_key").on("input", function () {
    settings.cohereApiKey = $(this).val()
  })

  $("#qdrant_cohere_model").on("change", function () {
    settings.cohereRerankModel = $(this).val()
  })

  $("#qdrant_rerank_candidates").on("input", function () {
    settings.rerankCandidates = Number.parseInt($(this).val())
    $("#rerank_candidates_display").text(settings.rerankCandidates)
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

  $("#qdrant_save").on("click", () => {
    saveSettings()
    $("#qdrant_status")
      .text("✓ 設定已儲存！")
      .css({ color: "green", background: "#d4edda", border: "1px solid green" })
    setTimeout(() => $("#qdrant_status").text("").css({ background: "", border: "" }), 3000)
  })

  $("#qdrant_test").on("click", async () => {
    $("#qdrant_status")
      .text("測試連線中…")
      .css({ color: "#004085", background: "#cce5ff", border: "1px solid #004085" })

    try {
      const response = await fetch(`${settings.qdrantUrl}/collections`, {
        headers: getQdrantHeaders(),
      })

      if (response.ok) {
        const data = await response.json()
        const collections = data.result?.collections || []
        $("#qdrant_status")
          .text(`✓ 連線成功！找到 ${collections.length} 個 collection。`)
          .css({ color: "green", background: "#d4edda", border: "1px solid green" })
      } else {
        $("#qdrant_status")
          .text("✗ 連線失敗，請檢查 URL。")
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
// EXTENSION INITIALIZATION
// ============================================================================

window.jQuery(async () => {
  loadSettings()
  createSettingsUI()

  // Hook into message events for automatic saving
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
    // Fallback: poll for new messages
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

  console.log("[Qdrant Memory] Extension loaded successfully (v4.0.0 - 繁中、ST 伺服器端設定、混合搜尋 + Cohere Rerank)")
})
