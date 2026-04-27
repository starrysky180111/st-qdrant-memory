// Qdrant Memory Extension for SillyTavern
// 從 Qdrant 取回相關記憶並注入對話情境
// Version 3.1.4 - 介面繁中化、設定改存 ST 伺服器端、支援中文角色名集合

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
  chunkTimeout: 30000, // 30 秒沒有新訊息就存檔
  // v3.1.2 新增
  dedupeThreshold: 0.92, // 區塊去重相似度門檻
  preventDuplicateInjection: true, // 防止同一段記憶重複注入
  streamFinalizePollMs: 250,
  streamFinalizeStableMs: 1200,
  streamFinalizeMaxWaitMs: 300000,
  flushAfterAssistant: true,
}

let settings = { ...defaultSettings }
const saveQueue = []
let processingSaveQueue = false

let messageBuffer = []
let lastMessageTime = 0
let chunkTimer = null
let pendingAssistantFinalize = null

// 追蹤已注入記憶的對話狀態，避免重複注入
const memoryInjectionTracker = new Set()

// 為對話狀態產生簡易 hash（用最後幾則訊息特徵組合）
function getChatHash(chat) {
  const lastMessages = chat.slice(-5).map(msg => {
    return `${msg.is_user ? 'U' : 'A'}_${msg.mes?.substring(0, 50) || ''}_${msg.send_date || ''}`
  }).join('|')

  return lastMessages
}

const EMBEDDING_MODEL_OPTIONS = {
  openai: [
    {
      value: "text-embedding-3-large",
      label: "text-embedding-3-large（最高品質）",
    },
    {
      value: "text-embedding-3-small",
      label: "text-embedding-3-small（速度較快）",
    },
    {
      value: "text-embedding-ada-002",
      label: "text-embedding-ada-002（舊版）",
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
// 日期／時間戳正規化
// ============================================================================

/**
 * 將各種日期格式正規化為毫秒級 Unix 時間戳
 */
function normalizeTimestamp(date) {
  // 已經是毫秒等級的時間戳
  if (typeof date === 'number' && date > 1000000000000) {
    return date;
  }

  // 秒等級時間戳，轉成毫秒
  if (typeof date === 'number' && date > 1000000000 && date < 1000000000000) {
    return date * 1000;
  }

  // Date 物件
  if (date instanceof Date) {
    const timestamp = date.getTime();
    if (!isNaN(timestamp)) {
      return timestamp;
    }
  }

  // 字串日期，嘗試解析
  if (typeof date === 'string' && date.trim()) {
    const parsed = new Date(date);
    const timestamp = parsed.getTime();
    if (!isNaN(timestamp)) {
      return timestamp;
    }
  }

  // 解析失敗，回傳當下時間
  if (settings.debugMode) {
    console.warn('[Qdrant Memory] Could not normalize timestamp, using current time. Input:', date);
  }
  return Date.now();
}

/**
 * 將時間戳格式化為 YYYY-MM-DD，用於記憶區塊的日期前綴
 */
function formatDateForChunk(timestamp) {
  try {
    const dateObj = new Date(timestamp);
    if (isNaN(dateObj.getTime())) {
      throw new Error('Invalid date');
    }
    return dateObj.toISOString().split('T')[0]; // YYYY-MM-DD
  } catch (e) {
    console.warn('[Qdrant Memory] Error formatting date:', e, 'timestamp:', timestamp);
    return new Date().toISOString().split('T')[0]; // 失敗則用今天
  }
}

// ============================================================================
// 設定管理（改存 ST 伺服器端 extension_settings）
// ============================================================================

/**
 * 取得 ST 的 extension_settings 儲存空間
 * 嘗試多個可能位置：window.extension_settings、SillyTavern.getContext().extensionSettings
 */
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

/**
 * 觸發 ST 將設定持久化到伺服器
 */
function triggerSTSave() {
  // 優先用 debounced 版本，避免短時間多次寫入
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

/**
 * 從 ST 伺服器端讀取設定（fallback 為 localStorage）
 */
function loadSettings() {
  const store = getExtensionSettingsStore()

  if (store) {
    // 確保 namespace 存在
    if (!store[extensionName] || typeof store[extensionName] !== "object") {
      store[extensionName] = {}
    }

    // 一次性遷移：若伺服器端沒設定但 localStorage 有舊資料，搬過來
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
    // 把預設值補進 store，確保新增的欄位有值
    Object.assign(store[extensionName], settings)
  } else {
    // 完全取不到 extension_settings 時，退回 localStorage
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

/**
 * 將設定寫回 ST 伺服器端
 */
function saveSettings() {
  const store = getExtensionSettingsStore()

  if (store) {
    store[extensionName] = { ...settings }
    const ok = triggerSTSave()
    if (settings.debugMode) {
      console.log(`[Qdrant Memory] 設定已寫入 extension_settings（伺服器持久化：${ok ? "成功" : "未觸發"}）`)
    }
  } else {
    // Fallback: localStorage
    try {
      localStorage.setItem(extensionName, JSON.stringify(settings))
    } catch (e) {
      console.error("[Qdrant Memory] localStorage 寫入失敗：", e)
    }
    console.log("[Qdrant Memory] 設定已儲存（localStorage fallback）")
  }
}

/**
 * 取得指定角色的集合名稱
 *
 * 已修改：保留中文（CJK）、日文（平假名／片假名）字元作為集合名稱
 * 支援：
 *   - a-z, 0-9, _, -
 *   - CJK 統一表意文字（U+4E00–U+9FFF）
 *   - CJK 擴充 A（U+3400–U+4DBF）
 *   - 平假名（U+3040–U+309F）
 *   - 片假名（U+30A0–U+30FF）
 */
function getCollectionName(characterName) {
  if (!settings.usePerCharacterCollections) {
    return settings.collectionName
  }

  // 注意：只用 toLowerCase() 不會影響中日文字
  const sanitized = characterName
    .toLowerCase()
    .replace(/[^a-z0-9_\-\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "")

  return `${settings.collectionName}_${sanitized}`
}

/**
 * 將集合名稱安全地放進 URL 路徑
 * 保留中文等 Unicode 但確保 URL 編碼正確
 */
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
    console.log(`[Qdrant Memory] 自動偵測到本地嵌入維度：${vectorSize}`)
  }
}

function getEmbeddingProviderError() {
  const provider = settings.embeddingProvider || "openai"

  const validProviders = ["openai", "openrouter", "local"]
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
// HTTP HEADERS 與 CSRF Token 處理
// ============================================================================

// 嘗試從各種可能的提供者取得 CSRF token
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

// 從 cookie 讀取指定名稱的值
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

// 取得 SillyTavern API 請求所需的 headers（含 CSRF token）
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

  // 嘗試多個可能位置的 header builder
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

  // 內建方法都失敗，手動組裝
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

// 取得 Qdrant 請求所需的 headers（含 API key）
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
// QDRANT 集合管理
// ============================================================================

// 檢查集合是否存在
async function collectionExists(collectionName) {
  try {
    const response = await fetch(`${settings.qdrantUrl}/collections/${encodeCollectionName(collectionName)}`, {
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

// 為角色建立集合
async function createCollection(collectionName, vectorSize) {
  try {
    const dimensions = Number.isFinite(vectorSize) && vectorSize > 0 ? vectorSize : getEmbeddingDimensions()

    if (!Number.isFinite(dimensions) || dimensions <= 0) {
      console.error(`[Qdrant Memory] Cannot create collection ${collectionName} - invalid embedding dimensions`)
      return false
    }

    const response = await fetch(`${settings.qdrantUrl}/collections/${encodeCollectionName(collectionName)}`, {
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

// 確保集合存在（若不存在則建立）
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
// 嵌入向量生成
// ============================================================================

// 用設定好的 provider 產生嵌入
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
// 記憶搜尋與檢索
// ============================================================================

// 檢查目前要存的區塊是否已存在（去重用）
async function chunkExistsInCollection(collectionName, embedding, text, dedupeThreshold) {
  try {
    const searchPayload = {
      vector: embedding,
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

// 在 Qdrant 中搜尋相關記憶
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

    // 取得所有需要排除的訊息 ID
    const context = getContext()
    const chat = context.chat || []
    const excludedMessageIds = new Set()

    if (settings.retainRecentMessages > 0 && chat.length > settings.retainRecentMessages) {
      const recentMessages = chat.slice(-settings.retainRecentMessages)

      recentMessages.forEach(msg => {
        // 用所有可能的格式產生此訊息的 ID
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

    const searchPayload = {
      vector: embedding,
      limit: settings.memoryLimit * 2, // 多取一些以便後續過濾
      score_threshold: settings.scoreThreshold,
      with_payload: true,
    }

    const filterConditions = []

    // 共用集合時加上角色過濾
    if (!settings.usePerCharacterCollections) {
      filterConditions.push({
        key: "character",
        match: { value: characterName },
      })
    }

    if (filterConditions.length > 0) {
      searchPayload.filter = {
        must: filterConditions,
      }
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
    let results = data.result || []

    // 過濾掉包含被排除訊息 ID 的區塊
    if (excludedMessageIds.size > 0) {
      const beforeFilterCount = results.length

      results = results.filter(memory => {
        const messageIds = memory.payload.messageIds || ""
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

    // 文字相似度去重
    const uniqueResults = []
    const seenTexts = new Set()

    for (const result of results) {
      const text = result.payload?.text || ""

      // 正規化字串以比對（去掉日期前綴與多餘空白）
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
      console.log(`[Qdrant Memory] Found ${results.length} unique memories (after deduplication)`)
    }

    return results
  } catch (error) {
    console.error("[Qdrant Memory] Error searching memories:", error)
    return []
  }
}

// 將取回的記憶格式化成要注入提示的字串
function formatMemories(memories) {
  if (!memories || memories.length === 0) return ""

  let formatted = "\n[過往對話記憶]\n\n"

  // 取得 persona 名稱顯示用
  const personaName = getPersonaName()

  memories.forEach((memory) => {
    const payload = memory.payload

    let speakerLabel
    if (payload.isChunk) {
      // 多人對話區塊：列出所有發話者
      speakerLabel = `對話（${payload.speakers}）`
    } else {
      // 單則訊息（舊格式）
      speakerLabel = payload.speaker === "user"
        ? `${personaName} 說`
        : "角色說"
    }

    let text = payload.text.replace(/\n/g, " ") // 把換行壓平

    const score = (memory.score * 100).toFixed(0)

    formatted += `• ${speakerLabel}：「${text}」（相關度：${score}%）\n\n`
  })

  return formatted
}

// ============================================================================
// 訊息切塊與緩衝
// ============================================================================

function getChatParticipants() {
  const context = getContext()
  const characterName = context.name2

  // 判斷是不是群聊
  const characters = context.characters || []
  const chat = context.chat || []

  // 群聊：從近期訊息抓出所有參與角色名
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

  // 單一角色聊天
  return characterName ? [characterName] : []
}

function createChunkFromBuffer() {
  if (messageBuffer.length === 0) return null

  let chunkText = ""
  const speakers = new Set()
  const messageIds = []
  let totalLength = 0
  const currentTimestamp = Date.now()

  // 取得當前 persona 名稱
  const personaName = getPersonaName()

  // 用發話者標籤組成區塊文字
  messageBuffer.forEach((msg) => {
    const speaker = msg.isUser ? personaName : msg.characterName
    speakers.add(speaker)
    messageIds.push(msg.messageId)

    const line = `${speaker}: ${msg.text}\n`
    chunkText += line
    totalLength += line.length
  })

  // 加上日期前綴
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
    // 為區塊文字產生嵌入
    const embedding = await generateEmbedding(chunk.text)
    if (!embedding) {
      console.error("[Qdrant Memory] Cannot save chunk - embedding generation failed")
      return false
    }

    // 存之前先檢查是否已有相似區塊
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

    // 準備 payload
    const payload = {
      text: chunk.text,
      speakers: chunk.speakers.join(", "),
      messageCount: chunk.messageCount,
      timestamp: chunk.timestamp,
      messageIds: chunk.messageIds.join(","),
      isChunk: true,
    }

    // 將同一個區塊存到所有參與者的集合
    const savePromises = participants.map(async (characterName) => {
      const collectionName = getCollectionName(characterName)

      const collectionReady = await ensureCollection(characterName, embedding.length)
      if (!collectionReady) {
        console.error(`[Qdrant Memory] Cannot save chunk - collection creation failed for ${characterName}`)
        return false
      }

      // 共用集合才把角色名加進 payload
      const characterPayload = settings.usePerCharacterCollections
        ? payload
        : { ...payload, character: characterName }

      // 寫入 Qdrant
      const response = await fetch(`${settings.qdrantUrl}/collections/${encodeCollectionName(collectionName)}/points`, {
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

  // 取得所有參與角色（群聊時不只一個）
  const participants = getChatParticipants()

  if (participants.length === 0) {
    console.error("[Qdrant Memory] No participants found for chunk")
    messageBuffer = []
    return
  }

  // 存到所有參與者的集合
  await saveChunkToQdrant(chunk, participants)

  // 存完清空 buffer
  messageBuffer = []
}

function bufferMessage(text, characterName, isUser, messageId) {
  if (!settings.enabled) return
  if (!settings.autoSaveMemories) return
  if (getEmbeddingProviderError()) return
  if (text.length < settings.minMessageLength) return

  // 確認此類訊息有開啟儲存
  if (isUser && !settings.saveUserMessages) return
  if (!isUser && !settings.saveCharacterMessages) return

  // 加進 buffer
  messageBuffer.push({ text, characterName, isUser, messageId })
  lastMessageTime = Date.now()

  // 計算目前 buffer 大小
  let bufferSize = 0
  messageBuffer.forEach((msg) => {
    bufferSize += msg.text.length + msg.characterName.length + 4 // 4 = ": " + "\n"
  })

  if (settings.debugMode) {
    console.log(`[Qdrant Memory] Buffer: ${messageBuffer.length} messages, ${bufferSize} chars`)
  }

  // 清掉舊計時器
  if (chunkTimer) {
    clearTimeout(chunkTimer)
  }

  // 超過上限就立刻處理
  if (bufferSize >= settings.chunkMaxSize) {
    if (settings.debugMode) {
      console.log(`[Qdrant Memory] Buffer reached max size (${bufferSize}), processing chunk`)
    }
    processMessageBuffer()
  }
  // 達到下限：設一個短計時器
  else if (bufferSize >= settings.chunkMinSize) {
    chunkTimer = setTimeout(() => {
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Buffer reached min size and timeout, processing chunk`)
      }
      processMessageBuffer()
    }, 5000) // 達到下限 5 秒後切塊
  }
  // 否則設長計時器
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

    // 嘗試取得角色的 avatar 檔名
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

    // 處理多種可能的回傳格式，只抽出檔名
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

    // 確保 chatFile 是字串
    if (typeof chatFile !== 'string') {
      console.error("[Qdrant Memory] chatFile is not a string:", chatFile)
      if (chatFile && chatFile.file_name) {
        chatFile = chatFile.file_name
      } else {
        return null
      }
    }

    // API 會自動補回 .jsonl，這裡先去掉
    const fileNameWithoutExt = chatFile.replace(/\.jsonl$/, '')

    const context = getContext()

    // 嘗試取得角色的 avatar 檔名
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

    // 處理多種可能的回傳格式
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
    // 用任一 messageId 比對是否存在
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
    // 跳過系統訊息
    if (msg.is_system) continue

    const text = msg.mes?.trim()
    if (!text || text.length < settings.minMessageLength) continue

    // 確認此類訊息有開啟儲存
    const isUser = msg.is_user || false
    if (isUser && !settings.saveUserMessages) continue
    if (!isUser && !settings.saveCharacterMessages) continue

    // 正規化 send_date
    const normalizedDate = normalizeTimestamp(msg.send_date || Date.now())

    if (settings.debugMode) {
      console.log("[Qdrant Memory] Message date - raw:", msg.send_date, "normalized:", normalizedDate, "formatted:", formatDateForChunk(normalizedDate))
    }

    // 建立訊息物件
    const messageObj = {
      text: text,
      characterName: characterName,
      isUser: isUser,
      messageId: `${characterName}_${normalizedDate}_${messages.indexOf(msg)}`,
      timestamp: normalizedDate,
    }

    const messageSize = text.length + characterName.length + 4

    // 加進去會超過上限就先存目前區塊
    if (currentSize + messageSize > settings.chunkMaxSize && currentChunk.length > 0) {
      chunks.push(createChunkFromMessages(currentChunk))
      currentChunk = []
      currentSize = 0
    }

    currentChunk.push(messageObj)
    currentSize += messageSize

    // 達到下限且訊息數夠多就切塊
    if (currentSize >= settings.chunkMinSize && currentChunk.length >= 3) {
      chunks.push(createChunkFromMessages(currentChunk))
      currentChunk = []
      currentSize = 0
    }
  }

  // 把剩餘訊息打包
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

  // 取得 persona 名稱
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

  // 加上日期前綴
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

  // 進度視窗
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
    // 取得此角色所有聊天檔
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

    // 處理每一個聊天檔
    for (let i = 0; i < chatFiles.length; i++) {
      if (cancelled) break

      const chatFile = chatFiles[i]
      const progress = ((i / chatFiles.length) * 100).toFixed(0)

      $("#qdrant_index_progress").css("width", `${progress}%`)
      $("#qdrant_index_status").text(`處理中：第 ${i + 1}/${chatFiles.length} 個聊天`)
      $("#qdrant_index_details").text(`檔案：${chatFile}`)

      // 載入聊天檔
      const chatData = await loadChatFile(characterName, chatFile)
      if (!chatData || !Array.isArray(chatData)) continue

      // 把訊息切成記憶區塊
      const chunks = createChunksFromChat(chatData, characterName)
      totalChunks += chunks.length

      // 儲存每個區塊
      for (const chunk of chunks) {
        if (cancelled) break

        // 區塊已存在則略過
        const exists = await chunkExists(collectionName, chunk.messageIds)
        if (exists) {
          skippedChunks++
          continue
        }

        // 取得參與角色（群聊時可能不只一個）
        const participants = [characterName]

        // 儲存區塊
        const success = await saveChunkToQdrant(chunk, participants)
        if (success) {
          savedChunks++
        }

        $("#qdrant_index_details").text(`已儲存：${savedChunks}｜略過：${skippedChunks}｜總計：${totalChunks}`)
      }
    }

    // 完成
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
// 生成攔截器
// ============================================================================

// 防止重複注入記憶
globalThis.qdrantMemoryInterceptor = async (chat, contextSize, abort, type) => {
  if (!settings.enabled) {
    if (settings.debugMode) {
      console.log("[Qdrant Memory] Extension disabled, skipping")
    }
    return
  }

  // 用 chat hash 取代 WeakMap 判斷
  if (settings.preventDuplicateInjection) {
    const chatHash = getChatHash(chat)

    if (memoryInjectionTracker.has(chatHash)) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Memories already injected for this chat state, skipping")
      }
      return
    }

    // 標記此狀態已注入
    memoryInjectionTracker.add(chatHash)

    // 控制 tracker 大小避免記憶體外洩（保留最近 50 筆）
    if (memoryInjectionTracker.size > 50) {
      const oldestHash = memoryInjectionTracker.values().next().value
      memoryInjectionTracker.delete(oldestHash)
    }
  }

  try {
    const context = getContext()
    const characterName = context.name2

    // 沒選角色就跳過
    if (!characterName) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] No character selected, skipping")
      }
      return
    }

    // 找最後一則使用者訊息當作查詢 query
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

    // 搜尋相關記憶
    const memories = await searchMemories(query, characterName)

    if (memories.length > 0) {
      const memoryText = formatMemories(memories)

      if (settings.debugMode) {
        console.log("[Qdrant Memory] Retrieved memories:", memoryText)
      }

      // 建立記憶條目
      const memoryEntry = {
        name: "System",
        is_user: false,
        is_system: true,
        mes: memoryText,
        send_date: Date.now(),
      }

      // 從末端往前數指定位置插入
      const insertIndex = Math.max(0, chat.length - settings.memoryPosition)
      chat.splice(insertIndex, 0, memoryEntry)

      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Injected ${memories.length} memories at position ${insertIndex}`)
      }

      const toastr = window.toastr
      if (settings.showMemoryNotifications) {
        toastr.info(`已取得 ${memories.length} 則相關記憶`, "Qdrant Memory", { timeOut: 2000 })
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

  // 安全檢查：確保是角色訊息
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

    // 把整則訊息放進 buffer
    bufferMessage(text, characterName, false, messageId)

    // 訊息夠多就立刻 flush
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

    // debug 模式每 4 秒記錄一次
    if (settings.debugMode && pollCount % 16 === 0) {
      console.log(`[Qdrant Memory] Stream poll check #${pollCount}: length=${currentText.length}`)
    }

    // 偵測文字是否變動
    if (currentText !== pendingAssistantFinalize.lastText) {
      if (settings.debugMode && pollCount % 4 === 0) {
        console.log(`[Qdrant Memory] Text changed: ${pendingAssistantFinalize.lastText.length} → ${currentText.length}`)
      }
      pendingAssistantFinalize.lastText = currentText
      pendingAssistantFinalize.lastChangeAt = now
    }

    const stableDuration = now - pendingAssistantFinalize.lastChangeAt
    const totalDuration = now - pendingAssistantFinalize.startedAt

    // 如果訊息變成使用者送出（對話繼續了），就 finalize 上一則
    if (currentLastMessage.is_user) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Detected new user message, finalizing previous assistant message")
      }
      finalizeAssistant(pendingAssistantFinalize.lastText, "new user message detected")
      return
    }

    // 文字穩定夠久就 finalize
    if (stableDuration >= stableMs) {
      finalizeAssistant(pendingAssistantFinalize.lastText, `stable for ${stableDuration}ms`)
      return
    }

    // 超過最大等待時間
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

    // 取最後一則訊息
    const lastMessage = chat[chat.length - 1]

    // 正規化 send_date 用來組 messageId
    const normalizedDate = normalizeTimestamp(lastMessage.send_date || Date.now())

    // 為這則訊息產生唯一 ID
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
      // 使用者訊息立刻緩衝
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Buffering user message immediately")
      }
      bufferMessage(lastMessage.mes, characterName, true, messageId)
    } else {
      // 角色訊息要等串流結束才能 finalize
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

  // 嘗試多個可能位置
  const personaName =
    context.name1 ||                              // 標準位置
    context.persona?.name ||                      // 替代位置
    window.name1 ||                               // window 直存
    window.SillyTavern?.getContext?.()?.name1 ||  // 透過 ST API
    "User"                                        // 通用 fallback

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
                <small style="color: #666;">從最後一則往前數，插入記憶的位置</small>
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

  // 掛上訊息事件以自動儲存
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
    // 降級：用輪詢偵測新訊息
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

  console.log("[Qdrant Memory] 擴充功能載入完成（v3.1.4 - 介面繁中化、設定改存伺服器、支援中文集合名）")
})
