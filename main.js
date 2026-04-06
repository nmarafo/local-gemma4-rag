import { db } from './db.js';

// UI Elements
const setupContainer = document.getElementById('setup-container');
const chatContainer = document.getElementById('chat-container');
const initBtn = document.getElementById('init-btn');
const statusText = document.getElementById('status-text');
const progressEmbedding = document.getElementById('progress-embedding');
const progressLLM = document.getElementById('progress-llm');
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');
const chatHistory = document.getElementById('chat-history');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const errorMessage = document.getElementById('error-message');
const docsList = document.getElementById('docs-list');

// Worker Initialization (with Cache Buster)
const worker = new Worker(new URL('./worker.js?v=401', import.meta.url), { type: 'module' });

let isReady = false;
let currentChatResponse = null;
let indexedFiles = new Set();
const modelSelect = document.getElementById('model-select');

// Initialization
initBtn.onclick = () => {
  initBtn.disabled = true;
  modelSelect.disabled = true;
  statusText.innerText = 'Initializing WebGPU Engine...';
  worker.postMessage({ action: 'init', payload: { modelId: modelSelect.value } });
};

// Worker Communication
worker.onmessage = async (e) => {
  const { action, payload } = e.data;

  switch (action) {
    case 'progress':
      if (payload.model === 'embedding') {
        progressEmbedding.style.width = `${payload.progress}%`;
      } else if (payload.model === 'llm') {
        progressLLM.style.width = `${payload.progress}%`;
      }
      break;

    case 'ready':
      isReady = true;
      setupContainer.classList.add('hidden');
      chatContainer.classList.remove('hidden');
      await db.init();
      break;

    case 'chunk':
      updateAIChat(payload.text);
      break;

    case 'generate_complete':
      finalizeAIChat(payload.text);
      break;

    case 'status':
      statusText.innerText = payload.text;
      break;

    case 'error':
      errorMessage.innerText = payload.message;
      errorMessage.style.display = 'block';
      initBtn.disabled = false;
      break;
  }
};

// File Processing
fileInput.onchange = async (e) => {
  const files = e.target.files;
  if (!files.length) return;

  uploadStatus.innerText = 'Processing files... ⏳';
  
  for (const file of files) {
    try {
      const text = await extractText(file);
      if (!text || text.trim().length < 10) {
        console.warn(`Empty or too short content for ${file.name}`);
        continue;
      }

      const chunks = chunkText(text, 1000, 200);
      
      for (const chunk of chunks) {
        const vector = await requestEmbedding(chunk);
        await db.addDocument(chunk, file.name, vector);
      }

      addToFileList(file.name);
    } catch (err) {
      console.error(`Error processing ${file.name}:`, err);
    }
  }

  uploadStatus.innerText = 'Indexing complete! 🎉 Ask away.';
  setTimeout(() => uploadStatus.innerText = 'Drop more files if needed', 3000);
};

async function extractText(file) {
  if (file.type === 'application/pdf') {
    return await extractPdfText(file);
  } else {
    // Default to text (TXT, MD, JS, etc.)
    return await file.text();
  }
}

async function extractPdfText(file) {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  let fullText = '';
  
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const strings = content.items.map(item => item.str);
    fullText += strings.join(' ') + '\n';
  }
  return fullText;
}

function addToFileList(fileName) {
  if (indexedFiles.has(fileName)) return;
  indexedFiles.add(fileName);
  
  const sidebarHint = document.querySelector('#sidebar .hint');
  if (sidebarHint) sidebarHint.remove();

  const li = document.createElement('li');
  li.className = 'doc-item';
  li.innerHTML = `
    <span class="doc-icon">📄</span>
    <span class="doc-name">${fileName}</span>
  `;
  docsList.appendChild(li);
}

// Chat Interaction
sendBtn.onclick = handleChat;
userInput.onkeydown = (e) => { if (e.key === 'Enter') handleChat(); };

async function handleChat() {
  const prompt = userInput.value.trim();
  if (!prompt || !isReady) return;

  userInput.value = '';
  appendUserMessage(prompt);

  // 1. Embed Query
  const queryVector = await requestEmbedding(prompt);

  // 2. Retrieve Context from Orama
  const results = await db.query(queryVector, 5);
  const context = results.length > 0 
    ? results.map(r => `[Source: ${r.fileName}]: ${r.text}`).join('\n---\n')
    : 'No relevant context found in local documents.';

  // 3. Generate with LLM (Pass sources to UI renderer)
  prepareAIChat(results);
  worker.postMessage({ action: 'generate', payload: { prompt, context } });
}

// Helpers
function chunkText(text, size, overlap) {
  const chunks = [];
  for (let i = 0; i < text.length; i += size - overlap) {
    if (i + size > text.length) {
      chunks.push(text.slice(i));
      break;
    }
    chunks.push(text.slice(i, i + size));
  }
  return chunks;
}

function requestEmbedding(text) {
  return new Promise((resolve) => {
    const handler = (e) => {
      if (e.data.action === 'embed_result') {
        worker.removeEventListener('message', handler);
        resolve(e.data.payload.vector);
      }
    };
    worker.addEventListener('message', handler);
    worker.postMessage({ action: 'embed', payload: { text } });
  });
}

function appendUserMessage(text) {
  const div = document.createElement('div');
  div.className = 'message user';
  div.innerHTML = `
    <div class="avatar">U</div>
    <div class="content">${text}</div>
  `;
  chatHistory.appendChild(div);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function prepareAIChat(sources = []) {
  currentChatResponse = document.createElement('div');
  currentChatResponse.className = 'message ai';
  
  let sourcesHtml = '';
  if (sources.length > 0) {
    const snippetsHtml = sources.map(s => `
      <div class="source-snippet">
        <span class="source-meta">${s.fileName} (Score: ${s.score.toFixed(2)})</span>
        <div class="source-text">${s.text}</div>
      </div>
    `).join('');
    
    sourcesHtml = `
      <div class="sources-container">
        <button class="sources-toggle" onclick="this.nextElementSibling.classList.toggle('show')">
          🔍 Show RAG Sources (${sources.length})
        </button>
        <div class="sources-content">
          ${snippetsHtml}
        </div>
      </div>
    `;
  }

  currentChatResponse.innerHTML = `
    <div class="avatar">G4</div>
    <div class="content-wrapper">
        <div class="content"><em>Generating...</em></div>
        ${sourcesHtml}
    </div>
  `;
  chatHistory.appendChild(currentChatResponse);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function updateAIChat(text) {
  if (currentChatResponse) {
    const contentDiv = currentChatResponse.querySelector('.content');
    contentDiv.innerHTML = window.marked ? window.marked.parse(text) : text;
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }
}

function finalizeAIChat(text) {
  if (currentChatResponse) {
    const contentDiv = currentChatResponse.querySelector('.content');
    contentDiv.innerHTML = window.marked ? window.marked.parse(text) : text;
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }
}
