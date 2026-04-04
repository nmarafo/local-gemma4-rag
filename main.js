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

// Worker Initialization
const worker = new Worker(new URL('./worker.js?v=401', import.meta.url), { type: 'module' });

let isReady = false;
let currentChatResponse = null;

// Initialization
initBtn.onclick = () => {
  initBtn.disabled = true;
  statusText.innerText = 'Initializing WebGPU Engine...';
  worker.postMessage({ action: 'init' });
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
    const text = await file.text();
    const chunks = chunkText(text, 500, 100);
    
    for (const chunk of chunks) {
      // Request embedding from worker
      const vector = await requestEmbedding(chunk);
      await db.addDocument(chunk, file.name, vector);
    }
  }

  uploadStatus.innerText = 'Indexing complete! 🎉 Ask away.';
  setTimeout(() => uploadStatus.innerText = 'Drop more files if needed', 3000);
};

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
  const results = await db.query(queryVector, 3);
  const context = results.map(r => `[From ${r.fileName}]: ${r.text}`).join('\n---\n');

  // 3. Generate with LLM
  prepareAIChat();
  worker.postMessage({ action: 'generate', payload: { prompt, context } });
}

// Helpers
function chunkText(text, size, overlap) {
  const chunks = [];
  for (let i = 0; i < text.length; i += size - overlap) {
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
  div.innerText = text;
  chatHistory.appendChild(div);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function prepareAIChat() {
  currentChatResponse = document.createElement('div');
  currentChatResponse.className = 'message ai';
  currentChatResponse.innerHTML = '<em>Thinking...</em>';
  chatHistory.appendChild(currentChatResponse);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function updateAIChat(text) {
  if (currentChatResponse) {
    // We use marked for rendering if available
    currentChatResponse.innerHTML = window.marked ? window.marked.parse(text) : text;
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }
}

function finalizeAIChat(text) {
  if (currentChatResponse) {
    currentChatResponse.innerHTML = window.marked ? window.marked.parse(text) : text;
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }
}
