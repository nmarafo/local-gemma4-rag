import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.1?v=401';

// Configure environment
env.allowLocalModels = false;
env.useBrowserCache = true;

// Define absolute path for local wasm hosting
const baseUrl = self.location.origin + self.location.pathname.substring(0, self.location.pathname.lastIndexOf('/') + 1);
env.backends.onnx.wasm.wasmPaths = baseUrl + 'wasm/';
env.backends.onnx.wasm.proxy = false;

let embeddingPipeline = null;
let generatorPipeline = null;

const EMBEDDING_MODEL = 'Xenova/all-MiniLM-L6-v2';
const LLM_MODEL = 'onnx-community/gemma-4-E2B-it-ONNX';

self.onmessage = async (e) => {
  const { action, payload } = e.data;

  try {
    switch (action) {
      case 'init':
        await initModels();
        break;
      case 'embed':
        const vector = await embedText(payload.text);
        self.postMessage({ action: 'embed_result', payload: { vector } });
        break;
      case 'generate':
        await generateResponse(payload.prompt, payload.context);
        break;
      default:
        console.warn('Unknown worker action:', action);
    }
  } catch (error) {
    self.postMessage({ action: 'error', payload: { message: error.message } });
    console.error('Worker error:', error);
  }
};

async function initModels() {
  self.postMessage({ action: 'status', payload: { text: 'Loading AI Models...', progress: 0 } });

  // 1. Load Embedding Model
  embeddingPipeline = await pipeline('feature-extraction', EMBEDDING_MODEL, {
    device: 'webgpu',
    progress_callback: (p) => {
      if (p.status === 'progress') {
        self.postMessage({ action: 'progress', payload: { model: 'embedding', progress: p.progress } });
      }
    }
  });

  self.postMessage({ action: 'status', payload: { text: 'Loading Gemma 4 (1.6GB)...', progress: 50 } });

  // 2. Load Gemma LLM
  generatorPipeline = await pipeline('text-generation', LLM_MODEL, {
    device: 'webgpu',
    dtype: 'q4',
    progress_callback: (p) => {
      if (p.status === 'progress') {
        self.postMessage({ action: 'progress', payload: { model: 'llm', progress: p.progress } });
      }
    }
  });

  // 3. GPU Warmup (compile shaders)
  self.postMessage({ action: 'status', payload: { text: 'Warming up GPU...', progress: 95 } });
  await generatorPipeline('warmup', { max_new_tokens: 1 });

  self.postMessage({ action: 'ready' });
}

async function embedText(text) {
  const output = await embeddingPipeline(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

async function generateResponse(userPrompt, context) {
  // Gemma 4 / Gemma 2 Chat Template
  const fullPrompt = `<|turn|>user
Answer the question based ONLY on the following context. If the context does not contain the answer, say that you don't know.
---
CONTEXT:
${context}
---
QUESTION:
${userPrompt}<turn|>
<|turn|>model
`;

  const output = await generatorPipeline(fullPrompt, {
    max_new_tokens: 1024, // Reasoning models like Gemma 4 need space for "thinking"
    do_sample: false,     // Faster for RAG
    repetition_penalty: 1.2,
    return_full_text: false,
    stop_sequences: ["<channel|>", "<turn|>", "<eos>", "<|turn|>"],
    callback_function: (beams) => {
      const decoded = generatorPipeline.tokenizer.decode(beams[0].output_token_ids, {
        skip_special_tokens: true,
      });
      
      // Gemma 4 specific filtering:
      // The model generates: thought [internal monologue] <|channel|> [actual text]
      // Or simply starts with "thought" when configured to skip special tokens.
      
      let textToPush = decoded;
      
      // 1. Remove the turn/model markers if they leak
      textToPush = textToPush.replace(/<\|turn\|>model\n/g, '');
      
      // 2. Filter internal reasoning/thought blocks
      // These usually look like "thought [text]" at the start
      if (textToPush.startsWith('thought')) {
        // Look for the end of the thought block if special tokens skipped
        // usually the model starts the real answer after a newline or specific marker
        const parts = textToPush.split(/\n\n/); // Reasoning models often separate thought from answer with double newline
        if (parts.length > 1) {
            textToPush = parts.slice(1).join('\n\n');
        } else {
            // If still thinking, we send a subtle indicator or nothing yet
            textToPush = '_Thinking..._'; 
        }
      }
      
      self.postMessage({ action: 'chunk', payload: { text: textToPush } });
    }
  });

  const finalResult = output[0].generated_text;
  self.postMessage({ action: 'generate_complete', payload: { text: finalResult } });
}
