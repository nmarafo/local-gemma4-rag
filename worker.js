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
    dtype: 'q4', // Quantized to 4-bit for speed/VRAM
    progress_callback: (p) => {
      if (p.status === 'progress') {
        self.postMessage({ action: 'progress', payload: { model: 'llm', progress: p.progress } });
      }
    }
  });

  self.postMessage({ action: 'ready' });
}

async function embedText(text) {
  const output = await embeddingPipeline(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

async function generateResponse(userPrompt, context) {
  const fullPrompt = `<start_of_turn>user
Answer the question based ONLY on the following context. If the context does not contain the answer, say that you don't know.
---
CONTEXT:
${context}
---
QUESTION:
${userPrompt}
<end_of_turn>
<start_of_turn>model
`;

  const output = await generatorPipeline(fullPrompt, {
    max_new_tokens: 512,
    temperature: 0.4, // Lower temperature for more factual RAG
    do_sample: true,
    return_full_text: false, // DO NOT return the prompt, only the new tokens
    callback_function: (beams) => {
      const decoded = generatorPipeline.tokenizer.decode(beams[0].output_token_ids, {
        skip_special_tokens: true,
      });
      
      // In some cases, even with return_full_text: false, the callback might see the whole sequence.
      // We'll handle that by only sending the part after the model turn start if needed.
      const modelTurnMarker = '<start_of_turn>model\n';
      const lastIndex = decoded.lastIndexOf(modelTurnMarker);
      const textToPush = lastIndex !== -1 ? decoded.substring(lastIndex + modelTurnMarker.length) : decoded;
      
      self.postMessage({ action: 'chunk', payload: { text: textToPush } });
    }
  });

  const finalResult = output[0].generated_text;
  self.postMessage({ action: 'generate_complete', payload: { text: finalResult } });
}
