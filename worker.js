import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.1?v=401';

// Configure environment
env.allowLocalModels = false;
env.useBrowserCache = true;

// Define absolute path for local wasm hosting
const baseUrl = self.location.origin + self.location.pathname.substring(0, self.location.pathname.lastIndexOf('/') + 1);
env.backends.onnx.wasm.wasmPaths = baseUrl + 'wasm/';
env.backends.onnx.wasm.proxy = false;

let generatorPipeline = null;
let embeddingPipeline = null;

const EMBEDDING_MODEL = 'Xenova/all-MiniLM-L6-v2';
let CURRENT_LLM_MODEL = 'onnx-community/gemma-4-E2B-it-ONNX'; // Default

self.onmessage = async (e) => {
  const { action, payload } = e.data;

  if (action === 'init') {
    if (payload && payload.modelId) {
        CURRENT_LLM_MODEL = payload.modelId;
    }
    await initModels();
  } else if (action === 'embed') {
    const vector = await embedText(payload.text);
    self.postMessage({ action: 'embed_result', payload: { vector } });
  } else if (action === 'generate') {
    await generateResponse(payload.prompt, payload.context);
  }
};

async function initModels() {
  const modelName = CURRENT_LLM_MODEL.split('/').pop().replace('-ONNX', '');
  self.postMessage({ action: 'status', payload: { text: `Loading Embedding Model...`, progress: 0 } });

  // 1. Load Embedding Model
  embeddingPipeline = await pipeline('feature-extraction', EMBEDDING_MODEL, {
    device: 'webgpu',
    progress_callback: (p) => {
      if (p.status === 'progress') {
        self.postMessage({ action: 'progress', payload: { model: 'embedding', progress: p.progress } });
      }
    }
  });

  self.postMessage({ action: 'status', payload: { text: `Loading ${modelName}...`, progress: 50 } });

  // 2. Load Selected Gemma LLM
  generatorPipeline = await pipeline('text-generation', CURRENT_LLM_MODEL, {
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
  // Use official chat template
  const isGeneralKnowledge = !context || context === 'No relevant context found in local documents.';
  
  const systemInstruction = isGeneralKnowledge 
    ? "Act as a helpful AI assistant. Answer using your own knowledge."
    : `Answer based on the following context. If the answer is not in the context, you MAY use your own knowledge but clearly state that the information was not found in the documents.\n\nCONTEXT:\n${context}`;

  const messages = [
    { 
      role: 'user', 
      content: `${systemInstruction}\n\nQUESTION:\n${userPrompt}` 
    }
  ];

  const fullPrompt = generatorPipeline.tokenizer.apply_chat_template(messages, {
    tokenize: false,
    add_generation_prompt: true,
  });

  const output = await generatorPipeline(fullPrompt, {
    max_new_tokens: 1024,
    do_sample: false,
    repetition_penalty: 1.2,
    return_full_text: false,
    stop_sequences: ["<turn|>", "<channel|>", "<eos>", "<|turn|>"],
    callback_function: (beams) => {
      const decoded = generatorPipeline.tokenizer.decode(beams[0].output_token_ids, {
        skip_special_tokens: true,
      });
      
      let textToPush = decoded;
      
      // Filter reasoning/thought blocks (Gemma 4 specific)
      // The model often starts with "thought\n..."
      if (textToPush.includes('thought')) {
        const parts = textToPush.split(/\n\n/);
        if (parts.length > 1) {
            textToPush = parts.slice(1).join('\n\n');
        } else {
            textToPush = '_Thinking..._';
        }
      }

      // Cleanup any leaked turn markers (should be handled by stop_sequences but just in case)
      textToPush = textToPush.replace(/<\|turn\|>model\n/g, '').replace(/<turn\|>/g, '');
      
      self.postMessage({ action: 'chunk', payload: { text: textToPush } });
    }
  });

  const finalResult = output[0].generated_text;
  self.postMessage({ action: 'generate_complete', payload: { text: finalResult } });
}
