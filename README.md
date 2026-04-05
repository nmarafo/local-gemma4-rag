---
title: Local Gemma 4 RAG
emoji: 🛡️
colorFrom: green
colorTo: gray
sdk: static
pinned: false
header: mini
models:
  - onnx-community/gemma-4-E2B-it-ONNX
  - Xenova/all-MiniLM-L6-v2
---

# 🛡️ Local RAG with Gemma 2B & WebGPU

This is a **privacy-first** document assistant that runs entirely in your browser. 

### Why this is cool:
1. **No Servers:** Your documents never leave your computer.
2. **GPU Accelerated:** Uses **WebGPU** for high-speed inference.
3. **Zero Cost:** No API keys required.

### Technical Details:
- **Engine:** Transformers.js v4 (WebGPU)
- **Vector DB:** Orama v2 (Local WASM)

### Hardware Requirements:
You need a **WebGPU-compatible browser** (Chrome 113+, Edge, or Safari 17.4+) and a dedicated or integrated GPU.
