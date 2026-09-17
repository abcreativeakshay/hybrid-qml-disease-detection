---
title: Hybrid QML Disease Detection
emoji: 🧬
colorFrom: indigo
colorTo: green
sdk: static
app_build_command: npm run build
app_file: dist/index.html
pinned: false
---

# Hybrid QML Disease Detection (Static Demo)

This is the static frontend for the Hybrid Quantum Machine Learning platform.
It is built with React + TypeScript + Vite and reads pre-computed model metrics and explainability data.

## Local Development

```bash
npm install
npm run dev
```

## Hugging Face Spaces Deployment
Because this uses the `static` SDK, it is 100% free to host on Hugging Face Spaces. 
When pushed to HF Spaces, the `app_build_command` automatically runs `npm run build` and serves `dist/index.html`.
