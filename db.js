import { create, insert, search } from 'https://cdn.jsdelivr.net/npm/@orama/orama@latest/+esm';

export class VectorDB {
  constructor() {
    this.db = null;
    this.isInitialized = false;
  }

  async init() {
    if (this.isInitialized) return;

    this.db = await create({
      schema: {
        text: 'string',
        fileName: 'string',
        embedding: 'vector[384]', // MiniLM-L6-v2 dimension
      },
      indexStyles: {
        embedding: 'cosine', // Recommended for normalized embeddings
      },
    });

    this.isInitialized = true;
    console.log('📦 Orama Vector DB initialized');
  }

  async addDocument(text, fileName, embedding) {
    if (!this.isInitialized) await this.init();

    return await insert(this.db, {
      text,
      fileName,
      embedding,
    });
  }

  async query(vector, limit = 5) {
    if (!this.isInitialized) await this.init();

    const results = await search(this.db, {
      mode: 'vector',
      vector: {
        value: vector,
        property: 'embedding',
      },
      similarity: 0.5,
      limit: limit,
    });

    return results.hits.map(hit => ({
      text: hit.document.text,
      fileName: hit.document.fileName,
      score: hit.score,
    }));
  }

  async clear() {
    this.isInitialized = false;
    await this.init();
  }
}

// Singleton instance
export const db = new VectorDB();
