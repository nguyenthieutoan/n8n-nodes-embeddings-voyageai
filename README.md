# n8n Voyage AI Embeddings Community Node

Developed and maintained by **[Jay Nguyen (Nguyễn Thiệu Toàn)](https://nguyenthieutoan.com)**.

🛡️ **[Verified n8n Creator](https://n8n.io/creators/nguyenthieutoan)** | 💼 CEO/Founder of **[GenStaff](https://genstaff.net)**

**Connect with me:**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nguyenthieutoan) [![Facebook](https://img.shields.io/badge/Facebook-1877F2?style=flat&logo=facebook&logoColor=white)](https://www.facebook.com/nguyenthieutoan) [![Website](https://img.shields.io/badge/Website-nguyenthieutoan.com-brightgreen?style=flat)](https://nguyenthieutoan.com) [![Email](https://img.shields.io/badge/Email-me%40nguyenthieutoan.com-blue?style=flat)](mailto:me@nguyenthieutoan.com)

This is an n8n community node that integrates **Voyage AI Embeddings** into the n8n AI ecosystem (LangChain). It acts as a sub-node connected to Vector Store root nodes (such as Supabase, Pinecone, Qdrant, PGVector) to generate vector representations of text queries and documents.

---

## Features

- **Voyage 4 Models Support**: Built-in options for `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-code-3`, `voyage-finance-2`, and `voyage-law-2`.
- **Zero Runtime Dependencies**: Conforms to the strict n8n packaging guideline, ensuring no package collision or dependency bloat in your production deployment.
- **Matryoshka Dimension Scaling**: Custom dimension parameters (256, 512, 1024, 2048) to scale downstream database storage requirements.
- **Output Quantization**: Encodes outputs to `float`, `int8`, `uint8`, `binary`, or `ubinary` representation formats.
- **Base64 Transfer Protocol**: Decodes base64 payload response patterns for high-throughput network performance.
- **Contextualized Truncation**: Truncates inputs safely if they exceed context lengths.

---

## Installation

### For Self-Hosted n8n Instance
In your self-hosted n8n instance directory, run:
```bash
npm install n8n-nodes-embeddings-voyageai
```
Or use the **Community Nodes** menu in your n8n settings dashboard:
1. Go to **Settings** > **Community Nodes**.
2. Click **Install a Node**.
3. Type: `n8n-nodes-embeddings-voyageai`
4. Accept and install.

---

## Local Development & Compilation

### Requirements
- Node.js `v18+` or `v22+`
- `pnpm` or `npm`

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/n8n-voyage-nodes-embedding.git
   cd n8n-voyage-nodes-embedding
   ```
2. Install dependencies:
   ```bash
   pnpm install
   ```
3. Build the node:
   ```bash
   pnpm run build
   ```
4. Link or load it in your local dev stack.

---

## Local Verification using Docker Compose

A `compose.yaml` file is provided in the repository to mount your compiled output directory directly into a local n8n instance for testing:

1. Compile your TS files to JS:
   ```bash
   pnpm run build
   ```
2. Launch n8n using Docker Compose:
   ```bash
   docker compose up -d
   ```
3. Open `http://localhost:5678` in your browser.
4. Navigate to the AI Canvas, add a **Vector Store**, and link the **Voyage AI Embeddings** node!

---

## Author

- **Jay Nguyen (Nguyễn Thiệu Toàn)**
  - **Verified n8n Creator**: [n8n Profile](https://n8n.io/creators/nguyenthieutoan)
  - **CEO & Founder**: [GenStaff](https://genstaff.net)
  - **Website**: [nguyenthieutoan.com](https://nguyenthieutoan.com)
  - **Email**: [me@nguyenthieutoan.com](mailto:me@nguyenthieutoan.com)
  - **LinkedIn**: [nguyenthieutoan](https://www.linkedin.com/in/nguyenthieutoan)
  - **Facebook**: [nguyenthieutoan](https://www.facebook.com/nguyenthieutoan)

---

## License
[MIT](LICENSE)
