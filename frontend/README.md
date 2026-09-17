# SatQuery AI — Frontend

React + TypeScript + Vite single-page app with two routes:

| Route | Page |
|---|---|
| `/` | Home |
| `/assistant` | Chat assistant (chat history in the browser's localStorage; results from the FastAPI backend) |

## Local development

```bash
npm install
npm run dev            # http://localhost:5173
```

The backend must be running (`cd ../backend && python3 -m uvicorn main:app --port 8000`).

## Configuration

| Variable | Where | Default | Meaning |
|---|---|---|---|
| `VITE_API_BASE` | frontend build | `http://localhost:8000` | Base URL of the FastAPI backend. Read at **build time**. |
| `SATQUERY_ALLOWED_ORIGINS` | backend runtime | *(empty)* | Comma-separated extra browser origins allowed by CORS, e.g. `https://satquery.vercel.app`. Local `http://localhost:*` / `http://127.0.0.1:*` are always allowed. |

Copy `.env.example` to `.env.local` to override `VITE_API_BASE` locally.

## Production build

```bash
npm run build          # tsc -b && vite build → dist/
npm run preview        # serves dist/ with SPA fallback on http://localhost:4173
```

## Deploying the frontend (Vercel or Netlify)

Only the static frontend is deployed. The backend runs separately and must be reachable over **HTTPS** from the
browser: a page served over `https://` cannot call an `http://` API (mixed content is blocked).

### 1. Expose the backend over HTTPS

The backend needs local model weights and Ollama, so it currently runs on the development Mac. For a demo, a
Cloudflare quick tunnel works well:

```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8000     # prints https://<random>.trycloudflare.com
```

Avoid the free ngrok tier for this: it returns an HTML warning page to browser requests (including map image
loads) unless every request sends an `ngrok-skip-browser-warning` header, which `<img>` tags cannot do.

⚠️ A tunnel makes the unauthenticated API (file upload + model execution) publicly reachable. Close it after the demo.

### 2. Create the site

**Vercel** — New Project → import `sudeep-07-hub/SatQuery_AI` → Root Directory `frontend` → framework Vite
(build `npm run build`, output `dist`, from `vercel.json`) → add environment variable
`VITE_API_BASE=https://<tunnel-host>` → Deploy.

**Netlify** — Add new site → import the repository → Base directory `frontend` (build `npm run build`, publish
`dist`, from `netlify.toml`) → add environment variable `VITE_API_BASE=https://<tunnel-host>` → Deploy.

Both configs rewrite unknown paths to `index.html`, so `/assistant` and shared `?job=` links load directly.

### 3. Allow the deployed origin on the backend

```bash
cd ../backend
SATQUERY_ALLOWED_ORIGINS=https://<your-site>.vercel.app python3 -m uvicorn main:app --port 8000
```

### 4. Click-through before promoting to production

On the preview URL: Home → Launch Assistant → attach images and send a real query → collapse and expand the
sidebar → expand an execution trace. Also open `/assistant` directly in a new tab to confirm SPA routing.

Note: `VITE_API_BASE` is baked in at build time. If the tunnel URL changes, update the variable and redeploy.
