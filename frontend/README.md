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

### 1. Host the backend over HTTPS

**Option A — Render free tier (no local machine needed, reduced capability).** `render.yaml` at the
repository root is a Render Blueprint: in Render choose *New → Blueprint*, connect this repository, and apply.
It builds `backend/` with `requirements-render.txt` (CPU-only PyTorch, no model weights) and runs it with:

| Setting | Value | Why |
|---|---|---|
| `SATQUERY_QWEN_BACKEND` | `none` | No LLM fits in 512 MB; tool selection uses the Tool Registry rules and answers use the evidence-only template |
| `SATQUERY_DISABLED_TOOLS` | `single_image_vqa,optical_sar_fusion` | PaliGemma and CROMA need several GB of RAM |
| `SATQUERY_ALLOWED_ORIGINS` | the Vercel site | CORS |
| `SATQUERY_MAX_UPLOAD_MB` | `20` | Uploads are held in memory |
| `SATQUERY_MAX_JOBS` | `15` | Finished jobs keep their rasters in memory; oldest are evicted |

What works there: change detection (classical SAR/optical), evidence, map overlays, exports. Single-image
questions are answered with an explicit "disabled on this deployment" message. The free plan sleeps after
15 minutes without traffic (the next request takes about a minute), and its disk is wiped on restart, so shared
`?job=` links do not survive a sleep. Measured locally with the same packages on Python 3.12: ~234 MB idle,
~284 MB after several change jobs.

Then set `VITE_API_BASE=https://<service>.onrender.com` in Vercel and redeploy.

**Option B — full stack on the development Mac through a tunnel** (Qwen3, PaliGemma; the Mac must stay on):

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
Missing files under `/assets/` return 404 instead of the HTML page.

**CLI alternative (Vercel)** — from this `frontend/` directory:

```bash
npx vercel link --project satquery-ai
printf '%s' "https://<tunnel-host>" | npx vercel env add VITE_API_BASE production
printf '%s' "https://<tunnel-host>" | npx vercel env add VITE_API_BASE preview --yes
npx vercel deploy          # preview
npx vercel deploy --prod   # production
```

Pitfalls seen during the first deployment:

- Set `VITE_API_BASE` as a **project environment variable**, not only with `--build-env`: `vercel promote`
  rebuilds for production with the project's variables, and a missing value silently falls back to
  `http://localhost:8000`, which an HTTPS page cannot call.
- A project's **first** deployment is always assigned to production.
- Preview URLs are behind Vercel Deployment Protection (login required); use `npx vercel curl <path>
  --deployment <url>` to check them, or open them while logged in.
- A brand-new `*.trycloudflare.com` hostname can take a while to resolve on the machine that created it
  (negative DNS cache) even though it already works elsewhere.

### 3. Allow the deployed origin on the backend

```bash
cd ../backend
SATQUERY_ALLOWED_ORIGINS=https://<your-site>.vercel.app python3 -m uvicorn main:app --port 8000
```

### 4. Click-through before promoting to production

On the preview URL: Home → Launch Assistant → attach images and send a real query → collapse and expand the
sidebar → expand an execution trace. Also open `/assistant` directly in a new tab to confirm SPA routing.

Note: `VITE_API_BASE` is baked in at build time. If the tunnel URL changes, update the variable and redeploy.
