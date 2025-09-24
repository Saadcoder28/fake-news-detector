# 📰 Fake News Detector

An AI-powered web app that detects whether a given news statement is **real or fake**, using a trained machine learning model deployed on **Fly.io**, with a sleek **React frontend**.

---

## 🚀 Features

- 🧠 Intelligent fake news detection using a trained ML model
- ⚡ FastAPI backend hosted on Fly.io
- ⚛️ React SPA frontend deployed via Vercel
- 🔁 Smart proxy routing via `vercel.json` for API calls
- 📊 Optionally supports stats and explainability endpoints

---

## 🛠 Tech Stack

### Frontend
- React
- Vercel (Hosting)
- Tailwind CSS *(optional)*

### Backend
- FastAPI
- Scikit-learn / Transformers
- Uvicorn
- Fly.io (Deployment)

---

## 📦 Folder Structure

```
fake-news-detector/
├── api/              # FastAPI backend (Fly)
├── web/              # React frontend (Vercel)
├── proxy/            # Optional NGINX proxy setup
├── .github/          # GitHub Actions / CI
```

---

## ⚙️ Getting Started

### 🔧 Backend (API on Fly)

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload  # Run locally
flyctl deploy               # Deploy to Fly.io
```

### 💻 Frontend (React on Vercel)

```bash
cd web
npm install
npm start                  # Run locally

# Deploy:
# - Push to GitHub
# - Set root directory to "web" in Vercel
# - Add vercel.json to handle API routing
```

**`vercel.json` example:**

```json
{
  "rewrites": [
    { "source": "/predict",        "destination": "https://fake-news-api.fly.dev/predict" },
    { "source": "/batch_predict",  "destination": "https://fake-news-api.fly.dev/batch_predict" },
    { "source": "/stats",          "destination": "https://fake-news-api.fly.dev/stats" },
    { "source": "/explain/:id",    "destination": "https://fake-news-api.fly.dev/explain/:id" },
    { "source": "/health",         "destination": "https://fake-news-api.fly.dev/health" }
  ]
}
```

---

## 🔍 API Endpoints

| Method | Endpoint            | Description                       |
|--------|---------------------|-----------------------------------|
| POST   | `/predict`          | Detect if input text is fake news |
| POST   | `/batch_predict`    | Detect multiple statements         |
| GET    | `/stats`            | Return stats summary               |
| GET    | `/health`           | Health check for uptime monitor   |
| GET    | `/explain/:id`      | Optional: model explanation by ID |

---

## 🧊 Note on Cold Start

The backend hosted on Fly.io may experience a **short delay on first use** due to container cold start and model load time. Subsequent requests are fast.

💡 Add this to `App.js` to ping `/health` on load:

```js
useEffect(() => {
  fetch("/health").catch(() => {});
}, []);
```

---

## 📄 License

MIT License

---

## 🙌 Acknowledgments

Built with ❤️ by **Saad Amin**  
Powered by FastAPI, Vercel, Fly.io, Hugging Face, and React.
