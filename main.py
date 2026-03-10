"""
TrueScore — FastAPI Backend
============================
Espone la pipeline (Extractor → Collector → Engine → Report)
come API REST consumabile dal frontend React.

Endpoints:
  POST /api/analyze          → lancia analisi asincrona, ritorna job_id
  GET  /api/status/{job_id}  → stato avanzamento (SSE stream)
  GET  /api/report/{job_id}  → scarica il PDF generato
  GET  /api/result/{job_id}  → JSON completo del VerificationResult
  GET  /health               → healthcheck

Configurazione via .env:
  GROQ_API_KEY   → LLM gratuito (https://console.groq.com)
  OPENAI_API_KEY → alternativa OpenAI (opzionale)
  PORT           → default 8000
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

# ── FastAPI + CORS ────────────────────────────────────────────────────────────
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
log = logging.getLogger("truescore.api")

# ── Pipeline modules (in /app quando dockerizzato) ───────────────────────────
MODULES_PATH = Path(__file__).parent / "pipeline"
sys.path.insert(0, str(MODULES_PATH))

from extractor import ClaimExtractor, LLMAdapter
from data_collector import DataCollector, InMemoryCache
from verification_engine import VerificationEngine
from report_generator import ReportGenerator, ReportConfig

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TrueScore API",
    version="0.1.0",
    description="Business Verification Intelligence",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # in produzione: limita al dominio Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store (sostituire con Redis in produzione) ──────────────────
JOBS: dict[str, dict] = {}
REPORTS_DIR = Path(tempfile.gettempdir()) / "truescore_reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _get_llm() -> LLMAdapter:
    """Restituisce il miglior LLM disponibile in base alle API key configurate.
    Priorità: Gemini → Groq → OpenAI → Mock (solo dev)
    """
    if os.getenv("GEMINI_API_KEY"):
        log.info("LLM: Google Gemini 2.0 Flash (AI Studio)")
        return LLMAdapter.for_gemini(api_key=os.getenv("GEMINI_API_KEY"))
    if os.getenv("GROQ_API_KEY"):
        log.info("LLM: Groq (LLaMA 3.1 70B)")
        return LLMAdapter(provider="groq", api_key=os.getenv("GROQ_API_KEY"),
                          model="llama-3.1-70b-versatile")
    if os.getenv("OPENAI_API_KEY"):
        log.info("LLM: OpenAI GPT-4o-mini")
        return LLMAdapter(provider="openai", api_key=os.getenv("OPENAI_API_KEY"),
                          model="gpt-4o-mini")
    log.warning("Nessuna API key LLM trovata — usando MockAdapter (solo per dev)")
    from extractor import MockLLMAdapter
    return MockLLMAdapter()


def _job_update(job_id: str, **kwargs):
    """Aggiorna un campo del job e registra il timestamp."""
    if job_id in JOBS:
        JOBS[job_id].update(kwargs)
        JOBS[job_id]["updated_at"] = time.time()


def _emit(job_id: str, step: int, status: str, detail: str):
    """Aggiorna step nel job e logga."""
    _job_update(job_id,
        current_step=step,
        step_status=status,
        step_detail=detail,
    )
    log.info(f"[{job_id[:8]}] Step {step} [{status}] {detail}")


# ─────────────────────────────────────────────
#  Background pipeline task
# ─────────────────────────────────────────────

async def run_pipeline(
    job_id: str,
    company_name: str,
    pitch_text: Optional[str],
    bilancio_text: Optional[str],
    website_url: Optional[str],
    sector: Optional[str],
    crunchbase_api_key: Optional[str],
):
    try:
        _job_update(job_id, status="running", started_at=time.time())
        llm = _get_llm()

        # ── Step 1: Claim Extractor ────────────────────────────────────────
        _emit(job_id, 1, "running", "Analisi documenti e estrazione claim...")
        await asyncio.sleep(0.1)   # yield per permettere SSE flush

        extractor = ClaimExtractor(llm)

        declarative = []
        if pitch_text:
            declarative.append({"type": "text", "text": pitch_text, "label": "pitch_deck"})

        probatory = []
        if bilancio_text:
            probatory.append({"type": "text", "text": bilancio_text, "label": "bilancio_infocamere"})

        extraction = extractor.extract(
            company_name=company_name,
            declarative_sources=declarative,
            probatory_sources=probatory,
        )

        claim_count = len(extraction.claims)
        _emit(job_id, 1, "done", f"{claim_count} claim estratte")
        _job_update(job_id, claim_count=claim_count,
                    financial_data=extraction.financial_data.__dict__
                    if extraction.financial_data else None)

        # ── Step 2: Data Collector ─────────────────────────────────────────
        _emit(job_id, 2, "running", "Raccolta dati da fonti esterne...")
        await asyncio.sleep(0.1)

        fin = extraction.financial_data
        collector = DataCollector(
            financial_data=fin.__dict__ if fin else None,
            crunchbase_api_key=crunchbase_api_key or "",
            cache=InMemoryCache(),
        )

        collection = collector.collect(
            company_name=company_name,
            claims=extraction.claims,
            website_url=website_url or "",
        )

        _emit(job_id, 2, "done",
              f"{len([r for r in collection.results if r.found])} fonti trovate")

        # ── Step 3: Verification Engine ────────────────────────────────────
        _emit(job_id, 3, "running", "Calcolo verdicts e Trust Score...")
        await asyncio.sleep(0.1)

        engine = VerificationEngine()
        verification = engine.verify(
            company_name=company_name,
            claims=extraction.claims,
            collection=collection,
            sector=sector or "default",
        )

        _emit(job_id, 3, "done",
              f"Trust Score: {verification.trust_score:.1f}/10")

        # ── Step 4: Report Generator ───────────────────────────────────────
        _emit(job_id, 4, "running", "Composizione report PDF...")
        await asyncio.sleep(0.1)

        pdf_path = REPORTS_DIR / f"{job_id}.pdf"
        config = ReportConfig(
            company_name=company_name,
            sector=sector or "",
            website_url=website_url or "",
            report_id=f"TS-{job_id[:8].upper()}",
        )

        gen = ReportGenerator()
        gen.generate(verification, config, output_path=str(pdf_path))

        _emit(job_id, 4, "done", "Report pronto")

        # ── Store final result ─────────────────────────────────────────────
        result_dict = {
            "company_name":      verification.company_name,
            "trust_score":       verification.trust_score,
            "trust_score_label": verification.trust_score_label,
            "verdicts": [
                {
                    "id":         v.claim_id,
                    "type":       v.claim_type,
                    "text":       v.claim_text,
                    "declared":   v.declared_value,
                    "verified":   v.verified_value,
                    "verdict":    v.verdict.value if hasattr(v.verdict,"value") else str(v.verdict),
                    "confidence": round(v.evidence_confidence, 2),
                    "magnitude":  round(v.magnitude, 2),
                    "reasoning":  v.reasoning,
                    "sources":    v.sources_used,
                    "flags":      getattr(v, "flags", []),
                }
                for v in verification.verdicts
            ],
            "red_flags":    [v.claim_id for v in verification.red_flags],
            "warnings":     [v.claim_id for v in verification.warnings_list],
            "unverifiable": [v.claim_id for v in verification.unverifiable],
            "pdf_ready":    True,
            "report_id":    config.report_id,
            "generated_at": config.generated_at,
        }

        _job_update(job_id,
            status="done",
            result=result_dict,
            pdf_path=str(pdf_path),
            finished_at=time.time(),
        )
        log.info(f"[{job_id[:8]}] Pipeline completata — "
                 f"Trust Score: {verification.trust_score:.1f}/10")

    except Exception as e:
        log.exception(f"[{job_id[:8]}] Errore pipeline: {e}")
        _job_update(job_id, status="error", error=str(e))


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    company_name:       str             = Form(...),
    pitch_text:         Optional[str]   = Form(None),
    bilancio_text:      Optional[str]   = Form(None),
    website_url:        Optional[str]   = Form(None),
    sector:             Optional[str]   = Form(None),
    crunchbase_api_key: Optional[str]   = Form(None),
    pitch_file:         Optional[UploadFile] = File(None),
    bilancio_file:      Optional[UploadFile] = File(None),
):
    """
    Lancia l'analisi in background.
    Accetta testo o file caricati (pitch deck e/o bilancio).
    Ritorna subito il job_id per il polling.
    """
    if not company_name.strip():
        raise HTTPException(400, "company_name obbligatorio")

    # Leggi contenuto file se forniti
    if pitch_file:
        raw = await pitch_file.read()
        pitch_text = raw.decode("utf-8", errors="replace")

    if bilancio_file:
        raw = await bilancio_file.read()
        bilancio_text = raw.decode("utf-8", errors="replace")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "id":          job_id,
        "status":      "queued",
        "company_name": company_name,
        "current_step": 0,
        "step_status":  "idle",
        "step_detail":  "",
        "created_at":   time.time(),
        "updated_at":   time.time(),
    }

    background_tasks.add_task(
        run_pipeline,
        job_id, company_name,
        pitch_text, bilancio_text,
        website_url, sector, crunchbase_api_key,
    )

    log.info(f"[{job_id[:8]}] Analisi avviata per '{company_name}'")
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    """Stato corrente del job (polling o SSE dal frontend)."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job non trovato")

    return {
        "job_id":       job_id,
        "status":       job["status"],           # queued | running | done | error
        "current_step": job.get("current_step", 0),
        "step_status":  job.get("step_status"),
        "step_detail":  job.get("step_detail"),
        "error":        job.get("error"),
        "claim_count":  job.get("claim_count"),
        "trust_score":  job.get("result", {}).get("trust_score") if job.get("result") else None,
    }


@app.get("/api/status/{job_id}/stream")
async def status_stream(job_id: str):
    """
    Server-Sent Events: emette aggiornamenti di stato fino al completamento.
    Il frontend si connette qui per l'animazione della pipeline in tempo reale.
    """
    if job_id not in JOBS:
        raise HTTPException(404, "Job non trovato")

    async def event_generator():
        last_update = 0.0
        for _ in range(300):   # max 5 minuti (300 × 1s)
            job = JOBS.get(job_id, {})
            updated = job.get("updated_at", 0)

            if updated != last_update:
                last_update = updated
                payload = json.dumps({
                    "status":       job.get("status"),
                    "current_step": job.get("current_step", 0),
                    "step_status":  job.get("step_status"),
                    "step_detail":  job.get("step_detail"),
                    "trust_score":  job.get("result", {}).get("trust_score")
                                    if job.get("result") else None,
                })
                yield f"data: {payload}\n\n"

            if job.get("status") in ("done", "error"):
                yield "data: {\"status\":\"closed\"}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    """JSON completo del VerificationResult."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job non trovato")
    if job["status"] != "done":
        raise HTTPException(202, "Analisi ancora in corso")
    return JSONResponse(job["result"])


@app.get("/api/report/{job_id}")
def download_report(job_id: str):
    """Scarica il PDF del report."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job non trovato")
    if job["status"] != "done":
        raise HTTPException(202, "Report non ancora pronto")

    pdf_path = job.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(404, "PDF non trovato")

    company_slug = job["company_name"].lower().replace(" ", "_").replace(".", "")[:30]
    filename = f"truescore_{company_slug}_{job_id[:8]}.pdf"

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=filename,
    )


# ─────────────────────────────────────────────
#  Entry point locale
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    log.info(f"TrueScore API avviata su http://localhost:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
