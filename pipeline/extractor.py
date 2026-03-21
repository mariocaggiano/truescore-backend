"""
TrueScore — Claim Extractor
============================
Estrae claim verificabili da documenti aziendali (pitch deck, siti web,
bilanci Infocamere) e le classifica per tipo, specificità e verificabilità.

Compatibile con: Groq (free), Ollama (locale), OpenAI, Anthropic.
Dipendenze: pdfplumber, beautifulsoup4, requests, pytesseract (opzionale)
"""

import json
import re
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import time
import pdfplumber
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger("claim_extractor")


# ─────────────────────────────────────────────
#  Tipi di dato
# ─────────────────────────────────────────────

class ClaimType(str, Enum):
    REVENUE       = "revenue"
    PARTNER_COUNT = "partner_count"
    FUNDING       = "funding"
    TEAM_SIZE     = "team_size"
    OTHER         = "other"


class DocumentType(str, Enum):
    DECLARATIVE = "declarative"   # fonte di claim (pitch deck, sito)
    PROBATORY   = "probatory"     # fonte di verifica (bilancio)


class Specificity(str, Enum):
    HIGH   = "high"    # numero esatto, data, importo preciso
    MEDIUM = "medium"  # range, "circa", "oltre"
    LOW    = "low"     # "centinaia di", "molti", non quantificato


@dataclass
class Claim:
    id: str
    type: ClaimType
    text: str                        # testo esatto dalla fonte
    source_document: str             # nome del documento di origine
    source_location: str             # es. "slide 4", "homepage", "sezione About"
    specificity: Specificity
    verifiable: bool
    extraction_confidence: float     # 0.0 – 1.0: quanto è sicura l'estrazione
    normalized_value: Optional[str] = None   # valore pulito es. "3800000" per €3.8M
    notes: str = ""


@dataclass
class FinancialData:
    """Dati estratti da un bilancio Infocamere (documento probatorio)."""
    source_document: str
    exercise_year: Optional[int]       = None
    revenues: Optional[float]          = None   # ricavi vendite e prestazioni
    total_assets: Optional[float]      = None
    employees: Optional[int]           = None
    legal_form: Optional[str]          = None
    share_capital: Optional[float]     = None
    extraction_confidence: float       = 0.0
    raw_excerpt: str                   = ""


@dataclass
class ExtractionResult:
    company_name: str
    claims: list[Claim]             = field(default_factory=list)
    financial_data: Optional[FinancialData] = None
    key_people: list[dict]          = field(default_factory=list)
    tone_analysis: Optional[dict]   = None
    errors: list[str]               = field(default_factory=list)
    warnings: list[str]             = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"  Azienda     : {self.company_name}",
            f"  Claim       : {len(self.claims)} estratte",
            f"  Bilancio    : {'sì' if self.financial_data else 'no'}",
            f"  Errori      : {len(self.errors)}",
        ]
        if self.claims:
            by_type = {}
            for c in self.claims:
                by_type[c.type] = by_type.get(c.type, 0) + 1
            for t, n in by_type.items():
                lines.append(f"    [{t}] {n}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  LLM Adapter (intercambiabile)
# ─────────────────────────────────────────────

class LLMAdapter:
    """
    Adapter generico per chiamate LLM.
    Supporta: Groq, OpenAI, Gemini nativo, Ollama.
    """

    ENDPOINTS = {
        "groq":   "https://api.groq.com/openai/v1/chat/completions",
        "openai": "https://api.openai.com/v1/chat/completions",
        "ollama": "{base_url}/api/chat",
    }
    GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

    def __init__(
        self,
        provider: str = "groq",
        api_key: str = "",
        model: str = "llama-3.1-70b-versatile",
        base_url: str = "http://localhost:11434",
        timeout: int = 90,
        extra_base_url: str = "",   # sovrascrive ENDPOINTS[provider] se valorizzato
    ):
        self.provider       = provider
        self.api_key        = api_key
        self.model          = model
        self.base_url       = base_url
        self.timeout        = timeout
        self.extra_base_url = extra_base_url

    @classmethod
    def for_gemini(cls, api_key: str, model: str = "gemini-2.0-flash-lite") -> "LLMAdapter":
        """Usa API nativa Gemini — 15 req/min gratis, più stabile."""
        return cls(provider="gemini", api_key=api_key, model=model)

    def complete(self, system: str, user: str) -> str:
        """Esegue una chiamata LLM e restituisce il testo della risposta."""
        if self.provider == "gemini":
            return self._call_gemini_native(system, user)
        if self.provider == "ollama":
            return self._call_ollama(system, user)
        return self._call_openai_compatible(system, user)

    def _call_gemini_native(self, system: str, user: str) -> str:
        """Chiama l'API nativa Gemini (15 req/min free tier)."""
        url = self.GEMINI_ENDPOINT.format(model=self.model, key=self.api_key)
        payload = {
            "contents": [{
                "parts": [{"text": system + "\n\n" + user}]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048,
            },
        }
        delays = [10, 30, 60]
        for attempt, delay in enumerate(delays + [None], start=1):
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 429:
                if delay is None:
                    resp.raise_for_status()
                log.warning(f"Gemini rate limit (tentativo {attempt}/{len(delays)+1}) — riprovo tra {delay}s...")
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    def _call_openai_compatible(self, system: str, user: str) -> str:
        url = self.extra_base_url if self.extra_base_url else self.ENDPOINTS[self.provider]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }
        delays = [5, 15, 30]
        for attempt, delay in enumerate(delays + [None], start=1):
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code == 429:
                if delay is None:
                    resp.raise_for_status()
                log.warning(f"Rate limit 429 (tentativo {attempt}/{len(delays)+1}) — riprovo tra {delay}s...")
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def _call_ollama(self, system: str, user: str) -> str:
        url = self.ENDPOINTS["ollama"].format(base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


TONE_ANALYSIS_SYSTEM = """Sei un analista di comunicazione specializzato nella valutazione di materiali di marketing aziendale.
Il tuo compito è analizzare il tono e il linguaggio di un pitch deck o one-pager aziendale.

Cerca specificamente:

1. LINGUAGGIO INFLAZIONATO — frasi che esagerano o non sono verificabili:
   Esempi: "leader del mercato", "unici al mondo", "rivoluzionario", "disrupting", 
   "game changer", "best-in-class", "world-class", "innovativo" senza contesto,
   "traction", "ecosystem", "leverage", "synergy", "scalabile" senza dati

2. NUMERI SENZA CONTESTO — cifre presentate senza benchmark o spiegazione:
   Esempi: "crescita del 300%" (da cosa?), "10.000 clienti" (in quanto tempo?),
   "mercato da 10 miliardi" (quota accessibile?), "NPS di 90" (campione?)

3. VAGHEZZA STRATEGICA — affermazioni non falsificabili:
   Esempi: "stiamo conquistando il mercato", "forte interesse da parte dei clienti",
   "traction significativa", "pipeline molto solida", "team di esperti"

4. CLAIM ASSOLUTE — superlative non supportate:
   Esempi: "la migliore soluzione", "la più completa", "nessuno fa questo"

Per ogni istanza trovata rispondi SOLO con questo JSON (array):
[
  {
    "category": "inflated_language|number_without_context|strategic_vagueness|absolute_claim",
    "phrase": "frase esatta trovata nel testo (max 80 caratteri)",
    "explanation": "perché è un segnale di attenzione (max 120 caratteri)",
    "severity": "high|medium|low",
    "location": "dove appare nel documento (es. slide 2, sezione traction)"
  }
]

REGOLE:
- Sii preciso: estrai la frase ESATTA dal testo
- Non inventare problemi inesistenti
- Se il documento è sobrio e factual, restituisci []
- Severity HIGH = affermazione chiaramente non verificabile o fuorviante
- Severity MEDIUM = affermazione vaga ma comune nel settore
- Severity LOW = linguaggio di marketing standard accettabile
- Rispondi SOLO con JSON valido, nessun testo prima o dopo
"""

PEOPLE_EXTRACTION_SYSTEM = """Sei un analista di due diligence. Estrai dal testo le persone chiave dell'azienda.

Cerca: fondatori, CEO, CFO, COO, CTO, CMO, managing director, direttori, responsabili, partner, board members.

Per ogni persona trovata rispondi SOLO con un array JSON:
[
  {
    "name": "Nome Cognome",
    "role": "Ruolo esatto come appare nel testo",
    "role_category": "founder|ceo|cfo|coo|cto|cmo|director|other",
    "context": "frase o contesto in cui appare la persona",
    "confidence": 0.0-1.0
  }
]

REGOLE:
- Includi solo persone reali con nome e cognome (almeno 2 parole)
- Escludi nomi generici ("il fondatore", "il team", "i nostri esperti")
- Se non trovi persone, rispondi con []
- Rispondi SOLO con JSON valido, nessun testo prima o dopo
"""



# ─────────────────────────────────────────────
#  Ingestion — lettura documenti
# ─────────────────────────────────────────────

class DocumentIngester:
    """Converte documenti di vario formato in testo normalizzato."""

    @staticmethod
    def from_pdf(path: str | Path) -> tuple[str, str]:
        """
        Legge un PDF e restituisce (testo, metodo_usato).
        Prima prova pdfplumber (PDF nativi), poi OCR come fallback.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")

        text = DocumentIngester._extract_with_pdfplumber(path)
        method = "pdfplumber"

        if len(text.strip()) < 100:
            log.info("pdfplumber ha estratto poco testo, provo OCR...")
            text = DocumentIngester._extract_with_ocr(path)
            method = "ocr"

        return DocumentIngester._normalize_text(text), method

    @staticmethod
    def _extract_with_pdfplumber(path: Path) -> str:
        chunks = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    chunks.append(f"[PAGINA {i+1}]\n{page_text}")

                # Estrai anche testo dalle tabelle
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        row_text = " | ".join(str(c or "").strip() for c in row if c)
                        if row_text.strip():
                            chunks.append(row_text)

        return "\n\n".join(chunks)

    @staticmethod
    def _extract_with_ocr(path: Path) -> str:
        try:
            import pytesseract
            from PIL import Image
            import fitz  # PyMuPDF

            doc = fitz.open(str(path))
            texts = []
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                texts.append(pytesseract.image_to_string(img, lang="ita+eng"))
            return "\n\n".join(texts)

        except ImportError:
            log.warning("pytesseract o PyMuPDF non disponibili — OCR non eseguito.")
            return ""

    @staticmethod
    def from_url(url: str, timeout: int = 15) -> str:
        """Scarica e pulisce il testo di una pagina web."""
        headers = {"User-Agent": "Mozilla/5.0 (compatible; TrueScore/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Rimuovi elementi non-contenuto
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "iframe", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        return DocumentIngester._normalize_text(text)

    @staticmethod
    def from_text(text: str) -> str:
        """Accetta testo grezzo (es. corpo di una email)."""
        return DocumentIngester._normalize_text(text)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Rimuove rumore: spazi multipli, linee vuote eccessive, caratteri strani."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[^\S\n]+\n", "\n", text)
        return text.strip()


# ─────────────────────────────────────────────
#  Chunker semantico
# ─────────────────────────────────────────────

class SemanticChunker:
    """
    Divide il testo in chunk semanticamente coerenti.
    Rispetta i marker di pagina inseriti da DocumentIngester.
    Non fa chunking fisso a N caratteri.
    """

    MAX_CHUNK_CHARS = 3000   # limite per singola chiamata LLM
    OVERLAP_CHARS   = 200    # overlap tra chunk consecutivi

    @classmethod
    def chunk(cls, text: str) -> list[dict]:
        """
        Restituisce lista di dict: {text, label}
        dove label è es. "PAGINA 3" o "sezione_1"
        """
        # Se ci sono marker di pagina, usa quelli come delimitatori naturali
        page_chunks = cls._split_by_page_markers(text)
        if len(page_chunks) > 1:
            return cls._merge_small_chunks(page_chunks)

        # Altrimenti usa paragrafi
        return cls._split_by_paragraphs(text)

    @classmethod
    def _split_by_page_markers(cls, text: str) -> list[dict]:
        pattern = r"\[PAGINA (\d+)\]"

        # Verifica se ci sono marker prima di fare lo split
        if not re.search(pattern, text):
            return []

        parts = re.split(pattern, text)
        chunks = []
        i = 0

        while i < len(parts):
            part = parts[i].strip()
            # re.split con gruppo catturante alterna: [testo_pre, num, testo, num, testo...]
            if re.match(r"^\d+$", part):
                label = f"PAGINA {part}"
                content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if content:
                    chunks.append({"text": content, "label": label})
                i += 2
            else:
                if part:
                    chunks.append({"text": part, "label": "intro"})
                i += 1

        return chunks

    @classmethod
    def _split_by_paragraphs(cls, text: str) -> list[dict]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks, current, label_n = [], "", 0

        for para in paragraphs:
            if len(current) + len(para) > cls.MAX_CHUNK_CHARS and current:
                chunks.append({"text": current.strip(), "label": f"sezione_{label_n}"})
                current = current[-cls.OVERLAP_CHARS:]   # overlap
                label_n += 1
            current += "\n\n" + para

        if current.strip():
            chunks.append({"text": current.strip(), "label": f"sezione_{label_n}"})

        return chunks

    @classmethod
    def _merge_small_chunks(cls, chunks: list[dict]) -> list[dict]:
        """Unisce chunk troppo piccoli con il successivo."""
        merged, buffer = [], {}
        for chunk in chunks:
            if not buffer:
                buffer = chunk.copy()
            elif len(buffer["text"]) + len(chunk["text"]) < cls.MAX_CHUNK_CHARS:
                buffer["text"] += "\n\n" + chunk["text"]
                buffer["label"] += f"+{chunk['label']}"
            else:
                merged.append(buffer)
                buffer = chunk.copy()
        if buffer:
            merged.append(buffer)
        return merged


# ─────────────────────────────────────────────
#  Prompt templates
# ─────────────────────────────────────────────

CLAIM_EXTRACTION_SYSTEM = """Sei un analista di due diligence specializzato nella verifica di affermazioni aziendali.
Il tuo compito è estrarre dal testo SOLO le affermazioni fattuali verificabili con fonti pubbliche.

ESTRAI affermazioni che riguardano:
- Ricavi, fatturato, MRR/ARR (tipo: revenue)
- Numero di clienti, partner, strutture, location, operatori (tipo: partner_count)
- Funding ricevuto: round, importi, investitori (tipo: funding)
- Dimensione del team, numero dipendenti (tipo: team_size)
- Altre metriche quantificate verificabili (tipo: other)

NON estrarre:
- Opinioni soggettive ("siamo i migliori", "leader del mercato")
- Proiezioni future ("prevediamo di raggiungere")
- Piani strategici o intenzioni

ESTRAI ANCHE (con specificity "low"):
- Affermazioni quantificate in modo approssimativo: "decine di clienti" → partner_count, normalized_value: null
- Range numerici: "tra 20 e 30 persone" → team_size, normalized_value: "25" (valore medio)
- Numeri menzionati in contesto: "serviamo 50 aziende" → partner_count even if informal
- Anni di attività come proxy: "fondata nel 2018" → notes: "6 anni di attività"

Se il testo è in italiano, estrai comunque le claim in italiano.
Se il testo non ha NESSUN numero o quantità, allora restituisci [].

Per ogni claim estratta rispondi con questo JSON (array, anche se c'è un solo elemento):
[
  {
    "type": "revenue|partner_count|funding|team_size|other",
    "text": "testo esatto come appare nella fonte",
    "source_location": "dove si trova nel documento (es. slide 4, homepage, sezione Chi Siamo)",
    "specificity": "high|medium|low",
    "verifiable": true|false,
    "extraction_confidence": 0.0-1.0,
    "normalized_value": "valore numerico pulito se applicabile (es. 3800000 per €3.8M, 320 per 320 strutture)",
    "notes": "note opzionali sull'interpretazione"
  }
]

Se non trovi claim verificabili nel testo, rispondi con un array vuoto: []
Rispondi SOLO con JSON valido. Nessun testo prima o dopo.
"""

FINANCIAL_EXTRACTION_SYSTEM = """Sei un analista contabile specializzato nell'interpretazione di bilanci italiani.
Stai analizzando un bilancio scaricato dal portale Infocamere o documento simile.

Estrai i seguenti dati se presenti:
- exercise_year: anno di chiusura dell'esercizio (intero). Cerca "esercizio", "al 31/12/YYYY", "anno YYYY"
- revenues: ricavi totali in euro (numero intero, senza simboli, senza punti separatori).
  Cerca NELL'ORDINE: "Ricavi delle vendite e delle prestazioni", "A) Valore della produzione",
  "Totale valore della produzione", "Ricavi netti", "Fatturato", "Valore della produzione",
  "Proventi". Prendi il PRIMO valore trovato. Se trovi "869.619" scrivi 869619.
- total_assets: totale attivo in euro
- employees: numero dipendenti (intero). Cerca "dipendenti", "addetti", "n. medio"
- legal_form: forma giuridica (es. "S.r.l.", "S.p.A.")
- share_capital: capitale sociale in euro
- extraction_confidence: 0.0-1.0 (alta se hai trovato il dato direttamente, bassa se stimato)
- raw_excerpt: citazione testuale di max 200 caratteri con i ricavi trovati

ATTENZIONE: i numeri nei bilanci italiani usano il punto come separatore delle migliaia
(es. 1.234.567 = un milione duecentotrentaquattromila). Converti sempre in intero senza punti.

FORMATO GESTIONALE (due colonne separate da |):
Se il testo ha formato "COSTI Eur|RICAVI Eur" con righe tipo:
  "801 RICAVI ALLA LET.A)E B)ART.85______A1 649,04"
  "803 RICAVI PER PRESTAZIONI SERVIZI____A1 5.500,00"
  "809 ALTRI PROVENTI CONSIDERATI RICAVI 50.000,00"
Somma TUTTI i valori delle righe ricavi (801, 803, 809 etc.) per ottenere il totale.

Rispondi SOLO con un oggetto JSON valido con queste chiavi esatte.
Se un dato non è presente nel testo, usa null.
Nessun testo prima o dopo il JSON.
"""

DEDUP_SYSTEM = """Sei un analista che deve deduplicare un elenco di claim aziendali.
Alcune claim potrebbero essere la stessa affermazione formulata in modi diversi.

Ricevi un array JSON di claim. Per ognuna indica se è un duplicato di un'altra.
Rispondi con un array JSON dove ogni elemento ha:
{
  "id": "id originale della claim",
  "is_duplicate": true|false,
  "duplicate_of": "id della claim principale (solo se is_duplicate è true)"
}

Rispondi SOLO con JSON valido.
"""


# ─────────────────────────────────────────────
#  Claim Extractor principale
# ─────────────────────────────────────────────

class ClaimExtractor:
    """
    Pipeline principale del Modulo 1.

    Utilizzo base:
        llm = LLMAdapter(provider="groq", api_key="gsk_...", model="llama-3.1-70b-versatile")
        extractor = ClaimExtractor(llm)

        result = extractor.extract(
            company_name="MoveNow S.r.l.",
            declarative_sources=[
                {"type": "pdf",  "path": "pitch_deck.pdf"},
                {"type": "url",  "url": "https://movenow.it"},
                {"type": "text", "text": "Testo estratto da email...", "label": "email_commerciale"},
            ],
            probatory_sources=[
                {"type": "pdf", "path": "bilancio_2023.pdf", "label": "bilancio_infocamere"},
            ]
        )

        print(result.summary())
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    """

    def __init__(self, llm: LLMAdapter):
        self.llm      = llm
        self.ingester = DocumentIngester()
        self.chunker  = SemanticChunker()

    def extract(
        self,
        company_name: str,
        declarative_sources: list[dict],
        probatory_sources: list[dict] | None = None,
    ) -> ExtractionResult:

        result = ExtractionResult(company_name=company_name)
        all_raw_claims: list[dict] = []

        # ── 1. Processa documenti dichiarativi ────────────────────────────
        for source in declarative_sources:
            try:
                text, doc_label = self._ingest_source(source)
                chunks = self.chunker.chunk(text)
                log.info(f"[{doc_label}] {len(chunks)} chunk da processare")

                for chunk in chunks:
                    raw = self._extract_claims_from_chunk(
                        chunk["text"],
                        source_document=doc_label,
                        source_location=chunk["label"],
                    )
                    all_raw_claims.extend(raw)

            except Exception as e:
                result.errors.append(f"Errore su {source}: {e}")
                log.error(f"Errore ingestion: {e}")

        # ── 2. Deduplicazione ─────────────────────────────────────────────
        if len(all_raw_claims) > 1:
            all_raw_claims = self._deduplicate(all_raw_claims)

        # ── 3. Converti in oggetti Claim tipizzati ────────────────────────
        for i, raw in enumerate(all_raw_claims):
            try:
                claim = self._build_claim(raw, index=i)
                result.claims.append(claim)
            except Exception as e:
                result.warnings.append(f"Claim ignorata per errore di parsing: {e}")

        # ── 3b. Estrai persone chiave dal testo dichiarativo ─────────────
        # Usa tutto il testo concatenato dei documenti dichiarativi
        all_declarative_text = ""
        for source in declarative_sources:
            try:
                text, _ = self._ingest_source(source)
                all_declarative_text += "\n\n" + text
            except Exception:
                pass
        if all_declarative_text.strip():
            result.key_people   = self._extract_people(all_declarative_text)
            result.tone_analysis = self._analyze_tone(all_declarative_text)

        # ── 4. Processa documenti probatori (bilancio) ────────────────────
        if probatory_sources:
            for source in probatory_sources:
                try:
                    text, doc_label = self._ingest_source(source)
                    fin = self._extract_financial_data(text, doc_label)
                    if fin:
                        result.financial_data = fin
                        log.info(f"[{doc_label}] Dati finanziari estratti: ricavi={fin.revenues}")
                except Exception as e:
                    result.errors.append(f"Errore bilancio {source}: {e}")

        # ── Claim sintetica dal bilancio se nessuna claim revenue trovata ────
        if result.financial_data and result.financial_data.revenues:
            has_revenue = any(
                (c.type.value if hasattr(c.type, "value") else c.type) == "revenue"
                for c in result.claims
            )
            if not has_revenue:
                rev  = result.financial_data.revenues
                year = result.financial_data.exercise_year or "N/D"
                try:
                    synthetic = self._build_claim({
                        "type":                  "revenue",
                        "text":                  f"Ricavi esercizio {year}: €{int(rev):,} (da bilancio depositato)",
                        "normalized_value":      str(int(rev)),
                        "specificity":           "high",
                        "verifiable":            True,
                        "extraction_confidence": result.financial_data.extraction_confidence,
                        "source_document":       "bilancio_infocamere",
                        "source_location":       "bilancio",
                        "notes":                 "Generata automaticamente dai dati del bilancio",
                    }, index=len(result.claims))
                    result.claims.append(synthetic)
                    log.info(f"Claim revenue sintetica aggiunta: €{int(rev):,}")
                except Exception as e:
                    log.warning(f"Claim sintetica non creata: {e}")

        log.info(f"Estrazione completata: {result.summary()}")
        return result

    # ── Metodi privati ─────────────────────────────────────────────────────

    def _ingest_source(self, source: dict) -> tuple[str, str]:
        """Legge la sorgente e restituisce (testo, label)."""
        stype = source.get("type", "text")

        if stype == "pdf":
            path = source["path"]
            label = source.get("label", Path(path).stem)
            text, method = DocumentIngester.from_pdf(path)
            log.info(f"PDF letto con {method}: {len(text)} caratteri")
            return text, label

        elif stype == "url":
            url = source["url"]
            label = source.get("label", url.split("//")[-1].split("/")[0])
            text = DocumentIngester.from_url(url)
            return text, label

        elif stype == "text":
            label = source.get("label", "testo_manuale")
            text = DocumentIngester.from_text(source["text"])
            return text, label

        else:
            raise ValueError(f"Tipo sorgente non supportato: {stype}")

    def _extract_claims_from_chunk(
        self,
        text: str,
        source_document: str,
        source_location: str,
    ) -> list[dict]:
        """Chiama l'LLM su un singolo chunk e restituisce claim grezze."""
        user_prompt = (
            f"Documento: {source_document}\n"
            f"Posizione: {source_location}\n\n"
            f"--- TESTO ---\n{text}\n--- FINE TESTO ---\n\n"
            "Estrai tutte le claim verificabili."
        )

        try:
            response = self.llm.complete(CLAIM_EXTRACTION_SYSTEM, user_prompt)
            claims = self._parse_json_response(response)
            # Aggiungi metadati sorgente
            for c in claims:
                c["source_document"] = source_document
                if not c.get("source_location"):
                    c["source_location"] = source_location
            return claims

        except Exception as e:
            log.warning(f"LLM fallita su chunk [{source_location}]: {e}")
            return []

    def _analyze_tone(self, text: str) -> dict:
        """
        Analizza il tono e il linguaggio del pitch deck con Mistral.
        Rileva: linguaggio inflazionato, numeri senza contesto,
        vaghezza strategica, claim assolute.
        """
        try:
            user_prompt = (
                "--- TESTO PITCH DECK ---\n"
                + text[:5000]
                + "\n--- FINE ---\n\nAnalizza il tono e il linguaggio."
            )
            response = self.llm.complete(TONE_ANALYSIS_SYSTEM, user_prompt)
            flags = self._parse_json_response(response)
            if not isinstance(flags, list):
                flags = []

            # Filtra e valida
            valid_flags = []
            for f in flags:
                if not f.get("phrase") or not f.get("category"):
                    continue
                valid_flags.append({
                    "category":    f.get("category", "other"),
                    "phrase":      f.get("phrase", "")[:80],
                    "explanation": f.get("explanation", "")[:120],
                    "severity":    f.get("severity", "medium"),
                    "location":    f.get("location", "")[:60],
                })

            # Calcola score tono 0-10 (10 = tono sobrio/credibile)
            high   = sum(1 for f in valid_flags if f["severity"] == "high")
            medium = sum(1 for f in valid_flags if f["severity"] == "medium")
            low    = sum(1 for f in valid_flags if f["severity"] == "low")
            penalty = high * 1.5 + medium * 0.8 + low * 0.3
            tone_score = round(max(1.0, min(10.0, 10.0 - penalty)), 1)

            # Label
            if tone_score >= 8:   tone_label = "Tono sobrio e credibile"
            elif tone_score >= 6: tone_label = "Qualche affermazione da verificare"
            elif tone_score >= 4: tone_label = "Linguaggio spesso inflazionato"
            else:                 tone_label = "Linguaggio altamente promozionale"

            # Categorie trovate
            cats = list({f["category"] for f in valid_flags})
            cat_labels = {
                "inflated_language":      "Linguaggio inflazionato",
                "number_without_context": "Numeri senza contesto",
                "strategic_vagueness":    "Vaghezza strategica",
                "absolute_claim":         "Affermazioni assolute",
            }

            log.info(
                f"ToneAnalysis: {len(valid_flags)} flag ({high} high, {medium} medium, {low} low) "                f"— Tone Score: {tone_score}"
            )

            return {
                "flags":       valid_flags,
                "total":       len(valid_flags),
                "high_count":  high,
                "med_count":   medium,
                "low_count":   low,
                "tone_score":  tone_score,
                "tone_label":  tone_label,
                "categories":  [cat_labels.get(c, c) for c in cats],
            }

        except Exception as e:
            log.warning(f"Tone analysis error: {e}")
            return {"flags": [], "total": 0, "tone_score": None, "tone_label": ""}

    def _extract_people(self, text: str) -> list[dict]:
        """Estrae persone chiave dal testo del pitch deck usando l'LLM."""
        try:
            user_prompt = (
                "--- TESTO ---\n"
                + text[:4000]
                + "\n--- FINE ---\n\nEstrai tutte le persone chiave dell'azienda."
            )
            response = self.llm.complete(PEOPLE_EXTRACTION_SYSTEM, user_prompt)
            people = self._parse_json_response(response)
            if not isinstance(people, list):
                return []
            # Filtra: richiedi almeno nome + cognome
            valid = []
            for p in people:
                name = p.get("name","").strip()
                role = p.get("role","").strip()
                if name and role and len(name.split()) >= 2:
                    valid.append({
                        "name":          name,
                        "role":          role,
                        "role_category": p.get("role_category","other"),
                        "confidence":    float(p.get("confidence",0.7)),
                        "source":        "pitch_deck_llm",
                        "linkedin":      "",
                        "url":           "",
                    })
            log.info(f"PeopleExtraction: {len(valid)} persone trovate nel pitch deck")
            return valid
        except Exception as e:
            log.warning(f"Errore estrazione persone: {e}")
            return []

    def _extract_financial_data(self, text: str, doc_label: str) -> FinancialData | None:
        """
        Estrae dati strutturati da un bilancio.
        Strategia doppia: prima regex veloce, poi LLM come fallback/validazione.
        """
        # ── Step 1: regex veloce per valori più comuni ────────────────────
        regex_data = self._extract_financials_regex(text)

        # ── Step 2: LLM su sezione rilevante del testo ───────────────────
        # Trova la sezione del conto economico se presente
        eco_markers = ["conto economico", "ricavi", "valore della produzione",
                       "proventi", "fatturato", "a) ricavi"]
        text_lower = text.lower()
        start_idx = 0
        for marker in eco_markers:
            idx = text_lower.find(marker)
            if idx > 0:
                start_idx = max(0, idx - 200)
                break

        # Usa la sezione rilevante + inizio documento
        relevant = text[:2000] + "\n\n...\n\n" + text[start_idx:start_idx+4000]
        truncated = relevant[:6000]

        user_prompt = (
            f"Documento: {doc_label}\n\n"
            f"--- TESTO BILANCIO ---\n{truncated}\n--- FINE ---\n\n"
            "Estrai i dati finanziari richiesti. "
            "Se vedi numeri come '869.619' in un bilancio italiano, "
            "significa 869619 euro (il punto è separatore migliaia)."
        )

        try:
            response = self.llm.complete(FINANCIAL_EXTRACTION_SYSTEM, user_prompt)
            data = self._parse_json_response(response)
            if isinstance(data, list):
                data = data[0] if data else {}

            # Usa regex come fallback se LLM non trova i ricavi
            revenues = self._to_float(data.get("revenues"))
            if revenues is None and regex_data.get("revenues"):
                revenues = regex_data["revenues"]
                log.info(f"Ricavi trovati via regex: {revenues}")

            employees = self._to_int(data.get("employees"))
            if employees is None and regex_data.get("employees"):
                employees = regex_data["employees"]

            return FinancialData(
                source_document=doc_label,
                exercise_year=data.get("exercise_year") or regex_data.get("year"),
                revenues=revenues,
                total_assets=self._to_float(data.get("total_assets")),
                employees=employees,
                legal_form=data.get("legal_form"),
                share_capital=self._to_float(data.get("share_capital")),
                extraction_confidence=float(data.get("extraction_confidence", 0.5)),
                raw_excerpt=data.get("raw_excerpt", ""),
            )
        except Exception as e:
            log.error(f"Errore estrazione dati finanziari: {e}")
            # Restituisci almeno i dati da regex se disponibili
            if regex_data:
                return FinancialData(
                    source_document=doc_label,
                    revenues=regex_data.get("revenues"),
                    employees=regex_data.get("employees"),
                    exercise_year=regex_data.get("year"),
                    extraction_confidence=0.4,
                    raw_excerpt="Estratto via regex",
                )
            return None

    @staticmethod
    def _extract_financials_regex(text: str) -> dict:
        """
        Estrazione veloce via regex come backup all'LLM.
        Gestisce il formato numerico italiano (punto=migliaia, virgola=decimali).
        """
        result = {}
        text_lower = text.lower()

        def parse_it_number(s: str) -> Optional[float]:
            """Converte '1.234.567' o '1.234.567,89' in float."""
            s = s.strip().replace(" ", "")
            # Rimuovi simbolo euro
            s = s.replace("€", "").strip()
            # Formato italiano: punto=migliaia, virgola=decimali
            if "," in s:
                parts = s.rsplit(",", 1)
                integer_part = parts[0].replace(".", "")
                decimal_part = parts[1] if len(parts) > 1 else "0"
                s = integer_part + "." + decimal_part
            else:
                # Solo punti = separatori migliaia
                s = s.replace(".", "")
            try:
                return float(s)
            except Exception:
                return None

        # Pattern ricavi (conto economico italiano)
        revenue_patterns = [
            r"(?:a\)|ricavi delle vendite[^€\d]*)([\d\.]+(?:,\d+)?)",
            r"(?:valore della produzione)[^\d]*([\d\.]+(?:,\d+)?)",
            r"(?:totale valore della produzione)[^\d]*([\d\.]+(?:,\d+)?)",
            r"(?:ricavi netti|fatturato)[^\d]*([\d\.]+(?:,\d+)?)",
            r"(?:proventi totali)[^\d]*([\d\.]+(?:,\d+)?)",
        ]

        # Pattern specifici per formato gestionale (due colonne separate da |)
        # Es: "801 RICAVI ALLA LET.A)E B)ART.85______A1 649,04"
        # Es: "803 RICAVI PER PRESTAZIONI SERVIZI____A1 5.500,00"
        gestionale_patterns = [
            r"8\d{2}\s+RICAVI[^\|]*?([\d\.]+,\d{2})",
            r"ricavi per prestazioni[^\|]*?([\d\.]+,\d{2})",
            r"ricavi alla let[^\|]*?([\d\.]+,\d{2})",
            r"altri proventi considerati ricavi[^\|]*?([\d\.]+,\d{2})",
        ]

        # Prova i pattern standard
        for pat in revenue_patterns:
            m = re.search(pat, text_lower)
            if m:
                val = parse_it_number(m.group(1))
                if val and val > 1000:
                    result["revenues"] = val
                    break

        # Se non trovato, prova formato gestionale e somma tutte le voci ricavi
        if not result.get("revenues"):
            total_revenues = 0.0
            for pat in gestionale_patterns:
                for m in re.finditer(pat, text_lower):
                    val = parse_it_number(m.group(1))
                    if val and val > 0:
                        total_revenues += val
            if total_revenues > 0:
                result["revenues"] = total_revenues

        # Anno esercizio
        m = re.search(r"(?:esercizio|al 31/12/|anno)\s*(20\d{2})", text_lower)
        if m:
            result["year"] = int(m.group(1))

        # Dipendenti
        m = re.search(r"(?:dipendenti|addetti|n\.\s*medio)[^\d]*(\d+)", text_lower)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 100000:
                result["employees"] = val

        return result

    def _deduplicate(self, claims: list[dict]) -> list[dict]:
        """Rimuove duplicati semantici usando l'LLM."""
        # Prepara input minimale per il LLM
        compact = [
            {"id": f"C{i:03d}", "text": c.get("text", ""), "type": c.get("type", "")}
            for i, c in enumerate(claims)
        ]

        user_prompt = f"Claim da deduplicare:\n{json.dumps(compact, ensure_ascii=False)}"

        try:
            response = self.llm.complete(DEDUP_SYSTEM, user_prompt)
            dedup_result = self._parse_json_response(response)

            # Filtra i duplicati
            to_remove = {
                int(d["id"][1:])
                for d in dedup_result
                if d.get("is_duplicate")
            }

            filtered = [c for i, c in enumerate(claims) if i not in to_remove]
            removed = len(claims) - len(filtered)
            if removed:
                log.info(f"Deduplicazione: rimossi {removed} duplicati")
            return filtered

        except Exception as e:
            log.warning(f"Deduplicazione fallita, procedo senza: {e}")
            return claims

    def _build_claim(self, raw: dict, index: int) -> Claim:
        """Costruisce un oggetto Claim tipizzato da un dict grezzo."""
        claim_id = self._make_id(raw.get("text", ""), index)

        return Claim(
            id=claim_id,
            type=ClaimType(raw.get("type", "other")),
            text=raw.get("text", ""),
            source_document=raw.get("source_document", ""),
            source_location=raw.get("source_location", ""),
            specificity=Specificity(raw.get("specificity", "medium")),
            verifiable=bool(raw.get("verifiable", True)),
            extraction_confidence=float(raw.get("extraction_confidence", 0.5)),
            normalized_value=raw.get("normalized_value"),
            notes=raw.get("notes", ""),
        )

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json_response(text: str) -> list | dict:
        """Pulisce e parsa la risposta JSON dell'LLM."""
        # Rimuovi markdown code fences se presenti
        clean = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("```").strip()

        # Se l'LLM ha aggiunto testo prima/dopo il JSON, estrai solo il JSON
        match = re.search(r"(\[.*\]|\{.*\})", clean, re.DOTALL)
        if match:
            clean = match.group(1)

        return json.loads(clean)

    @staticmethod
    def _make_id(text: str, index: int) -> str:
        h = hashlib.md5(text.encode()).hexdigest()[:6]
        return f"C{index:03d}_{h}"

    @staticmethod
    def _to_float(val) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(str(val).replace(",", ".").replace(" ", ""))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _to_int(val) -> Optional[int]:
        if val is None:
            return None
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None
