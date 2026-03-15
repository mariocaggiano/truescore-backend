"""
TrueScore — Verification Engine
=================================
Incrocia le claim estratte con i dati raccolti e produce un verdict
per ognuna, con confidence scoring e Trust Score aggregato.

Verdict possibili (5 livelli):
  verified          — claim confermata dalle fonti
  warning           — scarto contenuto o dati parziali
  discrepancy       — scarto significativo con buona evidenza
  uncertain         — dati insufficienti per giudicare
  insufficient_data — nessuna fonte disponibile

Dipendenze: solo stdlib Python
"""

import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum

log = logging.getLogger("verification_engine")


# ─────────────────────────────────────────────
#  Tipi di dato
# ─────────────────────────────────────────────

class Verdict(str, Enum):
    VERIFIED          = "verified"
    WARNING           = "warning"
    DISCREPANCY       = "discrepancy"
    UNCERTAIN         = "uncertain"
    INSUFFICIENT_DATA = "insufficient_data"


VERDICT_SCORE = {
    Verdict.VERIFIED:          1.0,
    Verdict.WARNING:           0.6,
    Verdict.UNCERTAIN:         0.4,
    Verdict.DISCREPANCY:       0.1,
    Verdict.INSUFFICIENT_DATA: 0.3,
}

VERDICT_LABELS = {
    Verdict.VERIFIED:          "Verificata",
    Verdict.WARNING:           "Attenzione",
    Verdict.DISCREPANCY:       "Discrepanza",
    Verdict.UNCERTAIN:         "Incerta",
    Verdict.INSUFFICIENT_DATA: "Dati insufficienti",
}

# Pesi per il Trust Score aggregato
CLAIM_TYPE_WEIGHTS = {
    "revenue":       0.35,
    "partner_count": 0.30,
    "funding":       0.25,
    "team_size":     0.10,
    "other":         0.05,
}


@dataclass
class ClaimVerdict:
    claim_id: str
    claim_type: str
    claim_text: str
    declared_value: Optional[float]
    verified_value: Optional[float]
    verdict: Verdict
    evidence_confidence: float       # 0–1: quanto sono affidabili i dati raccolti
    magnitude: float                 # 0–1: entità dello scarto dichiarato/verificato
    reasoning: str                   # spiegazione in linguaggio naturale
    sources_used: list[str]          = field(default_factory=list)
    sources_consulted: list[str]     = field(default_factory=list)  # tutte le fonti tentate, anche senza risultati
    flags: list[str]                 = field(default_factory=list)
    notes: str                       = ""

    @property
    def verdict_score(self) -> float:
        return VERDICT_SCORE[self.verdict]

    @property
    def verdict_label(self) -> str:
        return VERDICT_LABELS[self.verdict]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["verdict_label"] = self.verdict_label
        d["verdict_score"] = self.verdict_score
        return d


@dataclass
class VerificationResult:
    company_name: str
    verdicts: list[ClaimVerdict]     = field(default_factory=list)
    trust_score: float               = 0.0   # 0–10
    trust_score_label: str           = ""
    red_flags: list[ClaimVerdict]    = field(default_factory=list)
    warnings_list: list[ClaimVerdict]= field(default_factory=list)
    unverifiable: list[ClaimVerdict] = field(default_factory=list)
    errors: list[str]                = field(default_factory=list)
    legal_status: Optional[dict]     = None   # dati OpenCorporates

    def summary(self) -> str:
        lines = [
            f"  Azienda       : {self.company_name}",
            f"  Trust Score   : {self.trust_score:.1f}/10 ({self.trust_score_label})",
            f"  Verdicts      : {len(self.verdicts)} totali",
            f"  Discrepanze   : {len(self.red_flags)}",
            f"  Attenzione    : {len(self.warnings_list)}",
            f"  Non verificab.: {len(self.unverifiable)}",
        ]
        for v in self.verdicts:
            bar = "█" * round(v.evidence_confidence * 5)
            bar += "░" * (5 - round(v.evidence_confidence * 5))
            lines.append(
                f"    [{v.claim_type:<15}] {v.verdict_label:<20} "
                f"conf={bar} ({v.evidence_confidence:.0%})"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "company_name": self.company_name,
            "trust_score": self.trust_score,
            "trust_score_label": self.trust_score_label,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "red_flags": [v.claim_id for v in self.red_flags],
            "warnings": [v.claim_id for v in self.warnings_list],
            "unverifiable": [v.claim_id for v in self.unverifiable],
            "legal_status": self.legal_status,
            "errors": self.errors,
        }


# ─────────────────────────────────────────────
#  Verifier per tipo di claim
# ─────────────────────────────────────────────

class BaseVerifier:
    """Interfaccia comune per i verifier specializzati per tipo."""

    CLAIM_TYPE = "base"

    def verify(self, claim: dict, collector_results: list[dict]) -> ClaimVerdict:
        raise NotImplementedError

    @staticmethod
    def _compute_verdict(magnitude: float, confidence: float) -> Verdict:
        """
        Matrice verdict basata su magnitude (scarto) e confidence (affidabilità dati).

        magnitude: percentuale di scarto normalizzata (0 = identico, 1 = totalmente diverso)
        confidence: affidabilità delle fonti (0 = nulla, 1 = massima)
        """
        if confidence < 0.35:
            return Verdict.INSUFFICIENT_DATA
        if magnitude < 0.20:
            return Verdict.VERIFIED
        if magnitude < 0.50 and confidence < 0.65:
            return Verdict.UNCERTAIN
        if magnitude >= 0.50 and confidence >= 0.60:
            return Verdict.DISCREPANCY
        if magnitude >= 0.20:
            return Verdict.WARNING
        return Verdict.UNCERTAIN

    @staticmethod
    def _magnitude(declared: float, verified: float) -> float:
        """Calcola lo scarto relativo normalizzato tra dichiarato e verificato."""
        if declared == 0 and verified == 0:
            return 0.0
        if declared == 0:
            return 1.0
        raw = abs(declared - verified) / max(abs(declared), abs(verified))
        return min(raw, 1.0)

    @staticmethod
    def _extract_connector(results: list[dict], connector_name: str) -> Optional[dict]:
        for r in results:
            if r.get("connector") == connector_name and r.get("found"):
                return r
        return None

    @staticmethod
    def _to_float(val) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(str(val).replace(",", ".").replace(" ", "").replace("€", ""))
        except (ValueError, TypeError):
            return None


class RevenueVerifier(BaseVerifier):
    """
    Verifica claim di tipo 'revenue'.
    Fonte primaria: bilancio caricato dall'utente (confidence 0.93).
    Fonte secondaria: proxy da headcount (confidence 0.45).
    """

    CLAIM_TYPE = "revenue"

    # Benchmark revenue/dipendente per settore (€)
    SECTOR_BENCHMARKS = {
        "mobilità":       80_000,
        "mobility":       80_000,
        "saas":          150_000,
        "software":      130_000,
        "logistica":      90_000,
        "default":        85_000,
    }

    def verify(self, claim: dict, collector_results: list[dict]) -> ClaimVerdict:
        claim_id   = claim.get("id", "")
        claim_text = claim.get("text", "")
        declared   = self._to_float(claim.get("normalized_value"))
        sources    = []
        flags      = []

        # ── Scenario A: bilancio disponibile ─────────────────────────────
        bilancio = self._extract_connector(collector_results, "bilancio")
        if bilancio:
            sources.append("bilancio_infocamere")
            verified_revenues = self._to_float(bilancio["data"].get("revenues"))
            confidence        = float(bilancio.get("confidence", 0.90))
            exercise_year     = bilancio["data"].get("exercise_year")

            if verified_revenues is None:
                return ClaimVerdict(
                    claim_id=claim_id, claim_type=self.CLAIM_TYPE,
                    claim_text=claim_text,
                    declared_value=declared, verified_value=None,
                    verdict=Verdict.INSUFFICIENT_DATA,
                    evidence_confidence=0.3, magnitude=0.0,
                    reasoning="Bilancio caricato ma ricavi non presenti (possibile bilancio abbreviato).",
                    sources_used=sources,
                )

            if declared is None:
                return ClaimVerdict(
                    claim_id=claim_id, claim_type=self.CLAIM_TYPE,
                    claim_text=claim_text,
                    declared_value=None, verified_value=verified_revenues,
                    verdict=Verdict.UNCERTAIN,
                    evidence_confidence=confidence, magnitude=0.0,
                    reasoning="Valore dichiarato non quantificabile con precisione. "
                              f"Bilancio riporta ricavi pari a €{verified_revenues:,.0f} "
                              f"(esercizio {exercise_year}).",
                    sources_used=sources,
                )

            magnitude = self._magnitude(declared, verified_revenues)
            verdict   = self._compute_verdict(magnitude, confidence)

            # Flag: gap temporale (bilancio vecchio vs claim recente)
            if exercise_year and (2026 - exercise_year) >= 2:
                flags.append(f"Gap temporale: bilancio {exercise_year}, claim potenzialmente più recente")
                # Ammorbidisci il verdict se gap > 1 anno (crescita legittima)
                if verdict == Verdict.DISCREPANCY and magnitude < 0.80:
                    verdict = Verdict.WARNING
                    flags.append("Verdict attenuato per gap temporale significativo")

            reasoning = self._build_revenue_reasoning(
                declared, verified_revenues, magnitude, verdict, exercise_year
            )

            return ClaimVerdict(
                claim_id=claim_id, claim_type=self.CLAIM_TYPE,
                claim_text=claim_text,
                declared_value=declared, verified_value=verified_revenues,
                verdict=verdict,
                evidence_confidence=confidence, magnitude=magnitude,
                reasoning=reasoning, sources_used=sources, flags=flags,
            )

        # ── Scenario B: nessun bilancio — usa proxy headcount ─────────────
        linkedin = self._extract_connector(collector_results, "linkedin")
        if linkedin:
            sources.append("linkedin_headcount")
            headcount = self._to_float(
                linkedin["data"].get("headcount_midpoint") or
                linkedin["data"].get("headcount_range", "").split("-")[0]
            )
            if headcount:
                sector   = claim.get("sector", "default").lower()
                bench    = self._sector_benchmark(sector)
                proxy    = headcount * bench
                confidence = 0.45   # proxy, non dato diretto

                if declared:
                    magnitude = self._magnitude(declared, proxy)
                    verdict   = self._compute_verdict(magnitude, confidence)
                    reasoning = (
                        f"Bilancio non disponibile. Proxy calcolato: "
                        f"{headcount:.0f} dipendenti × €{bench:,.0f}/dip. (benchmark settore) "
                        f"= €{proxy:,.0f}. "
                        f"Dichiarato: €{declared:,.0f}. "
                        f"Scarto: {magnitude:.0%}. Fonte proxy: LinkedIn headcount."
                    )
                    flags.append("Stima proxy — non sostituisce bilancio ufficiale")
                    return ClaimVerdict(
                        claim_id=claim_id, claim_type=self.CLAIM_TYPE,
                        claim_text=claim_text,
                        declared_value=declared, verified_value=proxy,
                        verdict=verdict,
                        evidence_confidence=confidence, magnitude=magnitude,
                        reasoning=reasoning, sources_used=sources, flags=flags,
                    )

        # ── Scenario C: UfficioCamerale (da P.IVA) ───────────────────────
        uc = self._extract_connector(collector_results, "ufficiocamerale")
        if uc:
            sources.append("ufficiocamerale")
            uc_revenues  = self._to_float(uc["data"].get("revenues"))
            uc_employees = self._to_float(uc["data"].get("employees"))
            confidence   = 0.75   # dati pubblici Registro Imprese, ma possono essere datati

            if uc_revenues and declared:
                magnitude = self._magnitude(declared, uc_revenues)
                verdict   = self._compute_verdict(magnitude, confidence)
                reasoning = (
                    f"Fonte: ufficiocamerale.it (Registro Imprese). "
                    f"Fatturato dichiarato nel registro: €{uc_revenues:,.0f}. "
                    f"Dichiarato: €{declared:,.0f}. "
                    f"Scarto: {magnitude:.0%}. "
                    f"Nota: i dati del Registro Imprese possono riferirsi all'ultimo bilancio depositato."
                )
                return ClaimVerdict(
                    claim_id=claim_id, claim_type=self.CLAIM_TYPE,
                    claim_text=claim_text,
                    declared_value=declared, verified_value=uc_revenues,
                    verdict=verdict,
                    evidence_confidence=confidence, magnitude=magnitude,
                    reasoning=reasoning, sources_used=sources,
                    flags=["fonte_registro_imprese_pubblico"],
                )

            if uc_employees and declared is None:
                # Almeno restituisci i dipendenti come contesto
                return ClaimVerdict(
                    claim_id=claim_id, claim_type=self.CLAIM_TYPE,
                    claim_text=claim_text,
                    declared_value=declared, verified_value=None,
                    verdict=Verdict.UNCERTAIN,
                    evidence_confidence=0.50, magnitude=0.0,
                    reasoning=f"Fatturato non trovato nel Registro Imprese. "
                              f"Dipendenti registrati: {int(uc_employees)}.",
                    sources_used=sources,
                )

        # ── Nessuna fonte disponibile ─────────────────────────────────────
        consulted = ["bilancio_infocamere", "ufficiocamerale", "linkedin_headcount"]
        return ClaimVerdict(
            claim_id=claim_id, claim_type=self.CLAIM_TYPE,
            claim_text=claim_text,
            declared_value=declared, verified_value=None,
            verdict=Verdict.INSUFFICIENT_DATA,
            evidence_confidence=0.0, magnitude=0.0,
            reasoning="Nessuna fonte ha restituito dati sui ricavi. "
                      "Per una verifica diretta caricare il bilancio Infocamere o inserire la Partita IVA.",
            sources_used=[], sources_consulted=consulted,
            flags=["bilancio_mancante"],
        )

    def _sector_benchmark(self, sector: str) -> float:
        for key, val in self.SECTOR_BENCHMARKS.items():
            if key in sector:
                return val
        return self.SECTOR_BENCHMARKS["default"]

    @staticmethod
    def _build_revenue_reasoning(
        declared: float, verified: float, magnitude: float,
        verdict: Verdict, exercise_year: Optional[int]
    ) -> str:
        year_str = f"esercizio {exercise_year}" if exercise_year else "esercizio n/d"
        if verdict == Verdict.VERIFIED:
            return (
                f"I ricavi dichiarati (€{declared:,.0f}) sono coerenti con il bilancio "
                f"depositato ({year_str}: €{verified:,.0f}). "
                f"Scarto: {magnitude:.0%}."
            )
        if verdict == Verdict.WARNING:
            return (
                f"I ricavi dichiarati (€{declared:,.0f}) presentano uno scarto del {magnitude:.0%} "
                f"rispetto al bilancio {year_str} (€{verified:,.0f}). "
                f"Possibile crescita nel periodo o differenza di perimetro contabile."
            )
        if verdict == Verdict.DISCREPANCY:
            return (
                f"Discrepanza significativa. Dichiarato: €{declared:,.0f}. "
                f"Bilancio {year_str}: €{verified:,.0f}. "
                f"Scarto: {magnitude:.0%} ({declared/verified:.1f}x il dato ufficiale)."
            )
        return f"Bilancio {year_str}: €{verified:,.0f}. Dichiarato: €{declared:,.0f}."


class PartnerCountVerifier(BaseVerifier):
    """
    Verifica claim di tipo 'partner_count'.

    Logica multi-segnale (in ordine di affidabilità):

    1. Pagina partner sul sito aziendale → conta nomi/loghi reali
    2. Menzioni Google da domini terzi → conta conferme pubbliche indipendenti
    3. Wayback Machine → confronto storico del numero dichiarato

    Verdict:
    - Se troviamo partner_found >= dichiarato * 0.7 → VERIFIED
    - Se troviamo partner_found tra 30% e 70% del dichiarato → WARNING
    - Se troviamo partner_found < 30% del dichiarato → DISCREPANCY
    - Se non troviamo nulla di verificabile → INSUFFICIENT_DATA

    Importante: partner_found è un lower bound (non tutte le partnership
    sono pubbliche), quindi usiamo tolleranze generose verso il basso.
    """

    CLAIM_TYPE = "partner_count"

    # Tolleranza: fino al 40% in meno è accettabile
    # (non tutti i partner appaiono sul sito o su Google)
    TOLERANCE = 0.40

    def verify(self, claim: dict, collector_results: list[dict]) -> ClaimVerdict:
        claim_id   = claim.get("id", "")
        claim_text = claim.get("text", "")
        declared   = self._to_float(claim.get("normalized_value"))
        sources    = []
        flags      = []

        partner_data = self._extract_connector(collector_results, "partner_website")
        wayback      = self._extract_connector(collector_results, "wayback")

        # ── Segnale principale: partner_website ───────────────────────────
        if partner_data and partner_data.get("found"):
            data           = partner_data.get("data", {})
            partners_found = self._to_float(data.get("partners_found", 0)) or 0
            confidence     = float(partner_data.get("confidence", 0.6))
            source_type    = data.get("source", "partner_website")
            source_url     = partner_data.get("source_url", "")
            partner_names  = data.get("partner_names", [])
            evidence_urls  = data.get("evidence_urls", [])
            signals        = data.get("signals_found", [])

            # Aggiungi fonti usate
            if "website_partner_page" in signals:
                sources.append("website_partner_page")
            if "google_partnership_mentions" in signals:
                sources.append("google_partnership_search")

            # ── Cross-reference Wayback ───────────────────────────────────
            wayback_flags = self._check_wayback_consistency(wayback, declared)
            flags.extend(wayback_flags)
            if wayback_flags:
                sources.append("wayback_machine")

            if declared is not None and partners_found > 0:
                # partners_found è lower bound → confrontiamo con tolleranza
                effective_declared = declared * (1 - self.TOLERANCE)
                magnitude = max(0.0, (effective_declared - partners_found) / declared)
                verdict   = self._compute_verdict(magnitude, confidence)

                # Costruisci reasoning dettagliato
                reasoning_parts = []

                if "website_partner_page" in signals:
                    reasoning_parts.append(
                        f"Pagina partner sul sito aziendale: "
                        f"{int(partners_found)} partner identificati"
                        + (f" (es. {', '.join(partner_names[:3])})" if partner_names else "") + "."
                    )

                if "google_partnership_mentions" in signals:
                    reasoning_parts.append(
                        f"Ricerca Google: {data.get('evidence_count', 0)} "
                        f"domini terzi confermano accordi di partnership."
                    )

                reasoning_parts.append(
                    f"Dichiarato: {int(declared)} partner. "
                    f"Verificati/trovati: {int(partners_found)} "
                    f"(lower bound — non tutte le partnership sono pubbliche). "
                    f"Scarto stimato (con tolleranza {int(self.TOLERANCE*100)}%): "
                    f"{magnitude:.0%}."
                )

                if magnitude > 0.5:
                    flags.append(
                        f"Partner verificabili ({int(partners_found)}) "
                        f"significativamente inferiori al dichiarato ({int(declared)})"
                    )

                if partner_names:
                    flags.append(f"Esempi trovati: {', '.join(partner_names[:5])}")

                return ClaimVerdict(
                    claim_id=claim_id, claim_type=self.CLAIM_TYPE,
                    claim_text=claim_text,
                    declared_value=declared, verified_value=partners_found,
                    verdict=verdict,
                    evidence_confidence=confidence, magnitude=magnitude,
                    reasoning=" ".join(reasoning_parts),
                    sources_used=sources, flags=flags,
                )

            if partners_found > 0 and declared is None:
                # Trovati partner ma valore dichiarato non quantificabile
                return ClaimVerdict(
                    claim_id=claim_id, claim_type=self.CLAIM_TYPE,
                    claim_text=claim_text,
                    declared_value=None, verified_value=partners_found,
                    verdict=Verdict.UNCERTAIN,
                    evidence_confidence=confidence, magnitude=0.0,
                    reasoning=f"Trovati {int(partners_found)} partner verificabili "
                              f"ma il numero dichiarato non è quantificabile con precisione.",
                    sources_used=sources,
                )

        # ── Solo Wayback disponibile ──────────────────────────────────────
        if wayback:
            sources.append("wayback_machine")
            wayback_flags = self._check_wayback_consistency(
                wayback, declared
            )
            if wayback_flags:
                flags.extend(wayback_flags)
                return ClaimVerdict(
                    claim_id=claim_id, claim_type=self.CLAIM_TYPE,
                    claim_text=claim_text,
                    declared_value=declared, verified_value=None,
                    verdict=Verdict.UNCERTAIN,
                    evidence_confidence=0.30, magnitude=0.0,
                    reasoning="Fonte partner non disponibile. "
                              "Storico sito web suggerisce incoerenze.",
                    sources_used=sources, flags=flags,
                )

        # ── Nessuna fonte ─────────────────────────────────────────────────
        return ClaimVerdict(
            claim_id=claim_id, claim_type=self.CLAIM_TYPE,
            claim_text=claim_text,
            declared_value=declared, verified_value=None,
            verdict=Verdict.INSUFFICIENT_DATA,
            evidence_confidence=0.0, magnitude=0.0,
            reasoning="Nessuna evidenza di partnership trovata. "
                      "Inserire il sito web aziendale per abilitare la verifica "
                      "della pagina partner.",
            sources_used=[],
            sources_consulted=["website_partner_page", "google_partnership_search",
                               "wayback_machine"],
        )

    @staticmethod
    def _check_wayback_consistency(
        wayback: Optional[dict], declared_current: Optional[float]
    ) -> list[str]:
        """
        Controlla se i numeri storici del sito sono coerenti con il dichiarato attuale.
        Segnala regressioni non spiegate (es. 500 → 300 senza comunicazione pubblica).
        """
        if not wayback or not declared_current:
            return []

        flags = []
        timeline = wayback.get("data", {}).get("timeline", [])
        historical_values = []

        for snapshot in timeline:
            metrics = snapshot.get("metrics", {})
            if "partner_count" in metrics:
                try:
                    val = float("".join(filter(str.isdigit, str(metrics["partner_count"]))))
                    historical_values.append(val)
                except ValueError:
                    pass

        if len(historical_values) >= 2:
            max_historical = max(historical_values)
            if declared_current < max_historical * 0.7:
                flags.append(
                    f"Regressione storica: valore massimo precedente "
                    f"({max_historical:.0f}) > dichiarato attuale ({declared_current:.0f}). "
                    f"Riduzione non comunicata pubblicamente."
                )
        return flags


class FundingVerifier(BaseVerifier):
    """
    Verifica claim di tipo 'funding'.
    Fonte: Crunchbase (API o news scraping).
    Il funding è il tipo più verificabile: o il round è tracciato o non lo è.
    """

    CLAIM_TYPE = "funding"

    def verify(self, claim: dict, collector_results: list[dict]) -> ClaimVerdict:
        claim_id   = claim.get("id", "")
        claim_text = claim.get("text", "")
        declared   = self._to_float(claim.get("normalized_value"))
        sources    = []
        flags      = []

        crunchbase = self._extract_connector(collector_results, "crunchbase")

        if not crunchbase:
            return ClaimVerdict(
                claim_id=claim_id, claim_type=self.CLAIM_TYPE,
                claim_text=claim_text,
                declared_value=declared, verified_value=None,
                verdict=Verdict.INSUFFICIENT_DATA,
                evidence_confidence=0.0, magnitude=0.0,
                reasoning="Nessuna fonte ha restituito dati sul funding.",
                sources_used=[], sources_consulted=["crunchbase", "news_scraping"],
            )

        sources.append("crunchbase")
        data   = crunchbase.get("data", {})
        source = data.get("source", "news_scraping")

        # ── Caso A: API Crunchbase con dati strutturati ───────────────────
        if source == "crunchbase_api":
            funding_usd = self._to_float(data.get("funding_total_usd"))
            confidence  = float(crunchbase.get("confidence", 0.85))

            if funding_usd is None:
                return ClaimVerdict(
                    claim_id=claim_id, claim_type=self.CLAIM_TYPE,
                    claim_text=claim_text,
                    declared_value=declared, verified_value=None,
                    verdict=Verdict.UNCERTAIN,
                    evidence_confidence=0.5, magnitude=0.0,
                    reasoning="Profilo Crunchbase trovato ma nessun round di funding tracciato. "
                              "Possibile round non annunciato pubblicamente (comune per seed/pre-seed).",
                    sources_used=sources,
                    flags=["round_non_annunciato_possibile"],
                )

            # Conversione USD → EUR approssimativa (1 USD ≈ 0.92 EUR)
            funding_eur = funding_usd * 0.92
            magnitude   = self._magnitude(declared, funding_eur) if declared else 0.0
            verdict     = self._compute_verdict(magnitude, confidence) if declared else Verdict.UNCERTAIN

            num_rounds = data.get("num_rounds", 0)
            last_type  = data.get("last_funding_type", "n/d")

            reasoning = (
                f"Crunchbase API: funding totale tracciato ${funding_usd:,.0f} "
                f"(≈€{funding_eur:,.0f}), {num_rounds} round(s), "
                f"ultimo tipo: {last_type}. "
            )
            if declared:
                reasoning += f"Dichiarato: €{declared:,.0f}. Scarto: {magnitude:.0%}."

            return ClaimVerdict(
                claim_id=claim_id, claim_type=self.CLAIM_TYPE,
                claim_text=claim_text,
                declared_value=declared, verified_value=funding_eur,
                verdict=verdict,
                evidence_confidence=confidence, magnitude=magnitude,
                reasoning=reasoning, sources_used=sources, flags=flags,
            )

        # ── Caso B: scraping news ─────────────────────────────────────────
        news_titles   = data.get("news_titles", [])
        fund_mentions = data.get("funding_mentions", [])
        confidence    = float(crunchbase.get("confidence", 0.50))

        if not news_titles and not fund_mentions:
            flags.append("Nessuna menzione pubblica di funding trovata")
            return ClaimVerdict(
                claim_id=claim_id, claim_type=self.CLAIM_TYPE,
                claim_text=claim_text,
                declared_value=declared, verified_value=None,
                verdict=Verdict.UNCERTAIN,
                evidence_confidence=0.30, magnitude=0.0,
                reasoning="Nessuna menzione pubblica di round di funding trovata su news italiane. "
                          "Il round potrebbe essere non annunciato (seed/pre-seed) "
                          "o potrebbe non essere avvenuto.",
                sources_used=sources, flags=flags,
            )

        # Cerca menzioni degli importi nei titoli
        verified_amounts = self._extract_amounts_from_news(news_titles + [
            m.get("title", "") for m in fund_mentions
        ])

        if verified_amounts and declared:
            # Prendi l'importo più vicino al dichiarato
            closest = min(verified_amounts, key=lambda x: abs(x - declared))
            magnitude = self._magnitude(declared, closest)
            verdict   = self._compute_verdict(magnitude, confidence)
        else:
            magnitude = 0.0
            verdict   = Verdict.UNCERTAIN

        reasoning = (
            f"Trovate {len(news_titles) + len(fund_mentions)} menzioni pubbliche. "
            f"Importi rilevati: {[f'€{a:,.0f}' for a in verified_amounts] or 'non quantificati'}. "
            f"Fonte: news scraping (confidence ridotta rispetto ad API)."
        )

        return ClaimVerdict(
            claim_id=claim_id, claim_type=self.CLAIM_TYPE,
            claim_text=claim_text,
            declared_value=declared,
            verified_value=verified_amounts[0] if verified_amounts else None,
            verdict=verdict,
            evidence_confidence=confidence, magnitude=magnitude,
            reasoning=reasoning, sources_used=sources, flags=flags,
        )

    @staticmethod
    def _extract_amounts_from_news(titles: list[str]) -> list[float]:
        """Estrae importi monetari dai titoli di news."""
        import re
        amounts = []
        patterns = [
            r"€\s*([\d,\.]+)\s*(M|K|milion|mila)?",
            r"([\d,\.]+)\s*(M|K|milion|mila)?\s*(?:euro|EUR|€)",
            r"([\d,\.]+)\s*milion",
        ]
        multipliers = {"M": 1_000_000, "K": 1_000, "milion": 1_000_000, "mila": 1_000}

        for title in titles:
            for pattern in patterns:
                for match in re.finditer(pattern, title, re.IGNORECASE):
                    try:
                        val = float(match.group(1).replace(",", "."))
                        suffix = match.group(2) if match.lastindex >= 2 else None
                        if suffix:
                            val *= multipliers.get(suffix.lower(), 1)
                        if val > 1000:   # importi < €1000 probabilmente non sono funding
                            amounts.append(val)
                    except (ValueError, IndexError):
                        pass
        return sorted(set(amounts), reverse=True)


class TeamSizeVerifier(BaseVerifier):
    """
    Verifica claim di tipo 'team_size'.
    Fonte: LinkedIn headcount.
    """

    CLAIM_TYPE = "team_size"

    def verify(self, claim: dict, collector_results: list[dict]) -> ClaimVerdict:
        claim_id   = claim.get("id", "")
        claim_text = claim.get("text", "")
        declared   = self._to_float(claim.get("normalized_value"))
        sources    = []
        flags      = []

        linkedin = self._extract_connector(collector_results, "linkedin")

        if not linkedin:
            return ClaimVerdict(
                claim_id=claim_id, claim_type=self.CLAIM_TYPE,
                claim_text=claim_text,
                declared_value=declared, verified_value=None,
                verdict=Verdict.INSUFFICIENT_DATA,
                evidence_confidence=0.0, magnitude=0.0,
                reasoning="Profilo LinkedIn non trovato o non accessibile.",
                sources_used=[], sources_consulted=["linkedin"],
            )

        sources.append("linkedin")
        data       = linkedin.get("data", {})
        midpoint   = self._to_float(data.get("headcount_midpoint"))
        hc_range   = data.get("headcount_range", "")
        confidence = float(linkedin.get("confidence", 0.65))

        if midpoint is None:
            return ClaimVerdict(
                claim_id=claim_id, claim_type=self.CLAIM_TYPE,
                claim_text=claim_text,
                declared_value=declared, verified_value=None,
                verdict=Verdict.UNCERTAIN,
                evidence_confidence=0.3, magnitude=0.0,
                reasoning="Profilo LinkedIn trovato ma headcount non disponibile.",
                sources_used=sources,
            )

        if declared is None:
            return ClaimVerdict(
                claim_id=claim_id, claim_type=self.CLAIM_TYPE,
                claim_text=claim_text,
                declared_value=None, verified_value=midpoint,
                verdict=Verdict.UNCERTAIN,
                evidence_confidence=confidence, magnitude=0.0,
                reasoning=f"LinkedIn riporta fascia: {hc_range} dipendenti (midpoint: {midpoint:.0f}). "
                          f"Valore dichiarato non quantificabile.",
                sources_used=sources,
            )

        magnitude = self._magnitude(declared, midpoint)
        verdict   = self._compute_verdict(magnitude, confidence)

        # LinkedIn è una stima — tollera scarti fino al 30% senza segnalare
        if magnitude < 0.30 and verdict == Verdict.WARNING:
            verdict = Verdict.VERIFIED
            flags.append("Scarto entro margine tolleranza LinkedIn (±30%)")

        reasoning = (
            f"LinkedIn headcount: fascia {hc_range} (midpoint: {midpoint:.0f}). "
            f"Dichiarato: {declared:.0f}. "
            f"Scarto: {magnitude:.0%}. "
            f"Nota: LinkedIn headcount è una stima — include solo profili auto-dichiarati."
        )

        return ClaimVerdict(
            claim_id=claim_id, claim_type=self.CLAIM_TYPE,
            claim_text=claim_text,
            declared_value=declared, verified_value=midpoint,
            verdict=verdict,
            evidence_confidence=confidence, magnitude=magnitude,
            reasoning=reasoning, sources_used=sources, flags=flags,
        )


class LegalStatusVerifier:
    """
    Non è un verifier di claim ma un modulo autonomo che
    arricchisce il VerificationResult con i dati legali da OpenCorporates.

    Produce la sezione "Stato Legale" del report con:
    - Badge stato (Attiva / Cessata / Irregolare / Non trovata)
    - Data costituzione, tipo societario, indirizzo
    - Flag critici: azienda cessata, costituita da < 1 anno, cambio nome recente
    """

    @staticmethod
    def enrich(result: "VerificationResult", raw_results: list[dict]) -> None:
        """Popola result.legal_status dai dati OpenCorporates disponibili."""
        oc_results = [
            r for r in raw_results
            if r.get("connector") == "opencorporates" and r.get("found")
        ]

        if not oc_results:
            result.legal_status = {
                "found": False,
                "status_normalized": "non_trovata",
                "status_label": "Non trovata su OpenCorporates",
                "flags": [],
                "source": "opencorporates",
            }
            return

        data  = oc_results[0].get("data", {})
        flags = []

        # ── Flag critici ──────────────────────────────────────────────────
        status = data.get("status_normalized", "sconosciuto")

        if status == "cessata":
            flags.append({
                "severity": "critical",
                "text": f"Azienda CESSATA"
                        + (f" il {data['dissolution_date']}" if data.get("dissolution_date") else "")
                        + ". Verificare se opera tramite altra entità.",
            })

        if data.get("incorporation_date"):
            try:
                from datetime import date
                inc = date.fromisoformat(data["incorporation_date"])
                age_days = (date.today() - inc).days
                if age_days < 365:
                    flags.append({
                        "severity": "warning",
                        "text": f"Azienda costituita da meno di un anno "
                                f"({inc.strftime('%d/%m/%Y')}). "
                                f"Storico operativo limitato.",
                    })
                elif age_days < 730:
                    flags.append({
                        "severity": "info",
                        "text": f"Azienda costituita da meno di due anni "
                                f"({inc.strftime('%d/%m/%Y')}).",
                    })
            except Exception:
                pass

        if data.get("previous_names"):
            flags.append({
                "severity": "warning",
                "text": f"Nomi precedenti registrati: "
                        f"{', '.join(data['previous_names'][:3])}. "
                        f"Verificare continuità operativa.",
            })

        # Propaghiamo flag critici anche nei red_flags del report principale
        for flag in flags:
            if flag["severity"] == "critical":
                result.red_flags.append(
                    type("LegalFlag", (), {
                        "claim_id": "LEGAL_STATUS",
                        "claim_type": "legal_status",
                        "verdict": type("V", (), {"value": "discrepancy"})(),
                        "claim_text": flag["text"],
                    })()
                )

        result.legal_status = {
            "found":              True,
            "name":               data.get("name", ""),
            "company_number":     data.get("company_number", ""),
            "company_type":       data.get("company_type", ""),
            "incorporation_date": data.get("incorporation_date"),
            "dissolution_date":   data.get("dissolution_date"),
            "current_status":     data.get("current_status", ""),
            "status_normalized":  status,
            "status_label":       LegalStatusVerifier._status_label(status),
            "registered_address": data.get("registered_address", ""),
            "previous_names":     data.get("previous_names", []),
            "opencorporates_url": data.get("opencorporates_url", ""),
            "registry_url":       data.get("registry_url", ""),
            "flags":              flags,
            "source":             "opencorporates",
        }

    @staticmethod
    def _status_label(status: str) -> str:
        return {
            "attiva":        "Attiva",
            "cessata":       "Cessata",
            "inattiva":      "Inattiva",
            "sconosciuto":   "Stato sconosciuto",
            "non_trovata":   "Non trovata",
        }.get(status, status.title())


class OtherVerifier(BaseVerifier):
    """Catch-all per tipi di claim non gestiti."""

    CLAIM_TYPE = "other"

    def verify(self, claim: dict, collector_results: list[dict]) -> ClaimVerdict:
        return ClaimVerdict(
            claim_id=claim.get("id", ""),
            claim_type="other",
            claim_text=claim.get("text", ""),
            declared_value=None, verified_value=None,
            verdict=Verdict.INSUFFICIENT_DATA,
            evidence_confidence=0.0, magnitude=0.0,
            reasoning="Tipo di claim non gestito nella v1. Nessuna fonte consultata.",
            sources_used=[], sources_consulted=[],
            flags=["fuori_scope_v1"],
        )


# ─────────────────────────────────────────────
#  Verification Engine — orchestratore
# ─────────────────────────────────────────────

class VerificationEngine:
    """
    Modulo 3 — orchestratore principale.
    Riceve claim (da ClaimExtractor) e dati raccolti (da DataCollector)
    e produce un VerificationResult con verdict per ogni claim
    e Trust Score aggregato.

    Utilizzo:
        engine = VerificationEngine()
        result = engine.verify(
            company_name="MoveNow S.r.l.",
            claims=extractor_result.claims,
            collection=collector_result,
            sector="mobilità",
        )
        print(result.summary())
    """

    VERIFIERS = {
        "revenue":       RevenueVerifier(),
        "partner_count": PartnerCountVerifier(),
        "funding":       FundingVerifier(),
        "team_size":     TeamSizeVerifier(),
        "other":         OtherVerifier(),
    }

    def verify(
        self,
        company_name: str,
        claims: list,
        collection,
        sector: str = "default",
    ) -> VerificationResult:

        result = VerificationResult(company_name=company_name)

        # Normalizza i collector results in una lista di dict
        if hasattr(collection, "results"):
            raw_results = [
                r.to_dict() if hasattr(r, "to_dict") else r
                for r in collection.results
            ]
        else:
            raw_results = collection

        for claim in claims:
            # Normalizza claim in dict
            claim_dict = claim if isinstance(claim, dict) else {
                "id":               claim.id,
                "type":             claim.type.value if hasattr(claim.type, "value") else claim.type,
                "text":             claim.text,
                "normalized_value": claim.normalized_value,
                "sector":           sector,
            }

            claim_type = claim_dict.get("type", "other")

            # Filtra i risultati del collector pertinenti a questa claim
            claim_results = [
                r for r in raw_results
                if r.get("claim_id") == claim_dict.get("id")
            ]

            verifier = self.VERIFIERS.get(claim_type, self.VERIFIERS["other"])
            log.info(f"Verifying [{claim_type}] {claim_dict.get('id')}")

            try:
                verdict = verifier.verify(claim_dict, claim_results)
                result.verdicts.append(verdict)
            except Exception as e:
                result.errors.append(f"Errore verifica {claim_dict.get('id')}: {e}")
                log.error(f"Errore verifica: {e}")

        # ── Categorizza per severity ──────────────────────────────────────
        result.red_flags = [
            v for v in result.verdicts if v.verdict == Verdict.DISCREPANCY
        ]
        result.warnings_list = [
            v for v in result.verdicts if v.verdict == Verdict.WARNING
        ]
        result.unverifiable = [
            v for v in result.verdicts
            if v.verdict in (Verdict.INSUFFICIENT_DATA, Verdict.UNCERTAIN)
        ]

        # ── Arricchisci con dati legali OpenCorporates ───────────────────
        LegalStatusVerifier.enrich(result, raw_results)

        # ── Calcola Trust Score ───────────────────────────────────────────
        result.trust_score = self._compute_trust_score(result.verdicts)
        result.trust_score_label = self._trust_label(result.trust_score)

        log.info(f"Verifica completata:\n{result.summary()}")
        return result

    @staticmethod
    def _compute_trust_score(verdicts: list[ClaimVerdict]) -> float:
        """
        Trust Score 0–10, calcolato SOLO sulle claim effettivamente verificate.

        Logica:
        - Le claim con INSUFFICIENT_DATA o UNCERTAIN non entrano nel calcolo
          del punteggio: non sapere ≠ inaffidabile.
        - Se la copertura verificabile è < 30% del peso totale possibile,
          il sistema restituisce -1 (segnale speciale = "non valutabile").
        - Il Trust Score riflette solo i verdict concreti (verified / warning /
          discrepancy) pesati per tipo di claim e confidence.
        """
        if not verdicts:
            return -1.0   # nessuna claim = non valutabile

        # Separa claim verificabili da quelle senza dati
        verifiable = [
            v for v in verdicts
            if v.verdict not in (Verdict.INSUFFICIENT_DATA, Verdict.UNCERTAIN)
        ]
        all_weight   = sum(CLAIM_TYPE_WEIGHTS.get(v.claim_type, 0.05) for v in verdicts)
        verif_weight = sum(CLAIM_TYPE_WEIGHTS.get(v.claim_type, 0.05) for v in verifiable)

        # Copertura insufficiente → non valutabile
        coverage = verif_weight / all_weight if all_weight > 0 else 0
        if coverage < 0.30 or not verifiable:
            return -1.0

        weighted_sum = 0.0
        weight_total = 0.0
        for v in verifiable:
            w = CLAIM_TYPE_WEIGHTS.get(v.claim_type, 0.05)
            score = v.verdict_score * v.evidence_confidence
            weighted_sum += w * score
            weight_total += w

        raw = weighted_sum / weight_total if weight_total > 0 else 0
        return round(max(0.5, min(9.5, raw * 10)), 1)

    @staticmethod
    def _trust_label(score: float) -> str:
        if score < 0:     return "Dati insufficienti per una valutazione"
        if score >= 7.5:  return "Alta affidabilità"
        if score >= 5.5:  return "Affidabilità moderata"
        if score >= 3.5:  return "Bassa affidabilità"
        return "Molto bassa affidabilità"
