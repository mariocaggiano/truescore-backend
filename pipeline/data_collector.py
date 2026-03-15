"""
TrueScore — Data Collector
===========================
Raccoglie dati da fonti esterne per verificare le claim estratte.
Tutti i connettori usano API gratuite o scraping rispettoso.

Connettori implementati:
  1. BilancioConnector      — dati dal bilancio caricato (input locale, costo zero)
  2. OverpassConnector      — strutture geografiche via OpenStreetMap (gratuito)
  3. CrunchbaseConnector    — funding e dati aziendali (free tier)
  4. LinkedInConnector      — headcount via scraping throttled (costo zero)
  5. WaybackConnector       — storico versioni sito web (gratuito)

Dipendenze: requests, beautifulsoup4
"""

import json
import time
import logging
import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup

log = logging.getLogger("data_collector")


# ─────────────────────────────────────────────
#  Strutture dati output
# ─────────────────────────────────────────────

@dataclass
class ConnectorResult:
    """Output standardizzato di ogni connettore."""
    connector: str
    claim_id: str
    claim_type: str
    found: bool                          # dato trovato o no
    data: dict = field(default_factory=dict)
    confidence: float = 0.0              # 0.0–1.0 sull'affidabilità del dato
    source_url: str = ""
    fetched_at: str = ""
    error: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.fetched_at:
            self.fetched_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CollectionResult:
    """Aggregato di tutti i risultati per un'azienda."""
    company_name: str
    results: list[ConnectorResult] = field(default_factory=list)
    cache_hits: int = 0
    errors: list[str] = field(default_factory=list)

    def by_claim(self, claim_id: str) -> list[ConnectorResult]:
        return [r for r in self.results if r.claim_id == claim_id]

    def by_type(self, claim_type: str) -> list[ConnectorResult]:
        return [r for r in self.results if r.claim_type == claim_type]

    def summary(self) -> str:
        found = sum(1 for r in self.results if r.found)
        lines = [
            f"  Azienda     : {self.company_name}",
            f"  Risultati   : {len(self.results)} totali, {found} con dati",
            f"  Cache hits  : {self.cache_hits}",
            f"  Errori      : {len(self.errors)}",
        ]
        by_connector = {}
        for r in self.results:
            by_connector[r.connector] = by_connector.get(r.connector, 0) + 1
        for c, n in by_connector.items():
            lines.append(f"    [{c}] {n} query")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Cache in memoria (Redis-compatible interface)
# ─────────────────────────────────────────────

class InMemoryCache:
    """
    Cache in memoria per i risultati delle chiamate API.
    In produzione sostituire con Redis (stessa interfaccia).
    TTL default: 30 giorni.
    """

    def __init__(self, ttl_seconds: int = 30 * 24 * 3600):
        self._store: dict[str, tuple[str, datetime]] = {}
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[str]:
        if key not in self._store:
            return None
        value, expires_at = self._store[key]
        if datetime.utcnow() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: str):
        expires_at = datetime.utcnow() + timedelta(seconds=self.ttl)
        self._store[key] = (value, expires_at)

    def make_key(self, connector: str, *args) -> str:
        raw = f"{connector}:" + ":".join(str(a) for a in args)
        return hashlib.md5(raw.encode()).hexdigest()


# ─────────────────────────────────────────────
#  Base Connector
# ─────────────────────────────────────────────

class BaseConnector(ABC):
    """Interfaccia comune per tutti i connettori."""

    NAME = "base"
    REQUEST_DELAY = 1.0   # secondi tra chiamate (throttling rispettoso)

    def __init__(self, cache: Optional[InMemoryCache] = None, timeout: int = 20):
        self.cache = cache or InMemoryCache()
        self.timeout = timeout
        self._last_request = 0.0

    def _throttle(self):
        """Rispetta il delay tra richieste per non sovraccaricare i server."""
        elapsed = time.time() - self._last_request
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request = time.time()

    def _get(self, url: str, headers: dict = None, params: dict = None, timeout: int = None) -> requests.Response:
        self._throttle()
        h = {"User-Agent": "TrueScore/1.0 (business verification tool; respectful crawler)"}
        if headers:
            h.update(headers)
        return requests.get(url, headers=h, params=params, timeout=timeout or self.timeout)

    @abstractmethod
    def fetch(self, claim: dict, company_name: str) -> ConnectorResult:
        pass

    def is_available(self) -> bool:
        return True

    def estimated_cost_eur(self) -> float:
        return 0.0


# ─────────────────────────────────────────────
#  Connettore 1 — Bilancio (locale)
# ─────────────────────────────────────────────

class BilancioConnector(BaseConnector):
    """
    Usa i dati finanziari già estratti dal bilancio caricato dall'utente.
    Non fa chiamate esterne — costo zero, confidence massima.
    """

    NAME = "bilancio"

    def __init__(self, financial_data: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.financial_data = financial_data or {}

    def fetch(self, claim: dict, company_name: str) -> ConnectorResult:
        claim_id   = claim.get("id", "")
        claim_type = claim.get("type", "")

        if not self.financial_data:
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=False, notes="Nessun bilancio caricato",
                confidence=0.0
            )

        if claim_type != "revenue":
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=False, notes="Connettore bilancio usato solo per claim revenue",
                confidence=0.0
            )

        revenues = self.financial_data.get("revenues")
        if revenues is None:
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=False, notes="Ricavi non presenti nel bilancio caricato",
                confidence=self.financial_data.get("extraction_confidence", 0.5)
            )

        return ConnectorResult(
            connector=self.NAME,
            claim_id=claim_id,
            claim_type=claim_type,
            found=True,
            data={
                "revenues": revenues,
                "exercise_year": self.financial_data.get("exercise_year"),
                "employees": self.financial_data.get("employees"),
                "total_assets": self.financial_data.get("total_assets"),
                "raw_excerpt": self.financial_data.get("raw_excerpt", ""),
            },
            confidence=self.financial_data.get("extraction_confidence", 0.9),
            source_url="infocamere_upload",
            notes=f"Dati da bilancio esercizio {self.financial_data.get('exercise_year', 'n/d')} caricato dall'utente"
        )


# ─────────────────────────────────────────────
#  Connettore 2 — Partner Website Scraper
# ─────────────────────────────────────────────

class PartnerWebsiteConnector(BaseConnector):
    """
    Verifica le claim di tipo partner_count cercando evidenze reali di partnership.

    Strategia multi-segnale (in ordine di affidabilità):

    1. PAGINA PARTNER sul sito aziendale
       Cerca URL tipo /partner, /clienti, /network, /ecosystem sul sito dichiarato.
       Conta i nomi/loghi aziendali elencati — quello è il numero verificabile.

    2. GOOGLE SEARCH PARTNERSHIP
       Cerca "[azienda] partner accordo collaborazione" e conta quante aziende
       terze confermano pubblicamente la relazione.
       Ogni menzione da un dominio diverso da quello aziendale = +1 partner verificato.

    3. CONFRONTO DICHIARATO vs TROVATO
       Restituisce: partner_found (trovati), partner_declared (dichiarati),
       partner_names (lista nomi se disponibili), evidence_urls.
    """

    NAME          = "partner_website"
    REQUEST_DELAY = 2.0
    PARTNER_SLUGS = [
        "/partner", "/partners", "/clienti", "/clients",
        "/network", "/ecosystem", "/collaborazioni", "/brand-partner",
        "/brand", "/aziende-partner", "/chi-usa", "/chi-lo-usa",
        "/case-study", "/casi-studio",
    ]
    GOOGLE_SEARCH = "https://www.google.com/search?q="

    def fetch(self, claim: dict, company_name: str) -> ConnectorResult:
        claim_id    = claim.get("id", "")
        claim_type  = claim.get("type", "")
        website_url = claim.get("website_url", "").strip().rstrip("/")

        if claim_type != "partner_count":
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=False, notes="PartnerWebsite: usato solo per partner_count"
            )

        cache_key = self.cache.make_key(self.NAME, company_name)
        cached = self.cache.get(cache_key)
        if cached:
            data = json.loads(cached)
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=True, data=data, confidence=data.get("confidence", 0.6),
                source_url=data.get("source_url", ""),
                notes="Risultato da cache"
            )

        data = {}
        signals = []

        # ── Segnale 1: pagina partner sul sito ──────────────────────────
        if website_url:
            page_result = self._scrape_partner_page(website_url, company_name)
            if page_result:
                data.update(page_result)
                signals.append(("website_partner_page", page_result.get("confidence", 0.7)))

        # ── Segnale 2: Google search menzioni partnership ─────────────
        google_result = self._google_partnership_search(company_name)
        if google_result:
            data.update(google_result)
            signals.append(("google_partnership_mentions", google_result.get("confidence", 0.5)))

        if not signals:
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=False,
                notes=f"Nessuna evidenza di partnership trovata per '{company_name}'"
            )

        # Confidence finale = media pesata dei segnali trovati
        conf = sum(c for _, c in signals) / len(signals)
        data["confidence"] = round(conf, 2)
        data["signals_found"] = [s for s, _ in signals]

        self.cache.set(cache_key, json.dumps(data))
        return ConnectorResult(
            connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
            found=True, data=data, confidence=conf,
            source_url=data.get("source_url", ""),
            notes=f"Partner trovati: {data.get('partners_found', 0)} "
                  f"(segnali: {', '.join(data['signals_found'])})"
        )

    def _scrape_partner_page(self, base_url: str, company_name: str) -> Optional[dict]:
        """
        Cerca una pagina partner/clienti sul sito aziendale e conta i partner elencati.
        Timeout aggressivo per non bloccare la pipeline.
        """
        # Normalizza base_url
        if not base_url.startswith("http"):
            base_url = "https://" + base_url

        # Prima controlla la homepage per link a sezione partner
        try:
            resp = self._get(base_url, timeout=6)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.select("a[href]"):
                    href   = a.get("href", "").lower()
                    text   = a.get_text(strip=True).lower()
                    if any(kw in href or kw in text for kw in
                           ["partner", "client", "brand", "network",
                            "ecosystem", "collabor", "chi-usa"]):
                        full = href if href.startswith("http") else base_url + "/" + href.lstrip("/")
                        if full not in [base_url, base_url + "/"]:
                            self.PARTNER_SLUGS = [full.replace(base_url, "")] + self.PARTNER_SLUGS
        except Exception:
            pass

        for slug in self.PARTNER_SLUGS[:8]:   # max 8 slug, non tutti i 14
            url = base_url + slug if not slug.startswith("http") else slug
            try:
                time.sleep(0.5)   # delay ridotto da 2s a 0.5s
                resp = self._get(url, timeout=5)   # timeout aggressivo
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                page_text = soup.get_text(" ", strip=True)

                # Segnale debole: pagina troppo corta = non è una vera pagina partner
                if len(page_text) < 200:
                    continue

                # Conta nomi aziendali: img[alt], titoli, link esterni
                partner_names = self._extract_partner_names(soup, base_url)

                if len(partner_names) >= 2:
                    log.info(f"PartnerWebsite: trovata pagina partner a {url} "
                             f"({len(partner_names)} partner)")
                    return {
                        "source":          "website_partner_page",
                        "source_url":      url,
                        "partners_found":  len(partner_names),
                        "partner_names":   partner_names[:30],  # max 30 nomi
                        "confidence":      0.75,
                    }

            except Exception as e:
                log.debug(f"PartnerWebsite: {url} — {e}")
                continue

        return None

    def _extract_partner_names(self, soup: "BeautifulSoup", base_url: str) -> list[str]:
        """
        Estrae nomi di partner da una pagina HTML.
        Strategia: img[alt] con testo significativo + link esterni + heading list.
        """
        names = set()
        domain = re.sub(r"https?://(?:www\.)?", "", base_url).split("/")[0]

        # 1. Loghi partner: img con alt text aziendale
        for img in soup.find_all("img", alt=True):
            alt = img["alt"].strip()
            if 3 < len(alt) < 60 and not any(
                skip in alt.lower()
                for skip in ["logo", "icon", "banner", "sfondo", "background",
                             "home", "menu", "search", "freccia", "arrow"]
            ):
                names.add(alt)

        # 2. Link esterni (partner che linkano al proprio sito)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http") and domain not in href:
                link_text = a.get_text(strip=True)
                if 3 < len(link_text) < 60:
                    names.add(link_text)

        # 3. Liste di nomi in elementi strutturati (ul/ol, grid di card)
        for tag in soup.find_all(["li", "h3", "h4"]):
            txt = tag.get_text(strip=True)
            if 3 < len(txt) < 80 and not any(
                skip in txt.lower()
                for skip in ["scopri", "leggi", "clicca", "contatti",
                             "cookie", "privacy", "home", "blog"]
            ):
                names.add(txt)

        return list(names)

    def _google_partnership_search(self, company_name: str) -> Optional[dict]:
        """
        Cerca su Google menzioni di partnership con aziende terze.
        Conta quanti domini distinti (non aziendali) confermano relazioni.
        """
        try:
            query = f'"{company_name}" partner accordo collaborazione clienti'
            url   = f"{self.GOOGLE_SEARCH}{quote_plus(query)}&num=20"
            time.sleep(self.REQUEST_DELAY)
            resp  = self._get(url)
            soup  = BeautifulSoup(resp.text, "html.parser")

            company_slug = company_name.lower().replace(" ", "")
            confirmed_partners = set()
            evidence_urls      = []

            for result in soup.select("div.g, div[data-sokoban-container]"):
                # Titolo del risultato
                title_tag = result.select_one("h3")
                title = title_tag.get_text(strip=True) if title_tag else ""

                # URL del risultato
                link = result.select_one("a[href]")
                href = link["href"] if link else ""

                # Snippet
                snippet_tag = result.select_one("div.VwiC3b, span.aCOpRe")
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

                combined = (title + " " + snippet).lower()

                # Conta solo se il dominio è terzo (non il sito dell'azienda)
                if company_slug not in href.lower() and href.startswith("http"):
                    if any(kw in combined for kw in
                           ["partner", "accordo", "collabora", "client",
                            "integrazion", "ecosystem", "annunci"]):
                        domain = re.sub(r"https?://(?:www\.)?", "", href).split("/")[0]
                        if domain and domain not in confirmed_partners:
                            confirmed_partners.add(domain)
                            evidence_urls.append({"url": href, "title": title[:80]})

            if len(confirmed_partners) < 1:
                return None

            log.info(f"PartnerWebsite Google: {len(confirmed_partners)} "
                     f"menzioni di partnership per '{company_name}'")
            return {
                "source":              "google_partnership_mentions",
                "partners_found":      len(confirmed_partners),
                "evidence_count":      len(confirmed_partners),
                "evidence_urls":       evidence_urls[:10],
                "confidence":          0.55,  # Google è indiretto — confidence moderata
            }

        except Exception as e:
            log.warning(f"PartnerWebsite Google search error: {e}")
            return None


# ─────────────────────────────────────────────
#  Connettore 3 — Crunchbase (free tier)
# ─────────────────────────────────────────────

class CrunchbaseConnector(BaseConnector):
    """
    Recupera dati di funding da Crunchbase.
    Usa il free tier (rate limited) + scraping pubblico come fallback.
    """

    NAME = "crunchbase"
    API_BASE = "https://api.crunchbase.com/api/v4"
    WEB_BASE = "https://www.crunchbase.com/organization"
    REQUEST_DELAY = 3.0

    def __init__(self, api_key: str = "", **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key

    def fetch(self, claim: dict, company_name: str) -> ConnectorResult:
        claim_id   = claim.get("id", "")
        claim_type = claim.get("type", "")

        if claim_type not in ("funding", "revenue", "team_size"):
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=False, notes="Crunchbase: tipo claim non gestito"
            )

        slug = self._name_to_slug(company_name)
        cache_key = self.cache.make_key(self.NAME, slug)
        cached = self.cache.get(cache_key)

        if cached:
            data = json.loads(cached)
            return self._build_result(claim_id, claim_type, data, from_cache=True)

        # Prova API se disponibile, altrimenti scraping
        if self.api_key:
            data = self._fetch_via_api(slug)
        else:
            data = self._fetch_via_scraping(slug, company_name)

        if data:
            self.cache.set(cache_key, json.dumps(data))
            return self._build_result(claim_id, claim_type, data)

        return ConnectorResult(
            connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
            found=False,
            notes=f"Azienda '{company_name}' non trovata su Crunchbase",
            confidence=0.0
        )

    def _fetch_via_api(self, slug: str) -> Optional[dict]:
        """Crunchbase API v4 — richiede API key (free tier disponibile)."""
        try:
            url = f"{self.API_BASE}/entities/organizations/{slug}"
            params = {
                "user_key": self.api_key,
                "field_ids": "funding_total,num_funding_rounds,last_funding_type,"
                             "last_funding_at,num_employees_enum,founded_on"
            }
            resp = self._get(url, params=params)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            entity = resp.json().get("properties", {})

            return {
                "source": "crunchbase_api",
                "slug": slug,
                "funding_total_usd": entity.get("funding_total", {}).get("value_usd"),
                "num_rounds": entity.get("num_funding_rounds"),
                "last_funding_type": entity.get("last_funding_type"),
                "last_funding_date": entity.get("last_funding_at"),
                "employees_range": entity.get("num_employees_enum"),
                "founded_on": entity.get("founded_on"),
            }
        except Exception as e:
            log.warning(f"Crunchbase API error: {e}")
            return None

    def _fetch_via_scraping(self, slug: str, company_name: str) -> Optional[dict]:
        """
        Fallback: cerca su Google News i comunicati stampa di funding.
        Più robusto di fare scraping diretto su Crunchbase (che blocca i bot).
        """
        try:
            query = f'"{company_name}" round funding milioni investimento'
            url = f"https://news.google.com/search?q={quote_plus(query)}&hl=it&gl=IT"
            resp = self._get(url)
            soup = BeautifulSoup(resp.text, "html.parser")

            articles = []
            for article in soup.select("article")[:5]:
                title_el = article.select_one("h3, h4, a[class*='title']")
                title = title_el.get_text(strip=True) if title_el else ""
                if title:
                    articles.append(title)

            # Cerca anche su StartupItalia e Il Sole 24 Ore
            funding_mentions = self._search_news_funding(company_name)

            if not articles and not funding_mentions:
                return None

            return {
                "source": "news_scraping",
                "slug": slug,
                "news_titles": articles,
                "funding_mentions": funding_mentions,
                "note": "Dati da scraping news — verificare manualmente"
            }

        except Exception as e:
            log.warning(f"Crunchbase scraping error: {e}")
            return None

    def _search_news_funding(self, company_name: str) -> list[dict]:
        """Cerca menzioni di funding su siti di news italiani indicizzati."""
        results = []
        sources = [
            f"https://startupitalia.eu/?s={quote_plus(company_name)}",
        ]
        for url in sources:
            try:
                resp = self._get(url)
                soup = BeautifulSoup(resp.text, "html.parser")
                for el in soup.select("h2 a, h3 a, .entry-title a")[:3]:
                    title = el.get_text(strip=True)
                    href = el.get("href", "")
                    if company_name.lower().split()[0] in title.lower():
                        results.append({"title": title, "url": href})
            except Exception:
                pass
        return results

    def _build_result(
        self, claim_id: str, claim_type: str, data: dict, from_cache: bool = False
    ) -> ConnectorResult:
        confidence = 0.85 if data.get("source") == "crunchbase_api" else 0.55
        return ConnectorResult(
            connector=self.NAME,
            claim_id=claim_id,
            claim_type=claim_type,
            found=True,
            data=data,
            confidence=confidence,
            source_url=f"{self.WEB_BASE}/{data.get('slug', '')}",
            notes=("Cache. " if from_cache else "") +
                  f"Fonte: {data.get('source', 'n/d')}"
        )

    @staticmethod
    def _name_to_slug(name: str) -> str:
        """Converte nome azienda in slug Crunchbase."""
        slug = name.lower()
        slug = re.sub(r"\b(s\.r\.l\.|s\.p\.a\.|s\.r\.l|srl|spa|ltd|inc|gmbh)\b", "", slug)
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"\s+", "-", slug.strip())
        slug = re.sub(r"-+", "-", slug)
        return slug.strip("-")


# ─────────────────────────────────────────────
#  Connettore 4 — LinkedIn (scraping throttled)
# ─────────────────────────────────────────────

class LinkedInConnector(BaseConnector):
    """
    Recupera il headcount aziendale da LinkedIn via scraping pubblico.
    LinkedIn mostra il numero di dipendenti senza autenticazione.
    Throttling rispettoso: 1 richiesta ogni 4 secondi.
    """

    NAME = "linkedin"
    SEARCH_URL = "https://www.linkedin.com/search/results/companies/"
    REQUEST_DELAY = 4.0

    def fetch(self, claim: dict, company_name: str) -> ConnectorResult:
        claim_id   = claim.get("id", "")
        claim_type = claim.get("type", "")

        if claim_type != "team_size":
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=False, notes="LinkedIn: usato solo per team_size"
            )

        cache_key = self.cache.make_key(self.NAME, company_name)
        cached = self.cache.get(cache_key)
        if cached:
            data = json.loads(cached)
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=True, data=data, confidence=0.65,
                source_url=data.get("profile_url", ""),
                notes="Risultato da cache"
            )

        # Usa URL LinkedIn diretto se fornito dall'utente — molto più affidabile
        linkedin_url = claim.get("linkedin_url") or claim.get("_meta", {}).get("linkedin_url")
        data = self._scrape_company_headcount(company_name, direct_url=linkedin_url)

        if data:
            self.cache.set(cache_key, json.dumps(data))
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=True, data=data,
                confidence=0.65,   # LinkedIn headcount è stima, non dato esatto
                source_url=data.get("profile_url", ""),
                notes="Headcount LinkedIn (stima — non dato ufficiale)"
            )

        return ConnectorResult(
            connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
            found=False,
            notes=f"Profilo LinkedIn non trovato per '{company_name}'",
            confidence=0.0
        )

    def _scrape_company_headcount(self, company_name: str, direct_url: Optional[str] = None) -> Optional[dict]:
        """
        Strategia in due step:
        1. Usa URL diretto se fornito dall'utente (priorità assoluta)
        2. Altrimenti cerca il profilo aziendale su Google
        3. Accede alla pagina pubblica e legge il headcount
        """
        if direct_url and "linkedin.com/company/" in direct_url:
            # Normalizza: rimuovi trailing slash e parametri
            profile_url = direct_url.split("?")[0].rstrip("/")
            log.info(f"LinkedIn: uso URL diretto fornito dall'utente: {profile_url}")
        else:
            profile_url = self._find_linkedin_url(company_name)
        if not profile_url:
            return None

        try:
            resp = self._get(profile_url)
            soup = BeautifulSoup(resp.text, "html.parser")

            headcount = self._extract_headcount(soup)
            followers  = self._extract_followers(soup)
            description = self._extract_description(soup)

            if headcount is None:
                return None

            return {
                "company_name": company_name,
                "profile_url": profile_url,
                "headcount_range": headcount,
                "headcount_midpoint": self._range_midpoint(headcount),
                "followers": followers,
                "description_snippet": description[:200] if description else "",
            }

        except Exception as e:
            log.warning(f"LinkedIn scraping error for '{company_name}': {e}")
            return None

    def _find_linkedin_url(self, company_name: str) -> Optional[str]:
        """Trova l'URL LinkedIn aziendale via ricerca Google."""
        try:
            query = f'site:linkedin.com/company "{company_name}" Italy'
            url = f"https://www.google.com/search?q={quote_plus(query)}"
            resp = self._get(url)
            soup = BeautifulSoup(resp.text, "html.parser")

            for a in soup.select("a[href]"):
                href = a.get("href", "")
                if "linkedin.com/company/" in href:
                    # Estrai URL LinkedIn dal redirect Google
                    match = re.search(r"linkedin\.com/company/[a-zA-Z0-9_-]+", href)
                    if match:
                        return "https://www." + match.group(0)
        except Exception as e:
            log.warning(f"LinkedIn URL lookup failed: {e}")
        return None

    @staticmethod
    def _extract_headcount(soup: BeautifulSoup) -> Optional[str]:
        """Estrae la fascia di dipendenti dalla pagina LinkedIn."""
        patterns = [
            r"(\d[\d,.]*\s*[-–]\s*\d[\d,.]*)\s*dipendenti",
            r"(\d[\d,.]*\s*[-–]\s*\d[\d,.]*)\s*employees",
            r"(\d+\+?)\s*dipendenti",
            r"(\d+\+?)\s*employees",
        ]
        text = soup.get_text()
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _extract_followers(soup: BeautifulSoup) -> Optional[int]:
        text = soup.get_text()
        m = re.search(r"([\d.,]+)\s*follower", text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(".", "").replace(",", "")
            return int(val)
        return None

    @staticmethod
    def _extract_description(soup: BeautifulSoup) -> str:
        for sel in ["p.description", ".about-us", "meta[name='description']"]:
            el = soup.select_one(sel)
            if el:
                return el.get("content", "") or el.get_text(strip=True)
        return ""

    @staticmethod
    def _range_midpoint(headcount_str: str) -> Optional[int]:
        """Calcola il punto medio di un range tipo '11-50'."""
        nums = re.findall(r"\d+", headcount_str)
        if len(nums) >= 2:
            return (int(nums[0]) + int(nums[1])) // 2
        if len(nums) == 1:
            return int(nums[0])
        return None



# ─────────────────────────────────────────────
#  Connettore 5b — Ufficio Camerale (scraping gratuito)
# ─────────────────────────────────────────────

class UfficioCameraleConnector(BaseConnector):
    """
    Recupera fatturato e dipendenti da ufficiocamerale.it.
    Il sito espone gratuitamente dati da Registro Imprese / Infocamere.
    Richiede la Partita IVA dell'azienda.

    Strategia:
    1. Cerca la pagina aziendale tramite Google (site:ufficiocamerale.it + P.IVA)
    2. Scarica la pagina pubblica
    3. Estrae fatturato, dipendenti, ATECO, stato attività
    """

    NAME         = "ufficiocamerale"
    REQUEST_DELAY = 3.0
    GOOGLE_SEARCH = "https://www.google.com/search?q="

    def fetch(self, claim: dict, company_name: str) -> ConnectorResult:
        claim_id   = claim.get("id", "")
        claim_type = claim.get("type", "")
        vat_number = claim.get("vat_number", "").strip()

        # Utile per revenue e team_size; skip se P.IVA mancante
        if claim_type not in ("revenue", "team_size"):
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=False, notes="UfficioCamerale: usato solo per revenue e team_size"
            )

        if not vat_number:
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=False, notes="UfficioCamerale: Partita IVA non fornita"
            )

        cache_key = self.cache.make_key(self.NAME, vat_number)
        cached = self.cache.get(cache_key)
        if cached:
            data = json.loads(cached)
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=True, data=data, confidence=0.80,
                source_url=data.get("page_url", ""),
                notes="Risultato da cache"
            )

        data = self._scrape_company_data(vat_number, company_name)

        if data:
            self.cache.set(cache_key, json.dumps(data))
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=True, data=data, confidence=0.80,
                source_url=data.get("page_url", ""),
                notes="Dati da Registro Imprese via ufficiocamerale.it"
            )

        return ConnectorResult(
            connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
            found=False,
            notes=f"Azienda non trovata su ufficiocamerale.it (P.IVA: {vat_number})",
            confidence=0.0
        )

    def _scrape_company_data(self, vat_number: str, company_name: str) -> Optional[dict]:
        """Cerca la pagina su Google, poi la scrapa."""
        page_url = self._find_page_url(vat_number, company_name)
        if not page_url:
            log.warning(f"UfficioCamerale: pagina non trovata per P.IVA {vat_number}")
            return None

        try:
            time.sleep(self.REQUEST_DELAY)
            resp = self._get(page_url)
            soup = BeautifulSoup(resp.text, "html.parser")
            return self._extract_data(soup, page_url, vat_number)
        except Exception as e:
            log.warning(f"UfficioCamerale scraping error: {e}")
            return None

    def _find_page_url(self, vat_number: str, company_name: str) -> Optional[str]:
        """
        Trova l'URL della pagina aziendale su ufficiocamerale.it.
        Strategia (in ordine):
        1. URL diretto /{vat_number} — il più affidabile
        2. URL diretto /cerca?piva={vat_number}
        3. Ricerca Google come ultimo tentativo
        """
        # Tentativo 1: URL diretto con P.IVA
        candidates = [
            f"https://www.ufficiocamerale.it/{vat_number}",
            f"https://www.ufficiocamerale.it/visura-camerale-gratuita/{vat_number}",
            f"https://www.ufficiocamerale.it/cerca?piva={vat_number}",
        ]
        for direct in candidates:
            try:
                time.sleep(1.0)
                r = self._get(direct)
                if r.status_code == 200 and (vat_number in r.text or company_name.lower() in r.text.lower()):
                    log.info(f"UfficioCamerale: trovata pagina diretta: {direct}")
                    return direct
            except Exception:
                continue

        # Tentativo 2: ricerca sul sito stesso
        try:
            search_url = f"https://www.ufficiocamerale.it/cerca?q={quote_plus(vat_number)}"
            time.sleep(1.5)
            r = self._get(search_url)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.select("a[href]"):
                    href = a.get("href", "")
                    if vat_number in href or (
                        "ufficiocamerale.it" in href
                        and "/cerca" not in href
                        and "/trova-azienda" not in href
                        and len(href) > 30
                    ):
                        if href.startswith("/"):
                            href = "https://www.ufficiocamerale.it" + href
                        return href
        except Exception as e:
            log.debug(f"UfficioCamerale search page: {e}")

        # Tentativo 3: Google (potrebbe essere bloccato da server)
        try:
            query = f'site:ufficiocamerale.it "{vat_number}"'
            url   = f"{self.GOOGLE_SEARCH}{quote_plus(query)}"
            resp  = self._get(url)
            soup  = BeautifulSoup(resp.text, "html.parser")
            for a in soup.select("a[href]"):
                href = a.get("href", "")
                if "ufficiocamerale.it/" in href and "/trova-azienda" not in href:
                    match = re.search(r"https?://www\.ufficiocamerale\.it/[^\s&'\"]+", href)
                    if match:
                        return match.group(0)
        except Exception as e:
            log.debug(f"UfficioCamerale Google: {e}")

        return None

    def _extract_data(self, soup: "BeautifulSoup", page_url: str, vat_number: str) -> Optional[dict]:
        """Estrae fatturato, dipendenti e altri dati dalla pagina HTML."""
        data = {
            "vat_number": vat_number,
            "page_url":   page_url,
            "revenues":   None,
            "employees":  None,
            "ateco_code": None,
            "company_status": None,
            "legal_form": None,
            "source":     "ufficiocamerale_scraping",
        }

        page_text = soup.get_text(" ", strip=True).lower()

        # ── Fatturato ─────────────────────────────────────────────────────
        for pattern in [
            r"fatturato[:\s]+([€£]?\s*[\d\.,]+\s*(?:mln|miliard|milion|k|€)?)",
            r"ricavi[:\s]+([€£]?\s*[\d\.,]+\s*(?:mln|miliard|milion|k|€)?)",
            r"([\d\.,]+)\s*(?:mln|milioni).*?(?:fatturato|ricavi)",
        ]:
            m = re.search(pattern, page_text)
            if m:
                data["revenues"] = self._parse_revenue(m.group(1))
                break

        # Prova anche tag strutturati (tabelle, dt/dd)
        if data["revenues"] is None:
            for tag in soup.find_all(["td", "dd", "span", "div"]):
                txt = tag.get_text(strip=True).lower()
                if any(k in txt for k in ["fatturato", "ricavi"]):
                    sib = tag.find_next_sibling()
                    if sib:
                        data["revenues"] = self._parse_revenue(sib.get_text(strip=True))
                    if data["revenues"]:
                        break

        # ── Dipendenti ────────────────────────────────────────────────────
        for pattern in [
            r"dipendenti[:\s]+(\d[\d\.,]*)",
            r"addetti[:\s]+(\d[\d\.,]*)",
            r"(\d+)\s+dipendenti",
        ]:
            m = re.search(pattern, page_text)
            if m:
                try:
                    data["employees"] = int(m.group(1).replace(".", "").replace(",", ""))
                except Exception:
                    pass
                if data["employees"]:
                    break

        # Prova anche tag strutturati
        if data["employees"] is None:
            for tag in soup.find_all(["td", "dd", "span", "div"]):
                txt = tag.get_text(strip=True).lower()
                if any(k in txt for k in ["dipendenti", "addetti"]):
                    sib = tag.find_next_sibling()
                    if sib:
                        try:
                            data["employees"] = int(re.sub(r"[^\d]", "", sib.get_text(strip=True)) or "0") or None
                        except Exception:
                            pass
                    if data["employees"]:
                        break

        # ── ATECO ─────────────────────────────────────────────────────────
        m = re.search(r"ateco[:\s]+([0-9]{2}\.[0-9]{2}(?:\.[0-9]+)?)", page_text)
        if m:
            data["ateco_code"] = m.group(1)

        # ── Stato attività ────────────────────────────────────────────────
        if "attiva" in page_text:
            data["company_status"] = "attiva"
        elif "cessata" in page_text:
            data["company_status"] = "cessata"
        elif "inattiva" in page_text:
            data["company_status"] = "inattiva"

        # Considera valido se almeno uno dei dati principali è presente
        if data["revenues"] or data["employees"]:
            return data
        return None

    @staticmethod
    def _parse_revenue(text: str) -> Optional[float]:
        """Converte testo tipo '€1.2 mln' o '1.200.000' in float."""
        if not text:
            return None
        text = text.lower().strip()
        try:
            # Gestisci moltiplicatori
            if "mld" in text or "miliard" in text:
                num = float(re.sub(r"[^\d,\.]", "", text).replace(",", "."))
                return num * 1_000_000_000
            if "mln" in text or "milion" in text:
                num = float(re.sub(r"[^\d,\.]", "", text).replace(",", "."))
                return num * 1_000_000
            if "k" in text:
                num = float(re.sub(r"[^\d,\.]", "", text).replace(",", "."))
                return num * 1_000
            # Numero semplice (es. 1.200.000 o 1,200,000)
            clean = re.sub(r"[^\d]", "", text)
            return float(clean) if clean else None
        except Exception:
            return None


# ─────────────────────────────────────────────
#  Connettore 6 — OpenCorporates (API gratuita)
# ─────────────────────────────────────────────

class OpenCorporatesConnector(BaseConnector):
    """
    Recupera dati legali ufficiali da OpenCorporates.
    API gratuita, no key richiesta per ricerche base.
    Fonte: registri societari pubblici di 140+ giurisdizioni.

    Restituisce:
    - Stato attività (attiva / cessata / irregolare)
    - Data costituzione e tipo societario
    - Indirizzo legale registrato
    - Eventuali nomi precedenti (flag se il nome è cambiato di recente)
    - URL registro ufficiale
    """

    NAME     = "opencorporates"
    BASE_URL = "https://api.opencorporates.com/v0.4"
    REQUEST_DELAY = 2.0

    # Claim per cui questo connettore è rilevante
    RELEVANT_TYPES = {"revenue", "team_size", "funding", "partner_count"}

    def fetch(self, claim: dict, company_name: str) -> ConnectorResult:
        claim_id   = claim.get("id", "")
        claim_type = claim.get("type", "")

        # Esegui una volta sola per azienda — usa cache aggressiva
        cache_key = self.cache.make_key(self.NAME, company_name)
        cached = self.cache.get(cache_key)
        if cached:
            data = json.loads(cached)
            if data:
                return ConnectorResult(
                    connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                    found=True, data=data, confidence=0.90,
                    source_url=data.get("opencorporates_url", ""),
                    notes="Risultato da cache"
                )
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=False, notes="Azienda non trovata (da cache)"
            )

        data = self._search_company(company_name)

        if data:
            self.cache.set(cache_key, json.dumps(data))
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=True, data=data, confidence=0.90,
                source_url=data.get("opencorporates_url", ""),
                notes=f"Trovata su OpenCorporates: {data.get('current_status', 'N/D')}"
            )

        # Scrivi un risultato vuoto in cache per evitare query ripetute
        self.cache.set(cache_key, json.dumps({}))
        return ConnectorResult(
            connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
            found=False,
            notes=f"Azienda '{company_name}' non trovata su OpenCorporates"
        )

    def _search_company(self, company_name: str) -> Optional[dict]:
        """
        Cerca l'azienda su OpenCorporates.
        Strategia 1: API pubblica (funziona senza token per alcune query)
        Strategia 2: scraping della pagina di ricerca del sito web
        """
        # Tentativo 1: API (potrebbe funzionare senza token)
        try:
            time.sleep(self.REQUEST_DELAY)
            resp = self._get(
                f"{self.BASE_URL}/companies/search",
                params={
                    "q":                company_name,
                    "jurisdiction_code": "it",
                    "per_page":         5,
                    "order":            "score",
                }
            )
            if resp.status_code == 200:
                results = resp.json().get("results", {}).get("companies", [])
                if results:
                    best = self._pick_best_match(results, company_name)
                    if best:
                        return self._extract_data(best.get("company", {}))
        except Exception:
            pass

        # Tentativo 2: scraping pagina web OpenCorporates
        return self._scrape_opencorporates(company_name)

    def _scrape_opencorporates(self, company_name: str) -> Optional[dict]:
        """Fallback: scraping della pagina di ricerca pubblica di opencorporates.com"""
        try:
            time.sleep(self.REQUEST_DELAY)
            url  = f"https://opencorporates.com/companies?q={quote_plus(company_name)}&jurisdiction_code=it&type=company"
            resp = self._get(url)
            if resp.status_code != 200:
                return None

            soup    = BeautifulSoup(resp.text, "html.parser")
            results = soup.select("li.search-result")
            if not results:
                # Prova selettore alternativo
                results = soup.select("ul.companies li, .company-result")

            if not results:
                return None

            # Prendi il primo risultato
            first = results[0]
            link  = first.select_one("a[href*='/companies/it/']")
            if not link:
                return None

            company_url = "https://opencorporates.com" + link["href"]
            name        = link.get_text(strip=True)

            # Leggi lo stato dalla card
            status_tag  = first.select_one(".status, .inactive, span.label")
            inactive    = status_tag and "inactive" in (status_tag.get("class", []) + [status_tag.get_text(strip=True).lower()])

            # Ottieni numero registro dall'URL (es. /companies/it/12345678)
            company_number = link["href"].split("/")[-1]

            return {
                "name":               name,
                "company_number":     company_number,
                "jurisdiction_code":  "it",
                "company_type":       "",
                "incorporation_date": None,
                "dissolution_date":   None,
                "current_status":     "inactive" if inactive else "active",
                "status_normalized":  "cessata" if inactive else "attiva",
                "inactive":           inactive,
                "registered_address": "",
                "previous_names":     [],
                "opencorporates_url": company_url,
                "registry_url":       "",
                "source":             "opencorporates_scraping",
            }

        except Exception as e:
            log.warning(f"OpenCorporates scraping error per '{company_name}': {e}")
            return None

    def _pick_best_match(self, results: list, query: str) -> Optional[dict]:
        """Seleziona il risultato più pertinente dalla lista."""
        query_lower = query.lower()

        # Prima passa: cerca match esatto o molto vicino tra aziende attive
        for item in results:
            company = item.get("company", {})
            name    = company.get("name", "").lower()
            active  = not company.get("inactive", True)
            if active and (query_lower in name or name in query_lower):
                return item

        # Seconda passa: prendi il primo risultato attivo
        for item in results:
            company = item.get("company", {})
            if not company.get("inactive", True):
                return item

        # Fallback: primo risultato qualunque
        return results[0] if results else None

    def _extract_data(self, company: dict) -> dict:
        """Normalizza i dati di una company OpenCorporates."""
        # Stato
        inactive         = company.get("inactive", False)
        dissolution_date = company.get("dissolution_date")
        current_status   = company.get("current_status", "")

        if dissolution_date or inactive:
            status_normalized = "cessata"
        elif current_status and current_status.lower() in ("active", "attiva"):
            status_normalized = "attiva"
        elif current_status:
            status_normalized = current_status.lower()
        else:
            status_normalized = "sconosciuto"

        # Indirizzo
        addr = company.get("registered_address") or {}
        address_str = ", ".join(filter(None, [
            addr.get("street_address"),
            addr.get("locality"),
            addr.get("postal_code"),
            addr.get("country"),
        ]))

        # Nomi precedenti
        previous_names = [
            p.get("company_name", "")
            for p in company.get("previous_names", [])
        ]

        return {
            "name":               company.get("name", ""),
            "company_number":     company.get("company_number", ""),
            "jurisdiction_code":  company.get("jurisdiction_code", "it"),
            "company_type":       company.get("company_type", ""),
            "incorporation_date": company.get("incorporation_date"),
            "dissolution_date":   dissolution_date,
            "current_status":     current_status,
            "status_normalized":  status_normalized,
            "inactive":           inactive,
            "registered_address": address_str,
            "previous_names":     previous_names,
            "opencorporates_url": company.get("opencorporates_url", ""),
            "registry_url":       company.get("registry_url", ""),
            "source":             "opencorporates",
        }

# ─────────────────────────────────────────────
#  Connettore 5 — Wayback Machine
# ─────────────────────────────────────────────

class WaybackConnector(BaseConnector):
    """
    Recupera versioni storiche del sito web aziendale via Internet Archive.
    Utile per rilevare claim cambiate nel tempo senza spiegazione.
    API completamente gratuita.
    """

    NAME = "wayback"
    CDX_API = "https://web.archive.org/cdx/search/cdx"
    WAYBACK_BASE = "https://web.archive.org/web"
    REQUEST_DELAY = 2.0

    def fetch(self, claim: dict, company_name: str) -> ConnectorResult:
        claim_id   = claim.get("id", "")
        claim_type = claim.get("type", "")
        website    = claim.get("website_url", "")

        if not website:
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=False, notes="URL sito web non fornito"
            )

        domain = urlparse(website).netloc or website
        cache_key = self.cache.make_key(self.NAME, domain)
        cached = self.cache.get(cache_key)

        if cached:
            data = json.loads(cached)
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=True, data=data, confidence=0.80,
                source_url=f"{self.WAYBACK_BASE}/{domain}",
                notes="Risultato da cache"
            )

        snapshots = self._get_snapshots(domain)
        if not snapshots:
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=False,
                notes=f"Nessuno snapshot trovato per {domain}",
                confidence=0.0
            )

        # Recupera contenuto di snapshot chiave: più vecchio e più recente
        timeline = self._build_timeline(domain, snapshots)

        data = {
            "domain": domain,
            "total_snapshots": len(snapshots),
            "first_snapshot": snapshots[0]["timestamp"] if snapshots else None,
            "last_snapshot":  snapshots[-1]["timestamp"] if snapshots else None,
            "timeline": timeline,
        }

        self.cache.set(cache_key, json.dumps(data))

        return ConnectorResult(
            connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
            found=True, data=data, confidence=0.80,
            source_url=f"{self.WAYBACK_BASE}/{domain}",
            notes=f"Trovati {len(snapshots)} snapshot per {domain}"
        )

    def _get_snapshots(self, domain: str) -> list[dict]:
        """Recupera lista snapshot dal CDX API."""
        try:
            params = {
                "url": domain,
                "output": "json",
                "fl": "timestamp,statuscode,mimetype",
                "filter": "statuscode:200",
                "collapse": "timestamp:8",   # 1 snapshot per giorno
                "limit": 50,
                "from": "20200101",
            }
            resp = self._get(self.CDX_API, params=params)
            rows = resp.json()
            if not rows or len(rows) <= 1:
                return []
            headers = rows[0]
            return [dict(zip(headers, row)) for row in rows[1:]]
        except Exception as e:
            log.warning(f"Wayback CDX error: {e}")
            return []

    def _build_timeline(self, domain: str, snapshots: list[dict]) -> list[dict]:
        """
        Recupera il testo di 3 snapshot significativi:
        il più vecchio, uno a metà, il più recente.
        Estrae metriche quantitative per rilevare cambiamenti.
        """
        if not snapshots:
            return []

        indices = [0, len(snapshots) // 2, -1]
        selected = list({snapshots[i]["timestamp"] for i in indices})

        timeline = []
        for ts in selected[:3]:   # max 3 richieste
            url = f"{self.WAYBACK_BASE}/{ts}/{domain}"
            try:
                resp = self._get(url)
                soup = BeautifulSoup(resp.text, "html.parser")

                # Rimuovi nav/footer/script
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()

                text = soup.get_text(" ", strip=True)[:2000]
                metrics = self._extract_metrics(text)

                timeline.append({
                    "timestamp": ts,
                    "url": url,
                    "metrics": metrics,
                    "text_snippet": text[:300],
                })
            except Exception as e:
                log.warning(f"Wayback fetch error [{ts}]: {e}")

        return timeline

    @staticmethod
    def _extract_metrics(text: str) -> dict:
        """
        Estrae metriche quantitative dal testo della pagina storica.
        Cerca pattern come "300 partner", "€2M", "28 dipendenti".
        """
        metrics = {}

        patterns = {
            "partner_count": r"(\d+[\+]?)\s*(?:partner|strutture|operatori|clienti)",
            "revenue":        r"(?:€|EUR)\s*([\d,\.]+\s*(?:M|K|milion|mila)?)",
            "team_size":      r"(\d+)\s*(?:dipendenti|persone|collaboratori|team)",
            "funding":        r"(?:raccolto|round|finanziamento)[^\d]*(\d+[\d,\.]*\s*(?:M|K|milion)?)",
        }

        for key, pattern in patterns.items():
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                metrics[key] = m.group(1).strip()

        return metrics


# ─────────────────────────────────────────────
#  Data Collector — orchestratore
# ─────────────────────────────────────────────

class DataCollector:
    """
    Orchestratore principale del Modulo 2.
    Riceve le claim estratte dal ClaimExtractor e le distribuisce
    ai connettori appropriati.

    Utilizzo:
        collector = DataCollector(
            financial_data=result.financial_data,   # da ClaimExtractor
            crunchbase_api_key="",                  # opzionale, free tier
        )
        collection = collector.collect(
            company_name="MoveNow S.r.l.",
            claims=result.claims,                   # da ClaimExtractor
            website_url="https://movenow.it",
        )
        print(collection.summary())
    """

    # Quale connettore gestisce quale tipo di claim
    CLAIM_CONNECTOR_MAP = {
        "revenue":       ["bilancio"],        # ufficiocamerale aggiunto dinamicamente se P.IVA presente
        "partner_count": ["partner_website"],
        "funding":       ["crunchbase"],
        "team_size":     ["linkedin"],         # ufficiocamerale aggiunto dinamicamente se P.IVA presente
        "other":         [],
    }

    # Per tutte le claim, aggiungiamo sempre Wayback e OpenCorporates
    UNIVERSAL_CONNECTORS = ["wayback", "opencorporates"]

    def __init__(
        self,
        financial_data: Optional[dict] = None,
        crunchbase_api_key: str = "",
        cache: Optional[InMemoryCache] = None,
        linkedin_url: str = "",
        vat_number: str = "",
    ):
        shared_cache = cache or InMemoryCache()
        self.linkedin_url = linkedin_url.strip()
        self.vat_number   = vat_number.strip()

        self.connectors: dict[str, BaseConnector] = {
            "bilancio":        BilancioConnector(financial_data=financial_data, cache=shared_cache),
            "partner_website": PartnerWebsiteConnector(cache=shared_cache),
            "crunchbase":      CrunchbaseConnector(api_key=crunchbase_api_key, cache=shared_cache),
            "linkedin":        LinkedInConnector(cache=shared_cache),
            "ufficiocamerale": UfficioCameraleConnector(cache=shared_cache),
            "wayback":         WaybackConnector(cache=shared_cache),
            "opencorporates":  OpenCorporatesConnector(cache=shared_cache),
        }

    def collect(
        self,
        company_name: str,
        claims: list,
        website_url: str = "",
    ) -> CollectionResult:

        collection = CollectionResult(company_name=company_name)

        for claim in claims:
            # Converti in dict se è un dataclass
            claim_dict = claim if isinstance(claim, dict) else {
                "id": claim.id,
                "type": claim.type.value if hasattr(claim.type, "value") else claim.type,
                "text": claim.text,
                "normalized_value": claim.normalized_value,
                "website_url": website_url,
            }
            # Inietta sempre i metadati utente nel claim_dict per i connettori
            # (website_url va iniettato anche se claim è già un dict)
            if website_url:
                claim_dict = {**claim_dict, "website_url": website_url}
            if self.linkedin_url:
                claim_dict = {**claim_dict, "linkedin_url": self.linkedin_url}
            if self.vat_number:
                claim_dict = {**claim_dict, "vat_number": self.vat_number}

            claim_type = claim_dict.get("type", "other")
            connector_names = list(self.CLAIM_CONNECTOR_MAP.get(claim_type, []))

            # Wayback: solo se c'è un sito web
            if website_url and "wayback" not in connector_names:
                connector_names.append("wayback")

            # OpenCorporates: sempre (cerca per nome, non richiede altri input)
            if "opencorporates" not in connector_names:
                connector_names.append("opencorporates")

            # UfficioCamerale: solo se c'è la P.IVA
            if self.vat_number and "ufficiocamerale" not in connector_names:
                if claim_type in ("revenue", "team_size"):
                    connector_names.append("ufficiocamerale")

            log.info(f"Connettori per [{claim_type}]: {connector_names}")
            for name in connector_names:
                connector = self.connectors.get(name)
                if not connector:
                    log.warning(f"[{name}] Connettore non registrato — skip")
                    continue

                log.info(f"[{name}] Fetching claim {claim_dict.get('id')} ({claim_type})")
                try:
                    result = connector.fetch(claim_dict, company_name)
                    status = "✓ trovato" if result.found else f"✗ non trovato ({result.notes or ''})"
                    log.info(f"[{name}] {status}")
                    collection.results.append(result)
                except Exception as e:
                    log.error(f"[{name}] Errore fetch: {e}")
                    collection.errors.append(f"{name}: {e}")

        log.info(f"Collection completata:\n{collection.summary()}")
        return collection
