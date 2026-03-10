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

    def _get(self, url: str, headers: dict = None, params: dict = None) -> requests.Response:
        self._throttle()
        h = {"User-Agent": "TrueScore/1.0 (business verification tool; respectful crawler)"}
        if headers:
            h.update(headers)
        return requests.get(url, headers=h, params=params, timeout=self.timeout)

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
#  Connettore 2 — Overpass / OpenStreetMap
# ─────────────────────────────────────────────

class OverpassConnector(BaseConnector):
    """
    Verifica l'esistenza di strutture geografiche via OpenStreetMap.
    Usa l'API Overpass (gratuita, no key richiesta).
    """

    NAME = "overpass"
    ENDPOINT = "https://overpass-api.de/api/interpreter"
    REQUEST_DELAY = 2.0   # Overpass richiede più rispetto

    # Mapping tipo_struttura → tag OSM
    OSM_TAGS = {
        "bike_sharing":    [('amenity', 'bicycle_rental'), ('amenity', 'bicycle_parking')],
        "scooter":         [('amenity', 'scooter_rental')],
        "car_sharing":     [('amenity', 'car_sharing')],
        "charging_station":[('amenity', 'charging_station')],
        "mobility_hub":    [('amenity', 'bicycle_rental'), ('amenity', 'car_sharing')],
        "generic":         [('amenity', 'bicycle_rental')],
    }

    def fetch(self, claim: dict, company_name: str) -> ConnectorResult:
        claim_id   = claim.get("id", "")
        claim_type = claim.get("type", "")

        if claim_type != "partner_count":
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=False, notes="Overpass usato solo per claim partner_count"
            )

        declared_value = self._extract_number(claim.get("normalized_value") or claim.get("text", ""))
        structure_type = self._infer_structure_type(claim.get("text", "") + " " + company_name)
        country        = claim.get("country", "IT")

        cache_key = self.cache.make_key(self.NAME, structure_type, country)
        cached = self.cache.get(cache_key)
        if cached:
            data = json.loads(cached)
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=True, data=data, confidence=0.75,
                source_url=self.ENDPOINT,
                notes="Risultato da cache"
            )

        tags = self.OSM_TAGS.get(structure_type, self.OSM_TAGS["generic"])
        query = self._build_query(tags, country)

        try:
            resp = self._get(self.ENDPOINT, params={"data": query})
            resp.raise_for_status()
            osm_data = resp.json()

            elements = osm_data.get("elements", [])
            count = len(elements)

            # Campiona i primi 10 per cross-reference
            sample = [
                {
                    "id": el.get("id"),
                    "name": el.get("tags", {}).get("name", "senza nome"),
                    "lat": el.get("lat") or el.get("center", {}).get("lat"),
                    "lon": el.get("lon") or el.get("center", {}).get("lon"),
                }
                for el in elements[:10]
            ]

            data = {
                "osm_count": count,
                "declared_count": declared_value,
                "structure_type": structure_type,
                "sample": sample,
                "country": country,
            }

            self.cache.set(cache_key, json.dumps(data))

            # Confidence: OSM ha buona copertura in Italia ma non completa
            confidence = 0.70 if country == "IT" else 0.55

            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=True, data=data, confidence=confidence,
                source_url=self.ENDPOINT,
                notes=f"Trovate {count} strutture '{structure_type}' in {country} su OSM"
            )

        except Exception as e:
            log.error(f"Overpass error: {e}")
            return ConnectorResult(
                connector=self.NAME, claim_id=claim_id, claim_type=claim_type,
                found=False, error=str(e), confidence=0.0
            )

    def _build_query(self, tags: list[tuple], country: str) -> str:
        """Costruisce una query Overpass QL per i tag dati in un paese."""
        country_code = country.lower()
        filters = ""
        for key, val in tags:
            filters += f'  node["{key}"="{val}"](area.country);\n'
            filters += f'  way["{key}"="{val}"](area.country);\n'

        return f"""
[out:json][timeout:30];
area["ISO3166-1"="{country_code.upper()}"]["admin_level"="2"]->.country;
(
{filters}
);
out center count;
"""

    @staticmethod
    def _extract_number(text: str) -> Optional[int]:
        if not text:
            return None
        nums = re.findall(r"\d+", str(text))
        return int(nums[0]) if nums else None

    @staticmethod
    def _infer_structure_type(text: str) -> str:
        text_l = text.lower()
        if any(w in text_l for w in ["bici", "bike", "ciclismo", "cycling"]):
            return "bike_sharing"
        if any(w in text_l for w in ["scooter", "monopattino"]):
            return "scooter"
        if any(w in text_l for w in ["car sharing", "auto"]):
            return "car_sharing"
        if any(w in text_l for w in ["ricarica", "charging", "colonnina"]):
            return "charging_station"
        if any(w in text_l for w in ["mobilità", "mobility", "hub"]):
            return "mobility_hub"
        return "generic"


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

        data = self._scrape_company_headcount(company_name)

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

    def _scrape_company_headcount(self, company_name: str) -> Optional[dict]:
        """
        Strategia in due step:
        1. Cerca il profilo aziendale su Google (più affidabile di cercare su LI direttamente)
        2. Accede alla pagina pubblica e legge il headcount
        """
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
        "revenue":       ["bilancio"],
        "partner_count": ["overpass"],
        "funding":       ["crunchbase"],
        "team_size":     ["linkedin"],
        "other":         [],
    }

    # Per tutte le claim con website, aggiungiamo sempre Wayback
    UNIVERSAL_CONNECTORS = ["wayback"]

    def __init__(
        self,
        financial_data: Optional[dict] = None,
        crunchbase_api_key: str = "",
        cache: Optional[InMemoryCache] = None,
    ):
        shared_cache = cache or InMemoryCache()

        self.connectors: dict[str, BaseConnector] = {
            "bilancio":  BilancioConnector(financial_data=financial_data, cache=shared_cache),
            "overpass":  OverpassConnector(cache=shared_cache),
            "crunchbase": CrunchbaseConnector(api_key=crunchbase_api_key, cache=shared_cache),
            "linkedin":  LinkedInConnector(cache=shared_cache),
            "wayback":   WaybackConnector(cache=shared_cache),
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

            claim_type = claim_dict.get("type", "other")
            connector_names = self.CLAIM_CONNECTOR_MAP.get(claim_type, [])

            # Aggiungi Wayback se c'è un sito
            if website_url:
                for uc in self.UNIVERSAL_CONNECTORS:
                    if uc not in connector_names:
                        connector_names = connector_names + [uc]

            for name in connector_names:
                connector = self.connectors.get(name)
                if not connector:
                    continue

                log.info(f"[{name}] Fetching claim {claim_dict.get('id')} ({claim_type})")
                result = connector.fetch(claim_dict, company_name)
                collection.results.append(result)

        log.info(f"Collection completata:\n{collection.summary()}")
        return collection
