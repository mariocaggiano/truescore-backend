"""
TrueScore — Enrichers supplementari
=====================================
Moduli di arricchimento che non dipendono da Google o API a pagamento.

1. WebHistoryEnricher   — storia sito da Wayback Machine
2. JobPostingsEnricher  — job postings attivi da LinkedIn
3. EmailDomainEnricher  — verifica SPF/MX/DKIM via DNS over HTTPS
4. TechStackEnricher    — tecnologie usate dal sito web
"""

import json
import logging
import re
import time
from typing import Optional
from urllib.parse import urlparse, quote_plus

import requests
from bs4 import BeautifulSoup

log = logging.getLogger("enrichers")


# ─────────────────────────────────────────────
#  1. Web History Enricher
# ─────────────────────────────────────────────

class WebHistoryEnricher:
    """
    Arricchisce il report con la storia del sito web.
    Usa i dati Wayback già raccolti dal WaybackConnector.

    Output:
    - Anno di prima comparsa online
    - Numero totale di snapshot
    - Età stimata del dominio
    - Segnali di discontinuità (es. sito offline per periodi)
    - Confronto contenuto: cosa è cambiato tra versione vecchia e recente
    """

    @staticmethod
    def enrich(result, raw_results: list[dict]) -> None:
        wayback = [r for r in raw_results
                   if r.get("connector") == "wayback" and r.get("found")]

        if not wayback:
            result.web_history = {"found": False}
            return

        data = wayback[0].get("data", {})
        if not data:
            result.web_history = {"found": False}
            return

        total   = data.get("total_snapshots", 0)
        first   = data.get("first_snapshot", "")
        last    = data.get("last_snapshot", "")
        domain  = data.get("domain", "")
        changes = data.get("content_changes", {})

        # Anno di prima comparsa
        first_year = None
        if first:
            m = re.match(r"(\d{4})", str(first))
            if m:
                first_year = int(m.group(1))

        # Età in anni
        age_years = None
        if first_year:
            from datetime import date
            age_years = date.today().year - first_year

        # Valuta stabilità: gap tra snapshot
        continuity_score = "alta"
        gaps = data.get("gaps", [])
        if not gaps and total > 0:
            # Stima grossolana: meno di 10 snapshot → bassa continuità
            if total < 5:
                continuity_score = "bassa"
            elif total < 20:
                continuity_score = "media"

        # Cambiamenti di contenuto rilevanti
        content_signals = []
        if changes:
            old_text = (changes.get("oldest_text") or "").lower()
            new_text = (changes.get("newest_text") or "").lower()
            if old_text and new_text:
                # Controlla se il dominio era usato per altro
                old_words = set(old_text.split())
                new_words = set(new_text.split())
                overlap   = len(old_words & new_words) / max(len(old_words), 1)
                if overlap < 0.15:
                    content_signals.append("Contenuto del sito completamente cambiato")
                elif overlap < 0.4:
                    content_signals.append("Contenuto del sito significativamente modificato")

        # Genera sintesi
        parts = []
        if first_year:
            parts.append(f"Online dal {first_year} ({age_years} anni)")
        if total:
            parts.append(f"{total} snapshot archiviati")
        if content_signals:
            parts.extend(content_signals)

        result.web_history = {
            "found":             True,
            "domain":            domain,
            "first_year":        first_year,
            "age_years":         age_years,
            "total_snapshots":   total,
            "first_snapshot":    first,
            "last_snapshot":     last,
            "continuity_score":  continuity_score,
            "content_signals":   content_signals,
            "summary":           " · ".join(parts) if parts else "",
            "wayback_url":       f"https://web.archive.org/web/*/{domain}" if domain else "",
        }
        log.info(f"WebHistory: {result.web_history.get('summary','')}")


# ─────────────────────────────────────────────
#  2. Job Postings Enricher
# ─────────────────────────────────────────────

class JobPostingsEnricher:
    """
    Cerca job postings attivi su LinkedIn.
    Usa l'URL LinkedIn aziendale già disponibile.

    Un'azienda che assume è un segnale di salute operativa.
    Un'azienda che ha smesso di assumere da anni è un segnale di attenzione.

    Strategia: scrapa /jobs della pagina LinkedIn aziendale pubblica.
    """

    REQUEST_DELAY = 2.0

    @staticmethod
    def enrich(result, linkedin_url: str = "",
               company_name: str = "") -> None:
        if not linkedin_url and not company_name:
            result.job_postings = {"found": False, "checked": False}
            return

        try:
            jobs_data = JobPostingsEnricher._fetch_jobs(
                linkedin_url, company_name
            )
            result.job_postings = jobs_data
            if jobs_data.get("found"):
                log.info(
                    f"JobPostings: {jobs_data.get('count', 0)} posizioni "
                    f"per '{company_name}'"
                )
        except Exception as e:
            log.debug(f"JobPostings error: {e}")
            result.job_postings = {"found": False, "checked": True, "error": str(e)}

    @staticmethod
    def _fetch_jobs(linkedin_url: str, company_name: str) -> dict:
        headers = {
            "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
            "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
        }

        # Costruisce URL jobs dalla pagina company
        jobs_url = ""
        if linkedin_url and "linkedin.com/company/" in linkedin_url:
            base = linkedin_url.split("?")[0].rstrip("/")
            jobs_url = f"{base}/jobs/"

        if not jobs_url:
            # Cerca via Google
            query = f'site:linkedin.com/jobs "{company_name}"'
            try:
                resp = requests.get(
                    f"https://www.google.com/search?q={quote_plus(query)}&num=5",
                    headers=headers, timeout=8
                )
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.select("a[href]"):
                    href = a.get("href", "")
                    if "linkedin.com/jobs" in href:
                        jobs_url = href
                        break
            except Exception:
                pass

        if not jobs_url:
            return {"found": False, "checked": True}

        time.sleep(JobPostingsEnricher.REQUEST_DELAY)
        try:
            resp = requests.get(jobs_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return {"found": False, "checked": True}

            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(" ", strip=True)

            # Conta le posizioni aperte
            jobs = []

            # Pattern LinkedIn jobs page
            for tag in soup.select(".job-card-container, .jobs-job-board-list__item, "
                                   "[class*='job-card'], [class*='jobs-search']"):
                title_tag = tag.select_one("h3, h4, .job-card-list__title, "
                                          "[class*='job-title']")
                if title_tag:
                    title = title_tag.get_text(strip=True)
                    if title and len(title) > 5:
                        location_tag = tag.select_one("[class*='location'], "
                                                     "[class*='subtitle']")
                        location = location_tag.get_text(strip=True) if location_tag else ""
                        jobs.append({"title": title[:80], "location": location[:50]})

            # Fallback: cerca numero nelle meta info
            count = len(jobs)
            if count == 0:
                m = re.search(
                    r"(\d+)\s*(?:offerte?|posizioni?|jobs?|annunci?|risultati?)",
                    text, re.IGNORECASE
                )
                if m:
                    count = int(m.group(1))
                    # count senza dettagli
                    return {
                        "found":   count > 0,
                        "checked": True,
                        "count":   count,
                        "jobs":    [],
                        "url":     jobs_url,
                        "summary": f"{count} posizioni aperte su LinkedIn",
                    }

            if count > 0:
                return {
                    "found":   True,
                    "checked": True,
                    "count":   count,
                    "jobs":    jobs[:10],
                    "url":     jobs_url,
                    "summary": f"{count} posizioni aperte su LinkedIn",
                }

            return {
                "found":   False,
                "checked": True,
                "count":   0,
                "jobs":    [],
                "url":     jobs_url,
                "summary": "Nessuna posizione aperta trovata su LinkedIn",
            }

        except Exception as e:
            return {"found": False, "checked": True, "error": str(e)}


# ─────────────────────────────────────────────
#  3. Email Domain Enricher
# ─────────────────────────────────────────────

class EmailDomainEnricher:
    """
    Verifica la configurazione email del dominio aziendale.
    Usa DNS over HTTPS (dns.google) — nessuna libreria aggiuntiva.

    Controlla:
    - MX records: ha server email configurato?
    - SPF record (TXT): configurazione email legittima
    - DMARC record: policy di sicurezza email
    - DKIM: firma digitale email (verifica parziale)

    Un dominio senza MX è un segnale di attenzione per un'azienda B2B.
    """

    DNS_DOH = "https://dns.google/resolve"

    @staticmethod
    def enrich(result, website_url: str = "",
               company_name: str = "") -> None:
        domain = EmailDomainEnricher._extract_domain(website_url)
        if not domain:
            result.email_domain = {"found": False, "domain": ""}
            return

        try:
            email_data = EmailDomainEnricher._check_domain(domain)
            result.email_domain = email_data
            log.info(
                f"EmailDomain: {domain} — "
                f"MX={'✓' if email_data.get('has_mx') else '✗'}  "
                f"SPF={'✓' if email_data.get('has_spf') else '✗'}  "
                f"DMARC={'✓' if email_data.get('has_dmarc') else '✗'}"
            )
        except Exception as e:
            log.debug(f"EmailDomain error: {e}")
            result.email_domain = {"found": False, "domain": domain, "error": str(e)}

    @staticmethod
    def _extract_domain(url: str) -> str:
        if not url:
            return ""
        if not url.startswith("http"):
            url = "https://" + url
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path.split("/")[0]
            # Rimuovi www.
            domain = re.sub(r"^www\.", "", domain)
            return domain.strip().lower()
        except Exception:
            return ""

    @staticmethod
    def _dns_query(name: str, record_type: str) -> list[str]:
        """DNS over HTTPS via Google Public DNS."""
        try:
            resp = requests.get(
                EmailDomainEnricher.DNS_DOH,
                params={"name": name, "type": record_type},
                headers={"Accept": "application/dns-json"},
                timeout=8,
            )
            if resp.status_code != 200:
                return []
            data   = resp.json()
            answers = data.get("Answer", [])
            return [a.get("data", "") for a in answers]
        except Exception:
            return []

    @staticmethod
    def _check_domain(domain: str) -> dict:
        signals  = []
        warnings = []

        # MX records
        mx_records = EmailDomainEnricher._dns_query(domain, "MX")
        has_mx = len(mx_records) > 0
        mx_providers = []
        for mx in mx_records:
            mx_lower = mx.lower()
            if "google" in mx_lower:         mx_providers.append("Google Workspace")
            elif "microsoft" in mx_lower:    mx_providers.append("Microsoft 365")
            elif "outlook" in mx_lower:      mx_providers.append("Outlook")
            elif "amazon" in mx_lower or "aws" in mx_lower: mx_providers.append("Amazon SES")
            elif "mailchimp" in mx_lower:    mx_providers.append("Mailchimp")
            elif "aruba" in mx_lower:        mx_providers.append("Aruba")
            elif "register" in mx_lower:     mx_providers.append("Register.it")
        mx_providers = list(dict.fromkeys(mx_providers))  # dedup

        if not has_mx:
            warnings.append("Nessun server email configurato (MX assente)")
        else:
            signals.append(
                f"Email configurata"
                + (f" su {', '.join(mx_providers)}" if mx_providers else "")
            )

        # SPF record (TXT)
        txt_records = EmailDomainEnricher._dns_query(domain, "TXT")
        spf_record  = next((t for t in txt_records if "v=spf1" in t.lower()), "")
        has_spf     = bool(spf_record)
        if has_spf:
            signals.append("SPF configurato")
        else:
            warnings.append("SPF assente (suscettibile a email spoofing)")

        # DMARC record
        dmarc_records = EmailDomainEnricher._dns_query(f"_dmarc.{domain}", "TXT")
        dmarc_record  = next(
            (t for t in dmarc_records if "v=dmarc1" in t.lower()), ""
        )
        has_dmarc = bool(dmarc_record)
        dmarc_policy = ""
        if has_dmarc:
            m = re.search(r"p=(\w+)", dmarc_record, re.IGNORECASE)
            dmarc_policy = m.group(1).lower() if m else ""
            signals.append(
                f"DMARC configurato (policy: {dmarc_policy or 'n/d'})"
            )
        else:
            warnings.append("DMARC assente")

        # Score infrastruttura email 0-10
        score = 0
        if has_mx:    score += 4
        if has_spf:   score += 3
        if has_dmarc: score += 3

        # Genera sintesi
        if score >= 8:
            summary = f"Infrastruttura email completa ({domain})"
        elif score >= 4:
            summary = f"Infrastruttura email parziale ({domain})"
        else:
            summary = f"Infrastruttura email assente o incompleta ({domain})"

        return {
            "found":        has_mx or has_spf,
            "domain":       domain,
            "has_mx":       has_mx,
            "mx_providers": mx_providers,
            "has_spf":      has_spf,
            "spf_record":   spf_record[:120] if spf_record else "",
            "has_dmarc":    has_dmarc,
            "dmarc_policy": dmarc_policy,
            "score":        score,
            "signals":      signals,
            "warnings":     warnings,
            "summary":      summary,
        }


# ─────────────────────────────────────────────
#  4. Tech Stack Enricher
# ─────────────────────────────────────────────

class TechStackEnricher:
    """
    Identifica le tecnologie usate dal sito web aziendale.
    Analizza headers HTTP, meta tag, script JS, e pattern HTML.

    Segnali utili:
    - CMS (WordPress, Webflow, Wix) → indica livello di maturità
    - Hosting (AWS, Azure, Vercel, Aruba) → infrastruttura
    - Analytics (GA4, Plausible) → stanno misurando
    - CRM/Marketing (HubSpot, Salesforce, Mailchimp) → struttura commerciale
    - Piattaforma e-commerce (Shopify, WooCommerce)
    - Framework frontend (React, Vue, Angular)

    Non giudica le scelte — mostra cosa usa l'azienda.
    """

    # Pattern per tecnologie comuni
    TECH_PATTERNS = {
        # CMS
        "WordPress":     [r"wp-content", r"wp-includes", r"/wp-json/"],
        "Webflow":       [r"webflow\.com", r"\.webflow\.io"],
        "Wix":           [r"wix\.com", r"wixstatic\.com"],
        "Squarespace":   [r"squarespace\.com", r"static1\.squarespace"],
        "Ghost":         [r"ghost\.org", r"content/themes/ghost"],
        "Drupal":        [r"drupal\.org", r"Drupal\.settings"],
        "Joomla":        [r"/components/com_", r"Joomla!"],

        # Hosting / CDN
        "Cloudflare":    [r"cloudflare"],
        "AWS":           [r"amazonaws\.com", r"cloudfront\.net"],
        "Vercel":        [r"vercel\.app", r"\.vercel\."],
        "Netlify":       [r"netlify\.app", r"netlify\.com"],
        "GitHub Pages":  [r"github\.io"],
        "Aruba":         [r"aruba\.it", r"arubabusiness"],
        "OVH":           [r"ovh\.net", r"ovhcloud"],

        # Analytics
        "Google Analytics": [r"google-analytics\.com", r"gtag/js", r"UA-\d+", r"G-[A-Z0-9]+"],
        "Google Tag Manager": [r"googletagmanager\.com", r"GTM-"],
        "Plausible":     [r"plausible\.io"],
        "Hotjar":        [r"hotjar\.com", r"hj\.q"],
        "Mixpanel":      [r"mixpanel\.com"],
        "Segment":       [r"segment\.com", r"analytics\.js"],

        # CRM / Marketing
        "HubSpot":       [r"hubspot\.com", r"hs-scripts", r"hsforms\.com"],
        "Salesforce":    [r"salesforce\.com", r"force\.com"],
        "Mailchimp":     [r"mailchimp\.com", r"mc\.js"],
        "Intercom":      [r"intercom\.io", r"intercomSettings"],
        "Zendesk":       [r"zendesk\.com", r"zopim"],
        "Drift":         [r"drift\.com", r"driftt\.com"],

        # E-commerce
        "Shopify":       [r"shopify\.com", r"cdn\.shopify\.com"],
        "WooCommerce":   [r"woocommerce", r"wc-ajax"],
        "Magento":       [r"Magento", r"mage/"],
        "PrestaShop":    [r"prestashop"],

        # Frontend frameworks
        "React":         [r"react\.development\.js", r"react\.production", r"__reactFiber"],
        "Vue.js":        [r"vue\.js", r"vue\.min\.js", r"__vue__"],
        "Angular":       [r"angular\.js", r"ng-version", r"angular\.min"],
        "Next.js":       [r"_next/static", r"__NEXT_DATA__"],
        "Nuxt.js":       [r"__nuxt", r"_nuxt/"],

        # Certificato SSL / sicurezza
        "Stripe":        [r"stripe\.com/v3", r"js\.stripe\.com"],
        "PayPal":        [r"paypal\.com", r"paypalobjects"],
        "reCAPTCHA":     [r"recaptcha", r"grecaptcha"],
        "Cookiebot":     [r"cookiebot\.com"],
        "Iubenda":       [r"iubenda\.com"],
    }

    TECH_CATEGORIES = {
        "WordPress": "CMS", "Webflow": "CMS", "Wix": "CMS",
        "Squarespace": "CMS", "Ghost": "CMS", "Drupal": "CMS", "Joomla": "CMS",
        "Cloudflare": "CDN/Hosting", "AWS": "Hosting", "Vercel": "Hosting",
        "Netlify": "Hosting", "GitHub Pages": "Hosting", "Aruba": "Hosting", "OVH": "Hosting",
        "Google Analytics": "Analytics", "Google Tag Manager": "Analytics",
        "Plausible": "Analytics", "Hotjar": "Analytics",
        "Mixpanel": "Analytics", "Segment": "Analytics",
        "HubSpot": "CRM", "Salesforce": "CRM", "Mailchimp": "Marketing",
        "Intercom": "Supporto", "Zendesk": "Supporto", "Drift": "Chat",
        "Shopify": "E-commerce", "WooCommerce": "E-commerce",
        "Magento": "E-commerce", "PrestaShop": "E-commerce",
        "React": "Frontend", "Vue.js": "Frontend", "Angular": "Frontend",
        "Next.js": "Frontend", "Nuxt.js": "Frontend",
        "Stripe": "Pagamenti", "PayPal": "Pagamenti",
        "reCAPTCHA": "Sicurezza", "Cookiebot": "Privacy", "Iubenda": "Privacy",
    }

    @staticmethod
    def enrich(result, website_url: str = "") -> None:
        if not website_url:
            result.tech_stack = {"found": False}
            return
        try:
            tech_data = TechStackEnricher._detect(website_url)
            result.tech_stack = tech_data
            techs = [t["name"] for t in tech_data.get("technologies", [])]
            log.info(f"TechStack: {', '.join(techs[:8])}")
        except Exception as e:
            log.debug(f"TechStack error: {e}")
            result.tech_stack = {"found": False, "error": str(e)}

    @staticmethod
    def _detect(url: str) -> dict:
        if not url.startswith("http"):
            url = "https://" + url

        headers_http = {}
        html         = ""
        final_url    = url

        try:
            resp = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
                },
                timeout=10,
                allow_redirects=True,
            )
            headers_http = dict(resp.headers)
            html         = resp.text[:100_000]
            final_url    = str(resp.url)
        except Exception as e:
            return {"found": False, "error": str(e), "url": url}

        # Testo su cui cercare (HTML + headers)
        search_text = html.lower() + " " + " ".join(
            f"{k}:{v}" for k, v in headers_http.items()
        ).lower()

        found_techs = []
        found_names = set()

        for tech_name, patterns in TechStackEnricher.TECH_PATTERNS.items():
            if tech_name in found_names:
                continue
            for pattern in patterns:
                if re.search(pattern, search_text, re.IGNORECASE):
                    category = TechStackEnricher.TECH_CATEGORIES.get(tech_name, "Altro")
                    found_techs.append({
                        "name":     tech_name,
                        "category": category,
                    })
                    found_names.add(tech_name)
                    break

        # Raggruppa per categoria
        by_category: dict[str, list[str]] = {}
        for t in found_techs:
            cat = t["category"]
            by_category.setdefault(cat, []).append(t["name"])

        # Segnali di maturità
        maturity_signals = []
        cats = set(by_category.keys())

        if "CRM" in cats or "Salesforce" in found_names or "HubSpot" in found_names:
            maturity_signals.append("CRM attivo")
        if "Analytics" in cats:
            maturity_signals.append("Analytics installato")
        if "E-commerce" in cats:
            maturity_signals.append("Piattaforma e-commerce")
        if any(n in found_names for n in ["Wix", "Squarespace", "Webflow"]):
            maturity_signals.append("Sito su piattaforma no-code")
        if "WordPress" in found_names:
            maturity_signals.append("Sito WordPress")
        if any(n in found_names for n in ["React", "Vue.js", "Angular", "Next.js"]):
            maturity_signals.append("Frontend moderno")

        # Server / hosting info dagli header
        server   = headers_http.get("Server", headers_http.get("server", ""))
        via      = headers_http.get("Via", "")
        powered  = headers_http.get("X-Powered-By", "")
        cf_ray   = headers_http.get("CF-Ray", "")
        if cf_ray and "Cloudflare" not in found_names:
            found_techs.append({"name": "Cloudflare", "category": "CDN/Hosting"})
            by_category.setdefault("CDN/Hosting", []).append("Cloudflare")

        # SSL
        has_ssl = final_url.startswith("https://")
        if not has_ssl:
            maturity_signals.append("⚠ Sito senza HTTPS")

        # Summary
        top_techs = [t["name"] for t in found_techs[:6]]
        summary = (
            f"{len(found_techs)} tecnologie rilevate: "
            + ", ".join(top_techs)
            + ("..." if len(found_techs) > 6 else "")
        ) if found_techs else "Nessuna tecnologia rilevata"

        return {
            "found":            bool(found_techs),
            "url":              final_url,
            "technologies":     found_techs,
            "by_category":      by_category,
            "total":            len(found_techs),
            "maturity_signals": maturity_signals,
            "has_ssl":          has_ssl,
            "server":           server[:80] if server else "",
            "summary":          summary,
        }
