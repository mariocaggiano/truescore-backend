"""
TrueScore — Report Generator
==============================
Converte l'output del VerificationEngine in un PDF professionale.
Usa reportlab (costo zero, nessuna dipendenza esterna a pagamento).

Struttura del report:
  1. Copertina con Trust Score
  2. Executive Summary
  3. Claim Analysis (card per ogni claim)
  4. Red Flags (sezione dedicata alle discrepanze)
  5. Claim Non Verificabili
  6. Fonti & Disclaimer

Dipendenze: reportlab
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.flowables import Flowable

log = logging.getLogger("report_generator")

W, H = A4
MARGIN = 20 * mm
CONTENT_W = W - 2 * MARGIN


# ─────────────────────────────────────────────
#  Palette colori
# ─────────────────────────────────────────────

class C:
    NAVY        = colors.HexColor("#1A1A2E")
    NAVY_LIGHT  = colors.HexColor("#2D2D4E")
    CREAM       = colors.HexColor("#F4F3EF")
    WHITE       = colors.white
    RED         = colors.HexColor("#E53935")
    RED_LIGHT   = colors.HexColor("#FFEBEE")
    ORANGE      = colors.HexColor("#F57C00")
    ORANGE_LIGHT= colors.HexColor("#FFF3E0")
    GREEN       = colors.HexColor("#2E7D32")
    GREEN_LIGHT = colors.HexColor("#E8F5E9")
    GREY        = colors.HexColor("#757575")
    GREY_LIGHT  = colors.HexColor("#F5F5F5")
    BORDER      = colors.HexColor("#E0E0E0")
    TEXT        = colors.HexColor("#212121")
    TEXT_LIGHT  = colors.HexColor("#616161")
    ACCENT      = colors.HexColor("#4040FF")


VERDICT_CFG = {
    "discrepancy":       (C.RED,    C.RED_LIGHT,    "DISCREPANZA"),
    "warning":           (C.ORANGE, C.ORANGE_LIGHT, "ATTENZIONE"),
    "verified":          (C.GREEN,  C.GREEN_LIGHT,  "VERIFICATA"),
    "uncertain":         (C.GREY,   C.GREY_LIGHT,   "INCERTA"),
    "insufficient_data": (C.GREY,   C.GREY_LIGHT,   "DATI INSUFF."),
}

VERDICT_ICONS = {
    "discrepancy":       "X",
    "warning":           "!",
    "verified":          "OK",
    "uncertain":         "?",
    "insufficient_data": "-",
}


# ─────────────────────────────────────────────
#  Stili tipografici
# ─────────────────────────────────────────────

def build_styles() -> dict:
    def s(name, **kw) -> ParagraphStyle:
        return ParagraphStyle(name, **kw)

    return {
        "cover_company": s("cover_company",
            fontName="Helvetica-Bold", fontSize=26, textColor=C.WHITE,
            leading=32, spaceAfter=4),
        "cover_sub": s("cover_sub",
            fontName="Helvetica", fontSize=11, textColor=colors.HexColor("#9090B0"),
            leading=16, spaceAfter=8),
        "section_header": s("section_header",
            fontName="Helvetica-Bold", fontSize=9, textColor=C.NAVY,
            leading=13, spaceBefore=14, spaceAfter=6,
            borderPad=0),
        "body": s("body",
            fontName="Helvetica", fontSize=9.5, textColor=C.TEXT,
            leading=15, spaceAfter=5, alignment=TA_JUSTIFY),
        "body_italic": s("body_italic",
            fontName="Helvetica-Oblique", fontSize=9, textColor=C.TEXT_LIGHT,
            leading=14, spaceAfter=5, alignment=TA_JUSTIFY),
        "claim_type": s("claim_type",
            fontName="Helvetica-Bold", fontSize=9, textColor=C.TEXT, leading=13),
        "claim_text": s("claim_text",
            fontName="Helvetica", fontSize=8.5, textColor=C.TEXT_LIGHT, leading=13),
        "cell_label": s("cell_label",
            fontName="Helvetica-Bold", fontSize=7, textColor=C.TEXT_LIGHT,
            leading=10, spaceAfter=2),
        "cell_value": s("cell_value",
            fontName="Helvetica-Bold", fontSize=9, textColor=C.TEXT, leading=13),
        "reasoning": s("reasoning",
            fontName="Helvetica", fontSize=8, textColor=C.TEXT_LIGHT,
            leading=12),
        "disclaimer": s("disclaimer",
            fontName="Helvetica", fontSize=7.5, textColor=C.TEXT_LIGHT,
            leading=11, alignment=TA_JUSTIFY),
        "source_item": s("source_item",
            fontName="Helvetica", fontSize=8.5, textColor=C.TEXT, leading=13),
        "rf_title": s("rf_title",
            fontName="Helvetica-Bold", fontSize=10, textColor=C.RED,
            leading=14, spaceAfter=3),
        "unverif_label": s("unverif_label",
            fontName="Helvetica-Bold", fontSize=7, textColor=C.GREY,
            leading=10, alignment=TA_CENTER),
        "footer": s("footer",
            fontName="Helvetica", fontSize=6.5, textColor=colors.HexColor("#9090A0"),
            leading=10, alignment=TA_CENTER),
    }


# ─────────────────────────────────────────────
#  Flowable personalizzati
# ─────────────────────────────────────────────

class TrustBar(Flowable):
    def __init__(self, score: float, width: float = None):
        super().__init__()
        self.score = score
        self.width = width or CONTENT_W
        self.height = 8

    def wrap(self, *args):
        return self.width, self.height + 16

    def draw(self):
        pct   = max(0, min(1, self.score / 10.0))
        color = C.RED if self.score < 4 else (C.ORANGE if self.score < 6.5 else C.GREEN)

        # Track
        self.canv.setFillColor(colors.HexColor("#2D2D4E"))
        self.canv.roundRect(0, 8, self.width, self.height, 3, fill=1, stroke=0)

        # Fill
        if pct > 0:
            self.canv.setFillColor(color)
            self.canv.roundRect(0, 8, self.width * pct, self.height, 3, fill=1, stroke=0)

        # Labels
        self.canv.setFont("Helvetica", 6)
        self.canv.setFillColor(colors.HexColor("#7070A0"))
        self.canv.drawString(0, 0, "0")
        self.canv.drawCentredString(self.width / 2, 0, "5")
        self.canv.drawRightString(self.width, 0, "10")


# ─────────────────────────────────────────────
#  Page callbacks
# ─────────────────────────────────────────────

def _cover_cb(canvas, doc):
    canvas.saveState()
    # Navy header band
    canvas.setFillColor(C.NAVY)
    canvas.rect(0, H - 78*mm, W, 78*mm, fill=1, stroke=0)
    # Wordmark
    canvas.setFillColor(C.ACCENT)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(MARGIN, H - 16*mm, "TRUE")
    canvas.setFillColor(C.WHITE)
    canvas.drawString(MARGIN + 30, H - 16*mm, "SCORE")
    # Separator
    canvas.setStrokeColor(colors.HexColor("#2D2D4E"))
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, H - 20*mm, W - MARGIN, H - 20*mm)
    canvas.restoreState()


def _page_cb(canvas, doc):
    canvas.saveState()
    # Header
    canvas.setFillColor(C.NAVY)
    canvas.rect(0, H - 13*mm, W, 13*mm, fill=1, stroke=0)
    canvas.setFillColor(C.ACCENT)
    canvas.setFont("Helvetica-Bold", 7)
    canvas.drawString(MARGIN, H - 8.5*mm, "TRUE")
    canvas.setFillColor(C.WHITE)
    canvas.drawString(MARGIN + 22, H - 8.5*mm, "SCORE")
    canvas.setFillColor(colors.HexColor("#8080A0"))
    canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(W/2, H - 8.5*mm, doc.company_name)
    canvas.drawRightString(W - MARGIN, H - 8.5*mm, doc.report_id)
    # Footer
    canvas.setFillColor(C.BORDER)
    canvas.rect(0, 0, W, 9*mm, fill=1, stroke=0)
    canvas.setFillColor(C.GREY)
    canvas.setFont("Helvetica", 6)
    canvas.drawCentredString(
        W/2, 3*mm,
        f"Generato il {doc.generated_at}  |  Pagina {canvas.getPageNumber()}  |  "
        "Basato su fonti pubbliche — non costituisce parere legale o professionale"
    )
    canvas.restoreState()


# ─────────────────────────────────────────────
#  Config & Doc custom
# ─────────────────────────────────────────────

@dataclass
class ReportConfig:
    company_name: str
    sector: str = ""
    website_url: str = ""
    report_id: str = ""
    generated_at: str = ""
    sources: list = field(default_factory=list)

    def __post_init__(self):
        if not self.report_id:
            import hashlib, time
            h = hashlib.md5(f"{self.company_name}{time.time()}".encode()).hexdigest()[:8].upper()
            self.report_id = f"TS-{datetime.now().year}-{h}"
        if not self.generated_at:
            self.generated_at = datetime.now().strftime("%d %B %Y")


class TSDoc(BaseDocTemplate):
    def __init__(self, filename, config: ReportConfig, **kw):
        super().__init__(filename, **kw)
        self.company_name = config.company_name
        self.report_id    = config.report_id
        self.generated_at = config.generated_at


# ─────────────────────────────────────────────
#  Report Generator
# ─────────────────────────────────────────────

class ReportGenerator:
    """
    Modulo 4 — genera il PDF finale.

    Utilizzo:
        gen = ReportGenerator()
        path = gen.generate(
            verification_result=engine_result,
            config=ReportConfig(company_name="MoveNow S.r.l.", sector="Mobilita"),
            output_path="report_movenow.pdf"
        )
    """

    def __init__(self):
        self.S = build_styles()

    def generate(
        self,
        verification_result,
        config: ReportConfig,
        output_path: str = "truescore_report.pdf",
    ) -> str:

        doc = TSDoc(
            output_path, config=config, pagesize=A4,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN, bottomMargin=MARGIN,
        )

        cover_frame = Frame(MARGIN, MARGIN, CONTENT_W, H - 82*mm - MARGIN, id="cover")
        body_frame  = Frame(MARGIN, 11*mm, CONTENT_W, H - 26*mm, id="body")

        doc.addPageTemplates([
            PageTemplate(id="Cover",    frames=[cover_frame], onPage=_cover_cb),
            PageTemplate(id="Standard", frames=[body_frame],  onPage=_page_cb),
        ])

        story = []
        story += self._cover(verification_result, config)
        story.append(NextPageTemplate("Standard"))
        story.append(PageBreak())
        story += self._executive_summary(verification_result)
        story += self._claim_analysis(verification_result)

        if verification_result.red_flags:
            story += self._red_flags(verification_result)

        if verification_result.unverifiable:
            story += self._unverifiable(verification_result)

        story += self._sources_disclaimer(config, verification_result)

        doc.build(story)
        log.info(f"PDF generato: {output_path}")
        return output_path

    # ── Copertina ─────────────────────────────────────────────────────────

    def _cover(self, result, config: ReportConfig) -> list:
        S = self.S
        score = result.trust_score
        score_hex = "E53935" if score < 4 else ("F57C00" if score < 6.5 else "2E7D32")

        story = [Spacer(1, 26*mm)]

        # Azienda
        story.append(Paragraph(config.company_name, S["cover_company"]))
        meta = "  ·  ".join(filter(None, [config.sector, config.report_id, config.generated_at]))
        story.append(Paragraph(meta, S["cover_sub"]))
        story.append(Spacer(1, 5*mm))

        # Score + barra
        score_data = [[
            Paragraph(
                f'<font color="#9090B0" size="8">TRUST SCORE</font>',
                ParagraphStyle("sm", fontName="Helvetica", fontSize=8,
                               textColor=colors.HexColor("#9090B0"), leading=12)
            ),
            Paragraph(
                f'<font color="#{score_hex}" size="42"><b>{score:.1f}</b></font>'
                f'<font color="#606080" size="18">/10</font>',
                ParagraphStyle("sc", fontName="Helvetica-Bold", fontSize=42,
                               alignment=TA_RIGHT, leading=48)
            ),
        ]]
        st = Table(score_data, colWidths=[CONTENT_W*0.5, CONTENT_W*0.5])
        st.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "BOTTOM"),
            ("TOPPADDING", (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ]))
        story.append(st)
        story.append(Spacer(1, 3*mm))
        story.append(TrustBar(score, width=CONTENT_W))
        story.append(Spacer(1, 8*mm))

        # Stat strip
        stats = [
            ("Claim analizzate", len(result.verdicts)),
            ("Discrepanze",      len(result.red_flags)),
            ("Attenzione",       len(result.warnings_list)),
            ("Non verificabili", len(result.unverifiable)),
        ]
        stat_vals = [Paragraph(
            f'<font color="white" size="20"><b>{v}</b></font>',
            ParagraphStyle("sv", fontName="Helvetica-Bold", fontSize=20,
                           textColor=C.WHITE, alignment=TA_CENTER, leading=26)
        ) for _, v in stats]
        stat_keys = [Paragraph(k,
            ParagraphStyle("sk", fontName="Helvetica", fontSize=7,
                           textColor=colors.HexColor("#8080A0"), alignment=TA_CENTER,
                           leading=11)
        ) for k, _ in stats]

        stat_t = Table([stat_vals, stat_keys], colWidths=[CONTENT_W/4]*4)
        stat_t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C.NAVY_LIGHT),
            ("TOPPADDING",    (0,0), (-1,-1), 9),
            ("BOTTOMPADDING", (0,0), (-1,-1), 9),
            ("LINEAFTER",     (0,0), (2,1),   0.5, colors.HexColor("#2D2D4E")),
        ]))
        story.append(stat_t)
        return story

    # ── Executive Summary ─────────────────────────────────────────────────

    def _executive_summary(self, result) -> list:
        S = self.S
        score = result.trust_score
        score_hex = "E53935" if score < 4 else ("F57C00" if score < 6.5 else "2E7D32")

        story = [Spacer(1, 3*mm)]
        story.append(Paragraph("EXECUTIVE SUMMARY", S["section_header"]))
        story.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=C.BORDER, spaceAfter=5))

        story.append(Paragraph(
            f'<font color="#{score_hex}"><b>Trust Score: {score:.1f}/10 — '
            f'{result.trust_score_label}</b></font>',
            S["body"]
        ))
        story.append(Spacer(1, 2*mm))

        if result.red_flags:
            types = ", ".join(sorted({v.claim_type.replace("_"," ").title()
                                      for v in result.red_flags}))
            story.append(Paragraph(
                f"L'analisi ha rilevato <b>discrepanze significative</b> nelle aree: <b>{types}</b>. "
                f"I dettagli sono nella sezione Red Flags.", S["body"]
            ))

        if result.warnings_list:
            types = ", ".join(sorted({v.claim_type.replace("_"," ").title()
                                      for v in result.warnings_list}))
            story.append(Paragraph(
                f"Sono stati rilevati <b>segnali di attenzione</b> nelle aree: {types}.", S["body"]
            ))

        verified = [v for v in result.verdicts
                    if (v.verdict.value if hasattr(v.verdict,"value") else v.verdict) == "verified"]
        if verified:
            types = ", ".join(sorted({v.claim_type.replace("_"," ").title() for v in verified}))
            story.append(Paragraph(
                f"Le seguenti aree risultano <b>coerenti</b> con le fonti disponibili: {types}.",
                S["body"]
            ))

        if result.unverifiable:
            types = ", ".join(sorted({v.claim_type.replace("_"," ").title()
                                      for v in result.unverifiable}))
            story.append(Paragraph(
                f"Non e stato possibile verificare: {types}.", S["body"]
            ))

        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(
            "L'analisi e basata su fonti pubblicamente disponibili e sui documenti forniti "
            "dall'utente. I risultati devono essere integrati con verifiche dirette prima di "
            "assumere decisioni commerciali o di investimento.",
            S["body_italic"]
        ))
        return story

    # ── Claim Analysis ────────────────────────────────────────────────────

    def _claim_analysis(self, result) -> list:
        S = self.S
        story = [Spacer(1, 4*mm)]
        story.append(Paragraph("ANALISI CLAIM", S["section_header"]))
        story.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=C.BORDER, spaceAfter=4))

        for v in result.verdicts:
            story += self._claim_card(v)
            story.append(Spacer(1, 3*mm))

        return story

    def _claim_card(self, v) -> list:
        S = self.S
        verdict_str = v.verdict.value if hasattr(v.verdict, "value") else v.verdict
        color, bg, label = VERDICT_CFG.get(verdict_str, (C.GREY, C.GREY_LIGHT, verdict_str.upper()))
        icon  = VERDICT_ICONS.get(verdict_str, "?")
        type_label = v.claim_type.replace("_", " ").title()

        # ── Header row ────────────────────────────────────────────────────
        def white_p(txt, size=8, bold=True):
            fn = "Helvetica-Bold" if bold else "Helvetica"
            return Paragraph(f'<font color="white" size="{size}">{txt}</font>',
                             ParagraphStyle("wp", fontName=fn, fontSize=size,
                                            textColor=C.WHITE, alignment=TA_CENTER, leading=size+3))

        badge_p = Paragraph(
            f'<font color="white" size="7.5"><b>{label}</b></font>',
            ParagraphStyle("bp", fontName="Helvetica-Bold", fontSize=7.5,
                           textColor=C.WHITE, alignment=TA_CENTER, leading=10)
        )
        conf_p = Paragraph(
            f'Conf. {v.evidence_confidence:.0%}',
            ParagraphStyle("cp", fontName="Helvetica", fontSize=7.5,
                           textColor=C.TEXT_LIGHT, alignment=TA_RIGHT, leading=10)
        )

        hdr = Table([[
            white_p(icon, size=10),
            Paragraph(f"<b>{type_label}</b>", S["claim_type"]),
            Paragraph(v.claim_text, S["claim_text"]),
            badge_p,
            conf_p,
        ]], colWidths=[9*mm, 28*mm, CONTENT_W-9*mm-28*mm-24*mm-20*mm, 24*mm, 20*mm])

        hdr.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (0,0),   color),
            ("BACKGROUND",    (1,0), (2,0),   bg),
            ("BACKGROUND",    (3,0), (3,0),   color),
            ("BACKGROUND",    (4,0), (4,0),   bg),
            ("VALIGN",        (0,0), (-1,-1),  "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1),  5),
            ("BOTTOMPADDING", (0,0), (-1,-1),  5),
            ("LEFTPADDING",   (0,0), (-1,-1),  5),
            ("RIGHTPADDING",  (0,0), (-1,-1),  5),
        ]))

        # ── Detail rows ───────────────────────────────────────────────────
        def fmt(val):
            if val is None: return "—"
            try:
                f = float(val)
                if f >= 1_000_000: return f"EUR {f/1_000_000:.2f}M"
                if f >= 1_000:     return f"EUR {f:,.0f}"
                return f"{f:,.0f}"
            except (TypeError, ValueError):
                return str(val)

        def colored_val(val, c):
            txt = fmt(val)
            hex_c = self._hex(c)
            return Paragraph(f'<font color="#{hex_c}"><b>{txt}</b></font>',
                             ParagraphStyle("cv", fontName="Helvetica-Bold", fontSize=9,
                                            textColor=c, leading=13))

        sources_txt = ", ".join(v.sources_used) if v.sources_used else "—"
        col_w = [CONTENT_W*0.20, CONTENT_W*0.20, CONTENT_W*0.60]

        detail = Table([
            [Paragraph("DICHIARATO", S["cell_label"]),
             Paragraph("VERIFICATO",  S["cell_label"]),
             Paragraph("FONTI",       S["cell_label"])],
            [Paragraph(f"<b>{fmt(v.declared_value)}</b>", S["cell_value"]),
             colored_val(v.verified_value, color),
             Paragraph(sources_txt, S["claim_text"])],
            [Paragraph(v.reasoning, S["reasoning"]), "", ""],
        ], colWidths=col_w)

        detail.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C.WHITE),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
            ("SPAN",          (0,2), (2,2)),
            ("LINEBELOW",     (0,0), (-1,0), 0.3, C.BORDER),
            ("LINEBELOW",     (0,1), (-1,1), 0.3, C.BORDER),
            ("BOX",           (0,0), (-1,-1), 0.5, C.BORDER),
        ]))

        result = [hdr, detail]

        for flag in getattr(v, "flags", []):
            result.append(Paragraph(
                f"  >> {flag}",
                ParagraphStyle("fl", fontName="Helvetica-Oblique", fontSize=7.5,
                               textColor=color, leading=11, leftIndent=8)
            ))
        return result

    # ── Red Flags ─────────────────────────────────────────────────────────

    def _red_flags(self, result) -> list:
        S = self.S
        story = [Spacer(1, 4*mm)]
        story.append(Paragraph("RED FLAGS", S["section_header"]))
        story.append(HRFlowable(width=CONTENT_W, thickness=1.5, color=C.RED, spaceAfter=4))

        for v in result.red_flags:
            def fmt(val):
                if val is None: return "—"
                try:
                    f = float(val)
                    if f >= 1_000_000: return f"EUR {f/1_000_000:.2f}M"
                    if f >= 1_000:     return f"EUR {f:,.0f}"
                    return f"{f:,.0f}"
                except: return str(val)

            story.append(Paragraph(
                v.claim_type.replace("_"," ").title(), S["rf_title"]
            ))
            story.append(Paragraph(f"<b>Dichiarato:</b> {v.claim_text}", S["body"]))
            story.append(Spacer(1, 2*mm))

            if v.declared_value is not None or v.verified_value is not None:
                ctr = Table([[
                    Paragraph(
                        f'<b>Dichiarato</b><br/><font size="11">{fmt(v.declared_value)}</font>',
                        ParagraphStyle("rfd", fontName="Helvetica", fontSize=9,
                                       textColor=C.TEXT, leading=15, alignment=TA_CENTER)),
                    Paragraph(
                        f'<b>Verificato</b><br/>'
                        f'<font size="11" color="#E53935">{fmt(v.verified_value)}</font>',
                        ParagraphStyle("rfv", fontName="Helvetica", fontSize=9,
                                       textColor=C.RED, leading=15, alignment=TA_CENTER)),
                    Paragraph(
                        f'<b>Scarto</b><br/>'
                        f'<font size="11" color="#E53935">{v.magnitude:.0%}</font>',
                        ParagraphStyle("rfm", fontName="Helvetica", fontSize=9,
                                       textColor=C.RED, leading=15, alignment=TA_CENTER)),
                ]], colWidths=[CONTENT_W/3]*3)

                ctr.setStyle(TableStyle([
                    ("BOX",        (0,0), (-1,-1), 0.5, C.RED_LIGHT),
                    ("LINEAFTER",  (0,0), (1,0),   0.5, C.RED_LIGHT),
                    ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#FFF5F5")),
                    ("TOPPADDING",    (0,0), (-1,-1), 8),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                    ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
                ]))
                story.append(ctr)
                story.append(Spacer(1, 3*mm))

            story.append(Paragraph(v.reasoning, S["body"]))
            for flag in getattr(v, "flags", []):
                story.append(Paragraph(f"  - {flag}",
                    ParagraphStyle("rff", fontName="Helvetica-Oblique", fontSize=8,
                                   textColor=C.RED, leading=12, leftIndent=10)
                ))
            story.append(HRFlowable(width=CONTENT_W, thickness=0.3, color=C.RED_LIGHT, spaceAfter=4))

        return story

    # ── Unverifiable ──────────────────────────────────────────────────────

    def _unverifiable(self, result) -> list:
        S = self.S
        story = [Spacer(1, 3*mm)]
        story.append(Paragraph("CLAIM NON VERIFICABILI", S["section_header"]))
        story.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=C.BORDER, spaceAfter=3))
        story.append(Paragraph(
            "Le seguenti claim non hanno trovato riscontro nelle fonti disponibili. "
            "L'assenza di dati non implica necessariamente falsita: puo riflettere "
            "limiti delle fonti pubbliche o round non annunciati.",
            S["body_italic"]
        ))
        story.append(Spacer(1, 3*mm))

        for v in result.unverifiable:
            verdict_str = v.verdict.value if hasattr(v.verdict, "value") else v.verdict
            lbl = "INCERTA" if verdict_str == "uncertain" else "DATI INSUFF."
            row = Table([[
                Paragraph(lbl, S["unverif_label"]),
                Paragraph(f"<b>{v.claim_type.replace('_',' ').title()}</b>: {v.claim_text}", S["claim_text"]),
                Paragraph(v.reasoning, S["reasoning"]),
            ]], colWidths=[17*mm, 54*mm, CONTENT_W-71*mm])
            row.setStyle(TableStyle([
                ("BOX",        (0,0), (-1,-1), 0.5, C.BORDER),
                ("BACKGROUND", (0,0), (0,0),   C.GREY_LIGHT),
                ("BACKGROUND", (1,0), (-1,0),  C.WHITE),
                ("VALIGN",     (0,0), (-1,-1),  "TOP"),
                ("TOPPADDING",    (0,0), (-1,-1), 6),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                ("LEFTPADDING",   (0,0), (-1,-1), 6),
                ("RIGHTPADDING",  (0,0), (-1,-1), 6),
            ]))
            story.append(row)
            story.append(Spacer(1, 2*mm))

        return story

    # ── Fonti & Disclaimer ────────────────────────────────────────────────

    def _sources_disclaimer(self, config: ReportConfig, result) -> list:
        S = self.S
        story = [Spacer(1, 4*mm)]
        story.append(Paragraph("FONTI CONSULTATE", S["section_header"]))
        story.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=C.BORDER, spaceAfter=3))

        all_sources = set(config.sources)
        for v in result.verdicts:
            for src in getattr(v, "sources_used", []):
                all_sources.add(src)

        for i, src in enumerate(sorted(all_sources), 1):
            story.append(Paragraph(f"{i}.  {src}", S["source_item"]))

        story.append(Spacer(1, 5*mm))
        story.append(Paragraph("DISCLAIMER", S["section_header"]))
        story.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=C.BORDER, spaceAfter=3))

        story.append(Paragraph(
            f"Il presente report (ID: {config.report_id}) e stato generato da TrueScore "
            f"in data {config.generated_at} sulla base di fonti pubblicamente disponibili "
            f"e dei documenti forniti dall'utente. Qualora sia stato caricato un bilancio "
            f"aziendale, TrueScore non ha eseguito verifiche sull'autenticita del documento. "
            f"Il report non costituisce parere legale, finanziario o professionale e non deve "
            f"essere utilizzato come unica base per decisioni di investimento o accordi "
            f"commerciali. TrueScore declina ogni responsabilita per inesattezze nelle fonti "
            f"consultate. I dati sono aggiornati alla data di generazione.",
            S["disclaimer"]
        ))
        return story

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _hex(color) -> str:
        try:
            return f"{int(color.red*255):02X}{int(color.green*255):02X}{int(color.blue*255):02X}"
        except Exception:
            return "333333"
