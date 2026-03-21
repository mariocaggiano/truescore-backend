"""
TrueScore — Report Generator v2
================================
PDF professionale da management consulting.
Design: copertina forte, sezioni chiare, claim cards visive.

Dipendenze: reportlab
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    KeepTogether,
)
from reportlab.platypus.flowables import Flowable

log = logging.getLogger("report_generator")

W, H = A4
MARGIN = 18 * mm
CW     = W - 2 * MARGIN


# ── Palette ──────────────────────────────────────────────────────────────────

class C:
    NAVY        = colors.HexColor("#0A0D18")
    NAVY_MID    = colors.HexColor("#10142A")
    NAVY_LIGHT  = colors.HexColor("#1A1F3A")
    BORDER      = colors.HexColor("#252B4A")
    ACCENT      = colors.HexColor("#3A5FD9")
    WHITE       = colors.HexColor("#EEE9E0")
    WHITE_DIM   = colors.HexColor("#8A8799")
    RED         = colors.HexColor("#C23B22")
    RED_BG      = colors.HexColor("#2A1010")
    ORANGE      = colors.HexColor("#B05A20")
    ORANGE_BG   = colors.HexColor("#2A1A08")
    GREEN       = colors.HexColor("#2A7A4A")
    GREEN_BG    = colors.HexColor("#0A2018")
    GREY        = colors.HexColor("#555870")
    GREY_BG     = colors.HexColor("#181A28")
    TEXT        = colors.HexColor("#EEE9E0")
    TEXT_DIM    = colors.HexColor("#8A8799")
    PAGE_BG     = colors.HexColor("#0D0F1C")


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_color(score):
    if score < 0:   return C.WHITE_DIM
    if score < 4:   return C.RED
    if score < 6.5: return C.ORANGE
    return C.GREEN

def score_hex(score):
    if score < 0:   return "8A8799"
    if score < 4:   return "C23B22"
    if score < 6.5: return "B05A20"
    return "2A7A4A"

def verdict_colors(verdict):
    m = {
        "discrepancy":       (C.RED,      C.RED_BG,    "C23B22"),
        "warning":           (C.ORANGE,   C.ORANGE_BG, "B05A20"),
        "verified":          (C.GREEN,    C.GREEN_BG,  "2A7A4A"),
        "uncertain":         (C.WHITE_DIM, C.GREY_BG,  "555870"),
        "insufficient_data": (C.WHITE_DIM, C.GREY_BG,  "555870"),
    }
    return m.get(verdict, (C.WHITE_DIM, C.GREY_BG, "555870"))

VERDICT_LABELS = {
    "discrepancy":       "DISCREPANZA",
    "warning":           "ATTENZIONE",
    "verified":          "VERIFICATA",
    "uncertain":         "INCERTA",
    "insufficient_data": "DATI INSUFF.",
}

VERDICT_ICONS = {
    "discrepancy":       "X",
    "warning":           "!",
    "verified":          "OK",
    "uncertain":         "?",
    "insufficient_data": "-",
}

TYPE_LABELS = {
    "revenue":       "Ricavi",
    "partner_count": "Partner / Strutture",
    "funding":       "Funding",
    "team_size":     "Team",
    "other":         "Altro",
    "legal_status":  "Stato Legale",
    "news_flag":     "Segnalazione News",
}

def fmt_val(val):
    if val is None: return "n.d."
    try:
        n = float(val)
        if n >= 1_000_000: return "EUR {:.2f}M".format(n / 1_000_000)
        if n >= 1_000:     return "EUR {:,.0f}".format(n)
        return "{:.0f}".format(n)
    except Exception:
        return str(val)


# ── Custom Flowables ──────────────────────────────────────────────────────────

class HLine(Flowable):
    def __init__(self, width, color=None, thickness=0.4):
        super().__init__()
        self.line_width = width
        self.line_color = color or C.BORDER
        self.thickness  = thickness
        self.height     = self.thickness + 2

    def draw(self):
        self.canv.saveState()
        self.canv.setStrokeColor(self.line_color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.thickness, self.line_width, self.thickness)
        self.canv.restoreState()


class TrustBar(Flowable):
    def __init__(self, score, width):
        super().__init__()
        self.score = score
        self.bar_width = width
        self.height = 7

    def draw(self):
        self.canv.saveState()
        self.canv.setFillColor(C.BORDER)
        self.canv.roundRect(0, 0, self.bar_width, self.height, 3, fill=1, stroke=0)
        if self.score >= 0:
            pct    = min(max(self.score / 10, 0), 1)
            fill_w = self.bar_width * pct
            self.canv.setFillColor(score_color(self.score))
            self.canv.roundRect(0, 0, fill_w, self.height, 3, fill=1, stroke=0)
        self.canv.restoreState()


class SectionLabel(Flowable):
    def __init__(self, text, width):
        super().__init__()
        self.text        = text.upper()
        self.label_width = width
        self.height      = 18

    def draw(self):
        self.canv.saveState()
        self.canv.setFillColor(C.ACCENT)
        self.canv.rect(0, 7, 18, 2, fill=1, stroke=0)
        self.canv.setFont("Helvetica-Bold", 7.5)
        self.canv.drawString(24, 5, self.text)
        self.canv.restoreState()


# ── Stili ─────────────────────────────────────────────────────────────────────

def build_styles():
    def S(name, **kw):
        defaults = dict(fontName="Helvetica", fontSize=9,
                        textColor=C.TEXT, leading=14, spaceAfter=0)
        defaults.update(kw)
        return ParagraphStyle(name, **defaults)

    return {
        "cover_company": S("cover_company", fontName="Times-Bold", fontSize=26,
                           textColor=C.TEXT, leading=32, spaceAfter=4),
        "cover_meta":    S("cover_meta",    fontName="Helvetica", fontSize=8,
                           textColor=C.WHITE_DIM, leading=13),
        "h1":  S("h1",  fontName="Times-Bold", fontSize=16,
                 textColor=C.TEXT, leading=20, spaceAfter=6),
        "body":    S("body",    fontName="Helvetica", fontSize=8.5,
                     textColor=C.TEXT_DIM, leading=13),
        "body_sm": S("body_sm", fontName="Helvetica", fontSize=7.5,
                     textColor=C.TEXT_DIM, leading=11),
        "mono":    S("mono",    fontName="Courier", fontSize=7,
                     textColor=C.WHITE_DIM, leading=10),
        "disclaimer": S("disclaimer", fontName="Helvetica", fontSize=6.5,
                        textColor=C.WHITE_DIM, leading=10),
        "small":      S("small",      fontName="Helvetica", fontSize=7,
                        textColor=C.WHITE_DIM, leading=10),
    }


# ── Page callbacks ────────────────────────────────────────────────────────────

def _cover_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(C.NAVY)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # Accent top band
    canvas.setFillColor(C.ACCENT)
    canvas.rect(0, H - 3, W, 3, fill=1, stroke=0)
    # Left accent stripe
    canvas.setFillColor(C.BORDER)
    canvas.rect(0, 0, 1.5, H, fill=1, stroke=0)
    # Footer
    canvas.setFillColor(C.WHITE_DIM)
    canvas.setFont("Helvetica", 6)
    canvas.drawRightString(W - MARGIN, 6*mm, doc.report_id)
    canvas.restoreState()


def _body_page(canvas, doc):
    canvas.saveState()
    # Page background
    canvas.setFillColor(C.PAGE_BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # Header
    canvas.setFillColor(C.NAVY_MID)
    canvas.rect(0, H - 11*mm, W, 11*mm, fill=1, stroke=0)
    canvas.setFillColor(C.ACCENT)
    canvas.rect(0, H - 11*mm, W, 0.4, fill=1, stroke=0)
    # Brand
    canvas.setFillColor(C.ACCENT)
    canvas.setFont("Times-BoldItalic", 9.5)
    canvas.drawString(MARGIN, H - 6.5*mm, "True")
    canvas.setFillColor(C.TEXT)
    canvas.setFont("Times-Bold", 9.5)
    canvas.drawString(MARGIN + 18, H - 6.5*mm, "Score")
    # Company in header
    canvas.setFillColor(C.WHITE_DIM)
    canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(W / 2, H - 6.5*mm, doc.company_name.upper())
    # Page number
    canvas.drawRightString(W - MARGIN, H - 6.5*mm,
                           "{} | P.{}".format(doc.report_id, canvas.getPageNumber()))
    # Footer
    canvas.setFillColor(C.NAVY_MID)
    canvas.rect(0, 0, W, 9*mm, fill=1, stroke=0)
    canvas.setFillColor(C.BORDER)
    canvas.rect(0, 9*mm, W, 0.3, fill=1, stroke=0)
    canvas.setFillColor(C.WHITE_DIM)
    canvas.setFont("Helvetica", 5.5)
    canvas.drawCentredString(
        W / 2, 3*mm,
        "Generato il {}  -  Non costituisce parere legale o finanziario. "
        "Basato su fonti pubblicamente disponibili.".format(doc.generated_at)
    )
    canvas.restoreState()


# ── Config & Doc ──────────────────────────────────────────────────────────────

@dataclass
class ReportConfig:
    company_name: str
    sector:       str  = ""
    website_url:  str  = ""
    report_id:    str  = ""
    generated_at: str  = ""
    sources:      list = field(default_factory=list)

    def __post_init__(self):
        if not self.report_id:
            import hashlib, time
            h = hashlib.md5("{}{}".format(self.company_name, time.time()).encode()).hexdigest()[:8].upper()
            self.report_id = "TS-{}-{}".format(datetime.now().year, h)
        if not self.generated_at:
            self.generated_at = datetime.now().strftime("%d %B %Y")


class TSDoc(BaseDocTemplate):
    def __init__(self, filename, config: ReportConfig, **kw):
        super().__init__(filename, **kw)
        self.company_name = config.company_name
        self.report_id    = config.report_id
        self.generated_at = config.generated_at


# ── Report Generator ──────────────────────────────────────────────────────────

class ReportGenerator:

    def __init__(self):
        self.S = build_styles()

    def generate(self, verification_result, config: ReportConfig,
                 output_path: str = "truescore_report.pdf") -> str:

        doc = TSDoc(output_path, config=config, pagesize=A4,
                    leftMargin=MARGIN, rightMargin=MARGIN,
                    topMargin=MARGIN, bottomMargin=MARGIN)

        cover_frame = Frame(MARGIN, MARGIN, CW, H - 2*MARGIN, id="cover")
        body_frame  = Frame(MARGIN, 11*mm, CW, H - 25*mm, id="body")

        doc.addPageTemplates([
            PageTemplate(id="Cover",    frames=[cover_frame], onPage=_cover_page),
            PageTemplate(id="Standard", frames=[body_frame],  onPage=_body_page),
        ])

        story = []
        story += self._cover(verification_result, config)
        story.append(NextPageTemplate("Standard"))
        story.append(PageBreak())
        story += self._executive_summary(verification_result)
        story += self._claim_analysis(verification_result)

        if verification_result.red_flags:
            story += self._red_flags_section(verification_result)

        if verification_result.unverifiable:
            story += self._unverifiable_section(verification_result)

        kp = getattr(verification_result, "key_people", None)
        if kp and kp.get("found"):
            story += self._people_section(kp)

        nf = getattr(verification_result, "news_flags", None)
        if nf:
            story += self._news_section(nf)

        ls = getattr(verification_result, "legal_status", None)
        if ls:
            story += self._legal_section(ls)

        story += self._sources_disclaimer(config)
        doc.build(story)
        log.info("PDF generato: {}".format(output_path))
        return output_path

    # ── Copertina ─────────────────────────────────────────────────────────────

    def _cover(self, result, config):
        S  = self.S
        sc = result.trust_score
        story = [Spacer(1, 12*mm)]

        # Brand
        story.append(Paragraph(
            '<font color="#3A5FD9">True</font><font color="#EEE9E0">Score</font>',
            ParagraphStyle("brand", fontName="Times-BoldItalic", fontSize=13,
                           textColor=C.ACCENT, leading=18)
        ))
        story.append(Paragraph(
            "BUSINESS VERIFICATION INTELLIGENCE",
            ParagraphStyle("tag", fontName="Helvetica-Bold", fontSize=6.5,
                           textColor=C.WHITE_DIM, leading=10, spaceAfter=16)
        ))
        story.append(HLine(CW, C.BORDER, 0.5))
        story.append(Spacer(1, 10*mm))

        # Azienda
        story.append(Paragraph(config.company_name, S["cover_company"]))
        meta_parts = [x for x in [config.sector, config.website_url,
                                    config.report_id, config.generated_at] if x]
        story.append(Paragraph("  |  ".join(meta_parts), S["cover_meta"]))
        story.append(Spacer(1, 10*mm))

        # Score
        score_str   = "N/D" if sc < 0 else "{:.1f}".format(sc)
        score_label = self._score_label(sc)
        score_data  = [[
            Paragraph(
                '<font color="#8A8799" size="8">TRUST SCORE</font><br/>'
                '<font color="#8A8799" size="8.5">{}</font>'.format(score_label),
                ParagraphStyle("sl", fontName="Helvetica", fontSize=8,
                               textColor=C.WHITE_DIM, leading=14)
            ),
            Paragraph(
                '<font color="#{}" size="48"><b>{}</b></font>'.format(score_hex(sc), score_str)
                + ('' if sc < 0 else '<font color="#555870" size="20">/10</font>'),
                ParagraphStyle("sn", fontName="Times-Bold", fontSize=48,
                               alignment=TA_RIGHT, leading=52)
            ),
        ]]
        t = Table(score_data, colWidths=[CW * 0.55, CW * 0.45])
        t.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "BOTTOM"),
            ("TOPPADDING", (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ]))
        story.append(t)
        story.append(Spacer(1, 3*mm))
        story.append(TrustBar(sc, CW))
        story.append(Spacer(1, 10*mm))

        # Stats
        n_disc  = len(result.red_flags) if result.red_flags else 0
        n_total = len(result.verdicts)
        n_unver = len(result.unverifiable) if result.unverifiable else 0

        def stat_cell(num, label, col):
            return Paragraph(
                '<font color="{}" size="20"><b>{}</b></font><br/>'
                '<font color="#8A8799" size="7">{}</font>'.format(col, num, label.upper()),
                ParagraphStyle("stat", fontName="Helvetica-Bold", fontSize=20,
                               textColor=colors.HexColor(col), leading=24)
            )

        stats = Table([[
            stat_cell(n_total, "Claim",       "#EEE9E0"),
            stat_cell(n_disc,  "Discrepanze", "#C23B22" if n_disc else "#555870"),
            stat_cell(n_unver, "Non verif.",  "#555870"),
        ]], colWidths=[CW/3]*3)
        stats.setStyle(TableStyle([
            ("TOPPADDING",    (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
            ("RIGHTPADDING",  (0,0), (-1,-1), 0),
        ]))
        story.append(stats)
        story.append(Spacer(1, 8*mm))
        story.append(HLine(CW, C.BORDER, 0.5))
        story.append(Spacer(1, 6*mm))

        # Sintesi rapida
        if n_disc > 0:
            story.append(Paragraph(
                "{} discrepanza/e rilevata/e tra dichiarazioni e fonti verificate. "
                "Consultare la sezione Red Flags.".format(n_disc),
                ParagraphStyle("warn", fontName="Helvetica", fontSize=9,
                               textColor=C.RED, leading=13)
            ))
        elif sc >= 7.5:
            story.append(Paragraph(
                "Nessuna discrepanza significativa rilevata. "
                "Le dichiarazioni risultano coerenti con le fonti pubbliche.",
                ParagraphStyle("ok", fontName="Helvetica", fontSize=9,
                               textColor=C.GREEN, leading=13)
            ))
        else:
            story.append(Paragraph(
                "Analisi completata. Consultare il dettaglio nel corpo del report.",
                ParagraphStyle("neutral", fontName="Helvetica", fontSize=9,
                               textColor=C.WHITE_DIM, leading=13)
            ))
        return story

    # ── Executive Summary ──────────────────────────────────────────────────────

    def _executive_summary(self, result):
        S = self.S
        sc = result.trust_score
        story = [Spacer(1, 2*mm), SectionLabel("Executive Summary", CW), Spacer(1, 4*mm)]

        story.append(Paragraph(
            'Trust Score: <font color="#{}" size="14"><b>{}</b></font>'
            '  <font color="#8A8799" size="8">{}</font>'.format(
                score_hex(sc),
                "N/D" if sc < 0 else "{:.1f}/10".format(sc),
                self._score_label(sc)
            ),
            ParagraphStyle("es", fontName="Times-Bold", fontSize=11,
                           textColor=C.TEXT, leading=18, spaceAfter=6)
        ))

        n_disc  = len(result.red_flags) if result.red_flags else 0
        n_total = len(result.verdicts)
        n_verif = sum(1 for v in result.verdicts
                      if str(getattr(getattr(v,"verdict",None),"value",
                                     getattr(v,"verdict",""))
                              ) == "verified")
        parts = []
        if sc < 0:
            parts.append("Non e stato possibile calcolare un Trust Score per "
                         "insufficienza di valori numerici confrontabili.")
        else:
            if n_disc:
                parts.append("{} discrepanza/e significativa/e rilevata/e.".format(n_disc))
            if n_verif:
                parts.append("{} claim su {} coerenti con le fonti.".format(n_verif, n_total))
        if parts:
            story.append(Paragraph(" ".join(parts), S["body"]))

        story.append(Spacer(1, 4*mm))
        story.append(HLine(CW, C.BORDER))
        return story

    # ── Claim Analysis ─────────────────────────────────────────────────────────

    def _claim_analysis(self, result):
        story = [Spacer(1, 4*mm), SectionLabel("Analisi Claim", CW), Spacer(1, 3*mm)]
        for v in result.verdicts:
            story.append(self._claim_card(v))
            story.append(Spacer(1, 3*mm))
        return story

    def _claim_card(self, v):
        verdict_str = str(getattr(getattr(v,"verdict",None),"value",
                                  getattr(v,"verdict","")))
        fg, bg, bh  = verdict_colors(verdict_str)
        vlabel      = VERDICT_LABELS.get(verdict_str, verdict_str.upper())
        vicon       = VERDICT_ICONS.get(verdict_str, "-")
        ctype       = getattr(v, "claim_type", "other")
        tlabel      = TYPE_LABELS.get(ctype, ctype.upper())

        declared   = getattr(v, "declared_value", None)
        verified   = getattr(v, "verified_value", None)
        magnitude  = getattr(v, "magnitude", 0) or 0
        confidence = getattr(v, "evidence_confidence", 0) or 0
        reasoning  = getattr(v, "reasoning", "") or ""
        sources    = getattr(v, "sources_used", []) or []
        cons       = getattr(v, "sources_consulted", []) or []
        claim_text = getattr(v, "claim_text", "") or ""

        # Header
        hdr = Table([[
            Paragraph(
                '<font color="#8A8799" size="7">{}</font><br/>'
                '<b>{}</b>'.format(tlabel, claim_text[:88]),
                ParagraphStyle("ch", fontName="Helvetica", fontSize=8.5,
                               textColor=C.TEXT, leading=13)
            ),
            Paragraph(
                '<font color="#{}">[{}] {}</font>'.format(bh, vicon, vlabel),
                ParagraphStyle("vb", fontName="Helvetica-Bold", fontSize=7.5,
                               textColor=fg, leading=10, alignment=TA_RIGHT)
            ),
        ]], colWidths=[CW - 36*mm, 36*mm])
        hdr.setStyle(TableStyle([
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ]))

        # Values
        def vcell(label, val, col):
            return Paragraph(
                '<font color="#555870" size="6">{}</font><br/>'
                '<font color="{}" size="9"><b>{}</b></font>'.format(label, col, val),
                ParagraphStyle("vc", fontName="Helvetica-Bold", fontSize=9,
                               textColor=C.TEXT, leading=13)
            )

        sc_hex = "#" + bh
        vals = Table([[
            vcell("DICHIARATO", fmt_val(declared), "#8A8799"),
            vcell("VERIFICATO",  fmt_val(verified),  sc_hex),
            vcell("SCARTO",     "{}%".format(int(magnitude*100)) if magnitude else "n.d.", sc_hex),
            vcell("CONF.",      "{}%".format(int(confidence*100)), "#8A8799"),
        ]], colWidths=[CW/4]*4)
        vals.setStyle(TableStyle([
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
            ("RIGHTPADDING",  (0,0), (-1,-1), 0),
            ("LINEABOVE",     (0,0), (-1,0),  0.3, C.BORDER),
        ]))

        rows = [[hdr], [vals]]

        if reasoning:
            rows.append([Paragraph(reasoning[:260],
                ParagraphStyle("r", fontName="Helvetica", fontSize=7.5,
                               textColor=C.TEXT_DIM, leading=11))])

        src_parts = []
        for s in sources: src_parts.append("+ " + s)
        for s in cons:
            if s not in sources: src_parts.append("- " + s)
        if src_parts:
            rows.append([Paragraph("  ".join(src_parts[:6]),
                ParagraphStyle("s", fontName="Courier", fontSize=6.5,
                               textColor=colors.HexColor("#" + bh), leading=10))])

        card = Table(rows, colWidths=[CW])
        card.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#" + bh + "18")),
            ("LINEAFTER",     (0,0), (0,-1),  2, colors.HexColor("#" + bh)),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("RIGHTPADDING",  (0,0), (-1,-1), 8),
            ("TOPPADDING",    (0,0), (0,0),   8),
            ("BOTTOMPADDING", (0,-1),(-1,-1), 8),
            ("TOPPADDING",    (0,1), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-2), 0),
        ]))
        return KeepTogether([card])

    # ── Red Flags ──────────────────────────────────────────────────────────────

    def _red_flags_section(self, result):
        story = [Spacer(1, 4*mm), SectionLabel("Red Flags", CW), Spacer(1, 3*mm)]
        for v in result.red_flags:
            ctype      = getattr(v, "claim_type", "other")
            tlabel     = TYPE_LABELS.get(ctype, ctype.upper())
            declared   = getattr(v, "declared_value", None)
            verified   = getattr(v, "verified_value", None)
            magnitude  = getattr(v, "magnitude", 0) or 0
            reasoning  = getattr(v, "reasoning", "") or ""
            claim_text = getattr(v, "claim_text", "") or ""

            def vcell(label, val, col):
                return Paragraph(
                    '<font color="#555870" size="6">{}</font><br/>'
                    '<font color="{}" size="9"><b>{}</b></font>'.format(label, col, val),
                    ParagraphStyle("rfc", fontName="Helvetica-Bold", fontSize=9,
                                   textColor=C.TEXT, leading=13)
                )

            rows = [
                [Paragraph('<font color="#C23B22" size="8"><b>[!] {}</b></font>'.format(tlabel.upper()),
                    ParagraphStyle("rfh", fontName="Helvetica-Bold", fontSize=8,
                                   textColor=C.RED, leading=12))],
                [Paragraph(claim_text[:110],
                    ParagraphStyle("rft", fontName="Helvetica", fontSize=8.5,
                                   textColor=C.TEXT_DIM, leading=13))],
                [Table([[
                    vcell("DICHIARATO", fmt_val(declared), "#8A8799"),
                    vcell("VERIFICATO",  fmt_val(verified),  "#C23B22"),
                    vcell("SCARTO",     "{}%".format(int(magnitude*100)) if magnitude else "n.d.", "#C23B22"),
                ]], colWidths=[CW/3]*3, style=TableStyle([
                    ("TOPPADDING", (0,0), (-1,-1), 4),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 0),
                    ("LEFTPADDING", (0,0), (-1,-1), 0),
                    ("RIGHTPADDING", (0,0), (-1,-1), 0),
                ]))],
            ]
            if reasoning:
                rows.append([Paragraph(reasoning[:260],
                    ParagraphStyle("rfr", fontName="Helvetica", fontSize=7.5,
                                   textColor=C.TEXT_DIM, leading=11))])

            card = Table(rows, colWidths=[CW])
            card.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), C.RED_BG),
                ("LINEAFTER",     (0,0), (0,-1),  2, C.RED),
                ("LEFTPADDING",   (0,0), (-1,-1), 8),
                ("RIGHTPADDING",  (0,0), (-1,-1), 8),
                ("TOPPADDING",    (0,0), (0,0),   8),
                ("BOTTOMPADDING", (0,-1),(-1,-1), 8),
                ("TOPPADDING",    (0,1), (-1,-1), 3),
                ("BOTTOMPADDING", (0,0), (-1,-2), 0),
            ]))
            story.append(KeepTogether([card]))
            story.append(Spacer(1, 3*mm))
        return story

    # ── Non Verificabili ───────────────────────────────────────────────────────

    def _unverifiable_section(self, result):
        S = self.S
        story = [Spacer(1, 4*mm), SectionLabel("Claim Non Verificabili", CW), Spacer(1, 3*mm)]
        for v in result.unverifiable:
            ctype      = getattr(v, "claim_type", "other")
            tlabel     = TYPE_LABELS.get(ctype, ctype.upper())
            claim_text = getattr(v, "claim_text", "") or ""
            reasoning  = getattr(v, "reasoning", "") or ""
            cons       = getattr(v, "sources_consulted", []) or []

            rows = [[
                Paragraph(
                    '<font color="#555870" size="7">{}</font><br/>{}'.format(tlabel, claim_text[:80]),
                    ParagraphStyle("uv", fontName="Helvetica", fontSize=8.5,
                                   textColor=C.TEXT_DIM, leading=13)
                ),
                Paragraph(reasoning[:100],
                    ParagraphStyle("uvr", fontName="Helvetica", fontSize=7.5,
                                   textColor=C.WHITE_DIM, leading=11))
            ]]
            if cons:
                rows.append([
                    Paragraph("Fonti (nessun risultato): " + ", ".join(cons[:4]),
                        ParagraphStyle("uvc", fontName="Courier", fontSize=6.5,
                                       textColor=C.WHITE_DIM, leading=10)),
                    Paragraph("", S["small"])
                ])
            card = Table(rows, colWidths=[CW * 0.5, CW * 0.5])
            card.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), C.GREY_BG),
                ("LINEAFTER",     (0,0), (0,-1),  1.5, C.GREY),
                ("LEFTPADDING",   (0,0), (-1,-1), 8),
                ("RIGHTPADDING",  (0,0), (-1,-1), 8),
                ("TOPPADDING",    (0,0), (-1,-1), 6),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ]))
            story.append(card)
            story.append(Spacer(1, 2*mm))
        return story

    # ── Persone Chiave ─────────────────────────────────────────────────────────

    def _people_section(self, kp):
        S       = self.S
        people  = kp.get("people", [])
        sources = kp.get("sources", [])
        summary = kp.get("summary", "")

        story = [Spacer(1, 4*mm), SectionLabel("Persone Chiave", CW), Spacer(1, 3*mm)]
        if summary:
            story.append(Paragraph(summary, S["body"]))
            story.append(Spacer(1, 3*mm))

        if not people:
            return story

        rows = []
        for i in range(0, min(len(people), 12), 2):
            def pcell(p):
                role   = p.get("role", "")
                name   = p.get("name", "")
                src    = p.get("source", "")
                is_top = any(k in role.lower() for k in
                             ["ceo","founder","fondatore","presidente",
                              "managing","cfo","coo","cto","cmo"])
                nc = "#3A5FD9" if is_top else "#EEE9E0"
                return Paragraph(
                    '<font color="{}" size="8"><b>{}</b></font><br/>'
                    '<font color="#8A8799" size="7">{}</font><br/>'
                    '<font color="#555870" size="6">{}</font>'.format(
                        nc, name, role[:48],
                        "LinkedIn" if p.get("linkedin") else
                        ("Pitch deck" if "pitch" in src else src)
                    ),
                    ParagraphStyle("pc", fontName="Helvetica", fontSize=8,
                                   textColor=C.TEXT, leading=12)
                )

            left  = people[i]
            right = people[i+1] if i+1 < len(people) else None
            row   = [pcell(left), pcell(right) if right else Paragraph("", S["small"])]
            rows.append(row)

        tbl = Table(rows, colWidths=[CW/2 - 1]*2)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C.NAVY_MID),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("RIGHTPADDING",  (0,0), (-1,-1), 8),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("GRID",          (0,0), (-1,-1), 0.3, C.BORDER),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ]))
        story.append(tbl)
        if sources:
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph("Fonti: " + ", ".join(sources), S["mono"]))
        return story

    # ── News ──────────────────────────────────────────────────────────────────

    def _news_section(self, nf):
        S = self.S
        story = [Spacer(1, 4*mm),
                 SectionLabel("Segnalazioni Giornalistiche", CW),
                 Spacer(1, 3*mm)]

        if not nf.get("found"):
            story.append(Paragraph(
                "[OK] Nessuna menzione negativa su fonti giornalistiche italiane "
                "(cause legali, sanzioni, controversie, violazioni normative).",
                ParagraphStyle("nfok", fontName="Helvetica", fontSize=9,
                               textColor=C.GREEN, leading=13)
            ))
            return story

        summary = nf.get("summary", "")
        if summary:
            story.append(Paragraph(summary, S["body"]))
            story.append(Spacer(1, 3*mm))

        for a in (nf.get("articles") or [])[:8]:
            sev      = a.get("severity", "low")
            sh       = "C23B22" if sev == "high" else ("B05A20" if sev == "medium" else "555870")
            sbg      = C.RED_BG if sev == "high" else (C.ORANGE_BG if sev == "medium" else C.GREY_BG)
            sl       = colors.HexColor(sh)
            cat      = {"legal":"Legale","financial":"Finanziario",
                        "reputational":"Reputazione","regulatory":"Normativo"}.get(
                            a.get("category",""), a.get("category",""))
            sev_label = "ALTA" if sev == "high" else ("MEDIA" if sev == "medium" else "BASSA")

            rows = [
                [Paragraph('<font color="#{}" size="7"><b>[{}] {}</b></font>'.format(sh, sev_label, cat.upper()),
                    ParagraphStyle("ns", fontName="Helvetica-Bold", fontSize=7,
                                   textColor=sl, leading=10))],
                [Paragraph(a.get("title","")[:120],
                    ParagraphStyle("nt", fontName="Helvetica-Bold", fontSize=9,
                                   textColor=C.TEXT, leading=13))],
            ]
            desc = a.get("description", "")
            if desc:
                rows.append([Paragraph(desc[:150], S["body_sm"])])
            rows.append([Paragraph(
                "{}  |  {}".format(a.get("source",""), a.get("published_at","")),
                S["mono"]
            )])

            card = Table(rows, colWidths=[CW])
            card.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), sbg),
                ("LINEAFTER",     (0,0), (0,-1),  2, sl),
                ("LEFTPADDING",   (0,0), (-1,-1), 8),
                ("RIGHTPADDING",  (0,0), (-1,-1), 8),
                ("TOPPADDING",    (0,0), (0,0),   6),
                ("BOTTOMPADDING", (0,-1),(-1,-1), 6),
                ("TOPPADDING",    (0,1), (-1,-1), 3),
                ("BOTTOMPADDING", (0,0), (-1,-2), 0),
            ]))
            story.append(card)
            story.append(Spacer(1, 2*mm))
        return story

    # ── Stato Legale ───────────────────────────────────────────────────────────

    def _legal_section(self, ls):
        S = self.S
        story = [Spacer(1, 4*mm),
                 SectionLabel("Stato Legale - OpenCorporates", CW),
                 Spacer(1, 3*mm)]

        if not ls.get("found"):
            story.append(Paragraph("Azienda non trovata nel registro OpenCorporates (IT).", S["body"]))
            return story

        status = ls.get("status_normalized", "")
        sc_col = C.GREEN if status == "attiva" else (C.RED if status == "cessata" else C.ORANGE)

        info = [
            ("Denominazione",    ls.get("name", "n.d.")),
            ("Forma giuridica",  ls.get("company_type", "n.d.")),
            ("Stato",            ls.get("status_label", "n.d.")),
            ("N. registro",      ls.get("company_number", "n.d.")),
            ("Costituzione",     ls.get("incorporation_date") or "n.d."),
            ("Sede",             ls.get("registered_address", "n.d.")),
        ]

        rows = []
        for i in range(0, len(info), 2):
            l, r = info[i], (info[i+1] if i+1 < len(info) else ("", ""))
            def icell(label, val):
                tc = "#C23B22" if label == "Stato" and status == "cessata" else \
                     "#2A7A4A" if label == "Stato" and status == "attiva" else "#EEE9E0"
                return Paragraph(
                    '<font color="#555870" size="6">{}</font><br/>'
                    '<font color="{}" size="8.5"><b>{}</b></font>'.format(label.upper(), tc, str(val)[:55]),
                    ParagraphStyle("lc", fontName="Helvetica", fontSize=8.5,
                                   textColor=C.TEXT, leading=13)
                )
            rows.append([icell(*l), icell(*r)])

        tbl = Table(rows, colWidths=[CW/2]*2)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C.NAVY_MID),
            ("LINEAFTER",     (0,0), (0,-1),  1.5, sc_col),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("RIGHTPADDING",  (0,0), (-1,-1), 10),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("GRID",          (0,0), (-1,-1), 0.3, C.BORDER),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ]))
        story.append(tbl)

        for flag in ls.get("flags", []):
            sev = flag.get("severity", "info")
            fc  = C.RED if sev == "critical" else (C.ORANGE if sev == "warning" else C.WHITE_DIM)
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(">> " + flag.get("text", ""),
                ParagraphStyle("lf", fontName="Helvetica", fontSize=8,
                               textColor=fc, leading=12)))

        oc = ls.get("opencorporates_url", "")
        if oc:
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph("Scheda: " + oc, S["mono"]))

        return story

    # ── Fonti & Disclaimer ─────────────────────────────────────────────────────

    def _sources_disclaimer(self, config):
        S = self.S
        story = [Spacer(1, 4*mm), SectionLabel("Fonti & Disclaimer", CW), Spacer(1, 3*mm)]

        defaults = [
            "Materiale dichiarativo fornito dall'utente (pitch deck, one-pager)",
            "Bilancio depositato Infocamere — fornito dall'utente",
            "LinkedIn — headcount e persone chiave (pagina pubblica)",
            "Sito web aziendale — sezione partner/clienti",
            "OpenCorporates — Registro Imprese italiano",
            "NewsAPI — fonti giornalistiche italiane",
            "Wayback Machine — archivio storico sito web",
        ]
        for i, src in enumerate(config.sources or defaults):
            story.append(Paragraph(
                '<font color="#3A5FD9">{}.  </font>{}'.format(i+1, src),
                ParagraphStyle("si", fontName="Helvetica", fontSize=8,
                               textColor=C.TEXT_DIM, leading=13)
            ))

        story.append(Spacer(1, 5*mm))
        story.append(HLine(CW, C.BORDER))
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph(
            "Il report e stato generato automaticamente da TrueScore sulla base di fonti "
            "pubblicamente disponibili e dei documenti forniti dall'utente. TrueScore non "
            "ha eseguito verifiche sull'autenticita dei documenti caricati. Il report non "
            "costituisce parere legale, finanziario o professionale. Report ID: {}  "
            "Generato il {}.".format(config.report_id, config.generated_at),
            S["disclaimer"]
        ))
        return story

    # ── Helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _score_label(score):
        if score == -2.0: return "Fonti consultate, nessun valore da confrontare"
        if score < 0:     return "Dati insufficienti"
        if score >= 7.5:  return "Alta affidabilita"
        if score >= 5.5:  return "Affidabilita moderata"
        if score >= 3.5:  return "Bassa affidabilita"
        return "Molto bassa affidabilita"
