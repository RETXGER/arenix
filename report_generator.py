"""
Arenix Report Generator — Profesyonel HTML Rapor Üreteci
=========================================================
Arenix test sonuçlarından embed grafikli, interaktif
HTML rapor ve JSON export üretir.
"""

import html
import json
import time
from typing import Any, Dict, List, Optional


# ============================================================
# RENK PALETİ
# ============================================================

COLORS = {
    "critical": "#DC2626",
    "high": "#EA580C",
    "medium": "#D97706",
    "low": "#2563EB",
    "info": "#6B7280",
    "safe": "#16A34A",
    "bg_dark": "#0F172A",
    "bg_card": "#1E293B",
    "text": "#E2E8F0",
    "accent": "#8B5CF6",
    "grid": "#334155",
}


def _severity_color(severity: str) -> str:
    return {
        "Critical": COLORS["critical"],
        "High": COLORS["high"],
        "Medium": COLORS["medium"],
        "Low": COLORS["low"],
        "Informational": COLORS["info"],
    }.get(severity, COLORS["info"])


def _status_color(status: str) -> str:
    if status in ("CRITICAL", "COMPROMISED"):
        return COLORS["critical"]
    if status == "AT_RISK":
        return COLORS["high"]
    if status == "RESISTANT":
        return COLORS["medium"]
    return COLORS["safe"]


def _esc(text: Any) -> str:
    return html.escape(str(text))


# ============================================================
# SVG CHART HELPERS  (harici kütüphane gerekmez)
# ============================================================

def _svg_line_chart(
    series: Dict[str, List[float]],
    width: int = 700,
    height: int = 300,
    title: str = "",
) -> str:
    """Basit SVG çizgi grafik üretir. series: {label: [values]}"""
    palette = ["#8B5CF6", "#F59E0B", "#EF4444", "#10B981", "#3B82F6", "#EC4899"]
    margin = {"top": 40, "right": 120, "bottom": 40, "left": 55}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    all_vals = [v for vals in series.values() for v in vals]
    if not all_vals:
        return ""
    y_min = min(0, min(all_vals))
    y_max = max(100, max(all_vals))
    y_range = y_max - y_min or 1
    max_len = max(len(vals) for vals in series.values())

    def x_pos(i: int) -> float:
        return margin["left"] + (i / max(max_len - 1, 1)) * plot_w

    def y_pos(v: float) -> float:
        return margin["top"] + plot_h - ((v - y_min) / y_range) * plot_h

    lines_svg = []
    legend_items = []
    for idx, (label, values) in enumerate(series.items()):
        color = palette[idx % len(palette)]
        points = " ".join(f"{x_pos(i):.1f},{y_pos(v):.1f}" for i, v in enumerate(values))
        lines_svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>')
        # Noktalar
        for i, v in enumerate(values):
            lines_svg.append(
                f'<circle cx="{x_pos(i):.1f}" cy="{y_pos(v):.1f}" r="3" fill="{color}"/>'
            )
        ly = margin["top"] + 20 + idx * 22
        legend_items.append(
            f'<rect x="{width - margin["right"] + 10}" y="{ly - 8}" width="12" height="12" fill="{color}" rx="2"/>'
            f'<text x="{width - margin["right"] + 28}" y="{ly + 2}" fill="{COLORS["text"]}" font-size="11">{_esc(label)}</text>'
        )

    # Grid çizgileri
    grid_svg = []
    for step in range(5):
        gy = margin["top"] + (step / 4) * plot_h
        gv = y_max - (step / 4) * y_range
        grid_svg.append(f'<line x1="{margin["left"]}" y1="{gy:.1f}" x2="{margin["left"] + plot_w}" y2="{gy:.1f}" stroke="{COLORS["grid"]}" stroke-dasharray="4"/>')
        grid_svg.append(f'<text x="{margin["left"] - 8}" y="{gy + 4:.1f}" text-anchor="end" fill="{COLORS["text"]}" font-size="10">{gv:.0f}</text>')

    # X ekseni etiketleri
    x_labels = []
    for i in range(max_len):
        x_labels.append(
            f'<text x="{x_pos(i):.1f}" y="{height - margin["bottom"] + 20}" text-anchor="middle" fill="{COLORS["text"]}" font-size="10">T{i + 1}</text>'
        )

    return f"""<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:{COLORS['bg_card']};border-radius:12px;margin:12px 0;">
  <text x="{width / 2}" y="22" text-anchor="middle" fill="{COLORS['text']}" font-size="14" font-weight="bold">{_esc(title)}</text>
  {''.join(grid_svg)}
  {''.join(x_labels)}
  {''.join(lines_svg)}
  {''.join(legend_items)}
</svg>"""


def _svg_bar_chart(
    labels: List[str],
    values: List[float],
    width: int = 700,
    height: int = 300,
    title: str = "",
    color: str = "#8B5CF6",
) -> str:
    margin = {"top": 40, "right": 20, "bottom": 60, "left": 55}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    y_max = max(100, max(values)) if values else 100
    n = len(labels)
    bar_w = min(50, plot_w / max(n, 1) * 0.7)
    gap = (plot_w - bar_w * n) / max(n + 1, 1)

    bars = []
    for i, (label, val) in enumerate(zip(labels, values)):
        bx = margin["left"] + gap * (i + 1) + bar_w * i
        bh = (val / y_max) * plot_h
        by = margin["top"] + plot_h - bh
        bars.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}" rx="4" opacity="0.85"/>'
            f'<text x="{bx + bar_w / 2:.1f}" y="{by - 6:.1f}" text-anchor="middle" fill="{COLORS["text"]}" font-size="10">{val:.1f}</text>'
            f'<text x="{bx + bar_w / 2:.1f}" y="{margin["top"] + plot_h + 18}" text-anchor="middle" fill="{COLORS["text"]}" font-size="9" transform="rotate(-25,{bx + bar_w / 2:.1f},{margin["top"] + plot_h + 18})">{_esc(label[:12])}</text>'
        )

    return f"""<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:{COLORS['bg_card']};border-radius:12px;margin:12px 0;">
  <text x="{width / 2}" y="22" text-anchor="middle" fill="{COLORS['text']}" font-size="14" font-weight="bold">{_esc(title)}</text>
  {''.join(bars)}
</svg>"""


# ============================================================
# RAPOR OLUŞTURUCU
# ============================================================

class ReportGenerator:
    """Arenix test sonuçlarından profesyonel HTML rapor üretir."""

    def _normalize_report_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(report_data, dict):
            return {}

        analysis_report = report_data.get("analysis_report")
        if isinstance(analysis_report, dict):
            return analysis_report

        return report_data

    def _normalize_tournament_data(self, tournament_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(tournament_data, dict):
            return tournament_data

        if "leaderboard" in tournament_data:
            return tournament_data

        result = tournament_data.get("result")
        if isinstance(result, dict) and "leaderboard" in result:
            return result

        return tournament_data

    def generate_html(
        self,
        report_data: Dict[str, Any],
        compliance_data: Optional[Dict[str, Any]] = None,
        tournament_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        report_data = self._normalize_report_data(report_data)
        tournament_data = self._normalize_tournament_data(tournament_data)

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        model = report_data.get("model_tested", "Bilinmeyen Model")
        status = report_data.get("status", "UNKNOWN")
        industry = report_data.get("industry", "default")
        turns = report_data.get("turn_analyses", [])
        bp = report_data.get("break_point")

        sections = []
        sections.append(self._section_header(model, status, industry, ts))
        sections.append(self._section_kpi(report_data))
        sections.append(self._section_risk_chart(turns))
        sections.append(self._section_turn_details(turns))

        if bp:
            sections.append(self._section_breakpoint(bp))

        if compliance_data:
            sections.append(self._section_compliance(compliance_data))

        if tournament_data:
            sections.append(self._section_tournament(tournament_data))

        sections.append(self._section_footer(ts))

        body = "\n".join(sections)
        return self._wrap_html(body, model)

    def _wrap_html(self, body: str, title: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Arenix Rapor — {_esc(title)}</title>
<style>
:root {{ --bg: {COLORS['bg_dark']}; --card: {COLORS['bg_card']}; --text: {COLORS['text']}; --accent: {COLORS['accent']}; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; line-height:1.6; padding:24px; max-width:1100px; margin:auto; }}
.card {{ background:var(--card); border-radius:16px; padding:24px; margin:16px 0; box-shadow:0 4px 24px rgba(0,0,0,0.3); }}
.card h2 {{ color:var(--accent); margin-bottom:12px; font-size:1.2rem; }}
.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; }}
.kpi {{ text-align:center; padding:16px; background:rgba(139,92,246,0.08); border-radius:12px; border:1px solid rgba(139,92,246,0.15); }}
.kpi .value {{ font-size:2rem; font-weight:700; color:var(--accent); }}
.kpi .label {{ font-size:0.85rem; opacity:0.7; }}
.badge {{ display:inline-block; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; }}
table {{ width:100%; border-collapse:collapse; margin:8px 0; }}
th, td {{ padding:10px 12px; text-align:left; border-bottom:1px solid rgba(255,255,255,0.06); font-size:0.9rem; }}
th {{ color:var(--accent); font-weight:600; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.5px; }}
tr:hover {{ background:rgba(139,92,246,0.04); }}
.severity-critical {{ color:{COLORS['critical']}; font-weight:700; }}
.severity-high {{ color:{COLORS['high']}; font-weight:600; }}
.severity-medium {{ color:{COLORS['medium']}; }}
.severity-low {{ color:{COLORS['low']}; }}
.header {{ text-align:center; padding:40px 0 20px; }}
.header h1 {{ font-size:2.5rem; background:linear-gradient(135deg,#8B5CF6,#EC4899); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.header .subtitle {{ opacity:0.6; margin-top:4px; }}
.status-badge {{ padding:8px 20px; border-radius:24px; font-size:1.1rem; font-weight:700; display:inline-block; }}
.footer {{ text-align:center; padding:32px 0 16px; opacity:0.4; font-size:0.8rem; }}
svg {{ max-width:100%; height:auto; }}
details {{ margin:4px 0; }}
details summary {{ cursor:pointer; padding:6px 0; font-weight:500; }}
details summary:hover {{ color:var(--accent); }}
.remediation-item {{ padding:4px 0 4px 16px; border-left:2px solid var(--accent); margin:4px 0; font-size:0.85rem; }}
</style>
</head>
<body>
{body}
</body>
</html>"""

    def _section_header(self, model: str, status: str, industry: str, ts: str) -> str:
        sc = _status_color(status)
        return f"""<div class="header">
  <h1>ARENIX</h1>
  <p class="subtitle">AI Red Team Güvenlik Raporu</p>
  <p style="margin-top:12px"><span class="status-badge" style="background:{sc}22;color:{sc};border:2px solid {sc}">{_esc(status)}</span></p>
  <p style="margin-top:8px;opacity:0.5">{_esc(model)} · {_esc(industry.title())} · {_esc(ts)}</p>
</div>"""

    def _section_kpi(self, report: Dict) -> str:
        turns = report.get("turn_analyses", [])
        total = len(turns)
        avg_r = sum(t.get("resilience_score", 0) for t in turns) / max(total, 1)
        avg_c = sum(t.get("compromise_score", 0) for t in turns) / max(total, 1)
        max_c = report.get("max_compromise_score", 0)
        leaks = sum(1 for t in turns if t.get("leakage_detected"))
        bp = report.get("break_point")
        break_info = f"Tur {bp['turn_number']}" if bp else "Yok"

        return f"""<div class="card">
  <h2>Temel Metrikler</h2>
  <div class="kpi-grid">
    <div class="kpi"><div class="value">{total}</div><div class="label">Toplam Tur</div></div>
    <div class="kpi"><div class="value">{avg_r:.1f}</div><div class="label">Ort. Direnç</div></div>
    <div class="kpi"><div class="value" style="color:{COLORS['critical'] if avg_c > 50 else COLORS['safe']}">{avg_c:.1f}</div><div class="label">Ort. Compromise</div></div>
    <div class="kpi"><div class="value" style="color:{COLORS['critical'] if max_c > 70 else COLORS['medium']}">{max_c:.1f}</div><div class="label">Maks. Compromise</div></div>
    <div class="kpi"><div class="value" style="color:{COLORS['critical'] if leaks > 0 else COLORS['safe']}">{leaks}</div><div class="label">Sızıntı</div></div>
    <div class="kpi"><div class="value">{_esc(break_info)}</div><div class="label">Kırılma</div></div>
  </div>
</div>"""

    def _section_risk_chart(self, turns: List[Dict]) -> str:
        if not turns:
            return ""
        resilience = [t.get("resilience_score", 0) for t in turns]
        compromise = [t.get("compromise_score", 0) for t in turns]
        pressure = [t.get("attack_pressure_score", 0) for t in turns]
        chart = _svg_line_chart(
            {"Direnç": resilience, "Compromise": compromise, "Saldırı Basıncı": pressure},
            title="Risk Evrim Grafiği",
        )
        return f'<div class="card"><h2>Risk Evrimi</h2>{chart}</div>'

    def _section_turn_details(self, turns: List[Dict]) -> str:
        if not turns:
            return ""
        rows = []
        for t in turns:
            tn = t.get("turn_number") or t.get("turn_id", "?")
            r = t.get("resilience_score", 0)
            c = t.get("compromise_score", 0)
            leak = "Evet" if t.get("leakage_detected") else "—"
            attacks = ", ".join(
                a if isinstance(a, str) else getattr(a, "value", str(a))
                for a in t.get("attack_types", [])
            ) or "—"
            rc = COLORS["critical"] if c > 60 else (COLORS["medium"] if c > 30 else COLORS["safe"])
            rows.append(f"<tr><td>{tn}</td><td>{r:.1f}</td><td style='color:{rc}'>{c:.1f}</td><td>{_esc(attacks)}</td><td>{leak}</td></tr>")

        return f"""<div class="card">
  <h2>Tur Detayları</h2>
  <table>
    <thead><tr><th>Tur</th><th>Direnç</th><th>Compromise</th><th>Saldırı Tipleri</th><th>Sızıntı</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>"""

    def _section_breakpoint(self, bp: Dict) -> str:
        return f"""<div class="card" style="border-left:4px solid {COLORS['critical']}">
  <h2 style="color:{COLORS['critical']}">Kırılma Noktası</h2>
  <p><strong>Tur:</strong> {bp.get('turn_number', '?')}</p>
  <p><strong>Compromise:</strong> {bp.get('compromise_score', 0):.1f}</p>
  <p><strong>Sebep:</strong> {_esc(bp.get('why_broken', 'Bilinmiyor'))}</p>
</div>"""

    def _section_compliance(self, compliance: Dict) -> str:
        summary = compliance.get("executive_summary", {})
        findings = compliance.get("findings", [])
        regulations = summary.get("applicable_regulations", {})

        severity_html = ""
        for sev, count in summary.get("by_severity", {}).items():
            sc = _severity_color(sev)
            severity_html += f'<span class="badge" style="background:{sc}22;color:{sc};border:1px solid {sc};margin:0 4px">{sev}: {count}</span>'

        owasp_list = ", ".join(summary.get("owasp_categories_affected", [])) or "Yok"
        regulations_text = " | ".join(
            f"{_esc(group.title())}: {_esc(', '.join(items))}"
            for group, items in regulations.items()
            if items
        )
        risk_level = summary.get("compliance_risk_level", "LOW")
        risk_color = _severity_color({"CRITICAL": "Critical", "HIGH": "High", "MEDIUM": "Medium"}.get(risk_level, "Low"))

        findings_html = ""
        for f in findings:
            sev = f.get("severity", "Low")
            sc = _severity_color(sev)
            rem_html = "".join(f'<div class="remediation-item">{_esc(r)}</div>' for r in f.get("remediation", []))
            nist_controls = ", ".join(f.get("nist_controls", []))
            references = ", ".join(f.get("references", []))
            evidence_summary = f.get("evidence_summary")
            findings_html += f"""<details>
  <summary><span class="badge" style="background:{sc}22;color:{sc};border:1px solid {sc}">{sev}</span> {_esc(f.get('title', ''))}</summary>
  <div style="padding:8px 0 8px 16px">
    <p>{_esc(f.get('description', ''))}</p>
    {'<p><strong>OWASP:</strong> ' + _esc(f.get('owasp', '')) + '</p>' if f.get('owasp') else ''}
    {'<p><strong>NIST:</strong> ' + _esc(f.get('nist_function', '')) + ' / ' + _esc(f.get('nist_category', '')) + '</p>' if f.get('nist_function') else ''}
    {'<p><strong>NIST Kontrolleri:</strong> ' + _esc(nist_controls) + '</p>' if nist_controls else ''}
    {'<p><strong>Kanıt:</strong> ' + _esc(evidence_summary) + '</p>' if evidence_summary else ''}
    {'<p><strong>Referanslar:</strong> ' + _esc(references) + '</p>' if references else ''}
    {('<p><strong>İyileştirme:</strong></p>' + rem_html) if rem_html else ''}
  </div>
</details>"""

        return f"""<div class="card">
  <h2>Uyumluluk Analizi</h2>
  <p style="margin-bottom:8px"><strong>Risk Seviyesi:</strong> <span style="color:{risk_color};font-weight:700">{risk_level}</span></p>
  <p style="margin-bottom:8px">{severity_html}</p>
  <p style="margin-bottom:12px"><strong>Etkilenen OWASP Kategorileri:</strong> {owasp_list}</p>
    {f'<p style="margin-bottom:12px"><strong>Uygulanabilir Regülasyonlar:</strong> {regulations_text}</p>' if regulations_text else ''}
  {findings_html}
</div>"""

    def _section_tournament(self, tournament: Dict) -> str:
        lb = tournament.get("leaderboard", [])
        if not lb:
            return ""

        labels = [e["model_id"] for e in lb]
        scores = [e.get("composite_score", 0) for e in lb]
        chart = _svg_bar_chart(labels, scores, title="Model Karşılaştırma — Bileşik Skor")

        rows = []
        for e in lb:
            bp_info = f"Tur {e['break_turn']}" if e.get("break_turn") else "—"
            sc = _status_color(e.get("status", "SAFE"))
            rows.append(f"""<tr>
  <td style="font-weight:700">#{e['rank']}</td>
  <td>{_esc(e['model_id'])}</td>
  <td style="color:var(--accent);font-weight:600">{e.get('composite_score', 0):.1f}</td>
  <td>{e.get('avg_resilience', 0):.1f}</td>
  <td>{e.get('avg_compromise', 0):.1f}</td>
  <td>{bp_info}</td>
  <td><span style="color:{sc}">{_esc(e.get('status', ''))}</span></td>
</tr>""")

        h2h = tournament.get("head_to_head", [])
        h2h_html = ""
        if h2h:
            h2h_rows = "".join(
                f"<tr><td>{_esc(h['model_a'])}</td><td>{_esc(h['model_b'])}</td>"
                f"<td style='font-weight:600'>{_esc(h.get('winner') or 'Berabere')}</td>"
                f"<td>{h.get('margin', 0):.1f}</td><td>{h.get('p_value', 1):.4f}</td></tr>"
                for h in h2h
            )
            h2h_html = f"""<h3 style="margin-top:16px;color:var(--accent)">Birebir Karşılaştırma</h3>
<table><thead><tr><th>Model A</th><th>Model B</th><th>Kazanan</th><th>Fark</th><th>p-value</th></tr></thead>
<tbody>{h2h_rows}</tbody></table>"""

        return f"""<div class="card">
  <h2>Turnuva Sonuçları</h2>
  {chart}
  <table>
    <thead><tr><th>#</th><th>Model</th><th>Bileşik Skor</th><th>Ort. Direnç</th><th>Ort. Compromise</th><th>Kırılma</th><th>Durum</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {h2h_html}
</div>"""

    def _section_footer(self, ts: str) -> str:
        return f"""<div class="footer">
  <p>Arenix AI Red Team Platform · Rapor oluşturma: {_esc(ts)}</p>
  <p>Bu rapor otomatik olarak üretilmiştir. Tüm bulgular bağımsız doğrulama gerektirir.</p>
</div>"""

    def export_json(self, report_data: Dict, compliance_data: Optional[Dict] = None) -> str:
        report_data = self._normalize_report_data(report_data)
        output = {
            "meta": {
                "generator": "Arenix Report Generator",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "version": "2.0",
            },
            "report": report_data,
        }
        if compliance_data:
            output["compliance"] = compliance_data
        return json.dumps(output, indent=2, ensure_ascii=False, default=str)
