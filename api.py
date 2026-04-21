"""
Arenix REST API — FastAPI ile Programatik Erişim
=================================================
Red team testlerini, turnuvaları ve rapor üretimini
REST API üzerinden yönetir.
"""

import asyncio
import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

if HAS_FASTAPI:

    # ============================================================
    # PYDANTIC MODELLER
    # ============================================================

    class RunConfig(BaseModel):
        # Saldırgan model (zorunlu)
        attacker_provider: str = Field(..., description="Saldırgan provider: gemini, openai, anthropic, deepseek, ollama, custom")
        attacker_model: str = Field(..., description="Saldırgan model adı, ör: gemini-2.0-flash")
        attacker_api_key: Optional[str] = Field(None, description="Saldırgan API anahtarı (opsiyonel)")
        attacker_base_url: Optional[str] = Field(None, description="Custom provider için base URL")

        # Hedef/savunma model (opsiyonel — verilmezse saldırganla aynı)
        target_provider: Optional[str] = Field(None, description="Hedef provider (boşsa attacker_provider kullanılır)")
        target_model: Optional[str] = Field(None, description="Hedef model adı (boşsa attacker_model kullanılır)")
        target_api_key: Optional[str] = Field(None, description="Hedef API anahtarı (opsiyonel)")
        target_base_url: Optional[str] = Field(None, description="Custom provider için hedef base URL (opsiyonel)")

        turns: int = Field(10, ge=1, le=50, description="Test tur sayısı")
        industry: str = Field("default", description="Sektör: fintech, healthcare, ecommerce, government, education, legal, default")
        attack_profile: str = Field("balanced", description="Saldırı profili: soft, balanced, aggressive, compliance")
        generate_report: bool = Field(True, description="HTML rapor üret")
        compliance_check: bool = Field(True, description="Uyumluluk analizi yap")

        # Geriye dönük uyumluluk (eski tek-model kullanımı)
        model_provider: Optional[str] = Field(None, description="[Deprecated] attacker_provider kullanın")
        model_name: Optional[str] = Field(None, description="[Deprecated] attacker_model kullanın")
        api_key: Optional[str] = Field(None, description="[Deprecated] attacker_api_key kullanın")

        def resolved_attacker(self):
            prov = self.attacker_provider or self.model_provider or "mock"
            model = self.attacker_model or self.model_name or "mock"
            key = self.attacker_api_key or self.api_key
            return prov, model, key

        def resolved_target(self):
            atk_prov, atk_model, atk_key = self.resolved_attacker()
            prov = self.target_provider or atk_prov
            model = self.target_model or atk_model
            key = self.target_api_key or atk_key
            return prov, model, key

    class TournamentConfig(BaseModel):
        models: List[Dict[str, str]] = Field(..., description="[{provider, model_name, api_key?}] listesi")
        turns: int = Field(10, ge=1, le=50)
        industry: str = Field("default")
        attack_profile: str = Field("balanced", description="Saldırı profili: soft, balanced, aggressive, compliance")

    class RunStatus(BaseModel):
        run_id: str
        status: str  # pending, running, completed, failed
        progress: int = 0
        total_turns: int = 0
        attacker: Optional[str] = None
        target: Optional[str] = None
        result: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        started_at: Optional[str] = None
        completed_at: Optional[str] = None

    class TestLLMRequest(BaseModel):
        model: str = Field(..., description="Model adı (örn: gemini/gemini-2.0-flash veya gpt-4o-mini)")
        prompt: str = Field(..., description="Test prompt'u")

    # ============================================================
    # APP & STATE
    # ============================================================

    app = FastAPI(
        title="Arenix AI Red Team API",
        description="LLM güvenlik testleri için REST API — AI vs AI destekli",
        version="2.1.0",
    )

    # In-memory çalıştırma durumu (production'da Redis/DB kullanılmalı)
    _runs: Dict[str, RunStatus] = {}
    _run_events: Dict[str, List[Dict[str, Any]]] = {}  # WebSocket için canlı olaylar
    _run_created_at: Dict[str, float] = {}
    _runs_lock = threading.RLock()
    _RUN_TTL_SECONDS = int(os.getenv("ARENIX_RUN_TTL_SECONDS", "86400"))

    _SUPPORTED_PROVIDERS = ["gemini", "openai", "anthropic", "deepseek", "ollama", "custom", "mock"]
    _KNOWN_MODELS = {
        "gemini": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.0-pro"],
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
        "deepseek": ["deepseek-chat", "deepseek-reasoner"],
        "ollama": ["llama3", "mistral", "phi3", "gemma"],
        "custom": ["company-model-v1"],
        "mock": ["mock"],
    }

    def _cleanup_runs() -> None:
        now = time.time()
        with _runs_lock:
            stale_run_ids = []
            for run_id, created_at in _run_created_at.items():
                run = _runs.get(run_id)
                is_old = (now - created_at) > _RUN_TTL_SECONDS
                is_terminal = (run is None) or (run.status in ("completed", "failed"))
                if is_old and is_terminal:
                    stale_run_ids.append(run_id)

            for run_id in stale_run_ids:
                _runs.pop(run_id, None)
                _run_created_at.pop(run_id, None)
                _run_events.pop(run_id, None)

    def _push_event(run_id: str, event: Dict[str, Any]) -> None:
        """WebSocket akışı için olayı kuyruğa ekler."""
        with _runs_lock:
            if run_id not in _run_events:
                _run_events[run_id] = []
            _run_events[run_id].append(event)

    # ============================================================
    # YARDIMCI FONKSİYONLAR
    # ============================================================

    def _set_api_key_env(provider: str, api_key: Optional[str]) -> None:
        """Kullanıcıdan gelen API anahtarını geçici olarak çevre değişkenine yazar."""
        if api_key:
            if provider == "custom":
                os.environ["ARENIX_CUSTOM_API_KEY"] = api_key
            else:
                os.environ[f"{provider.upper()}_API_KEY"] = api_key

    def _set_custom_base_url_env(base_url: Optional[str]) -> None:
        if base_url and base_url.strip():
            os.environ["ARENIX_CUSTOM_BASE_URL"] = base_url.strip()

    def _resolve_model_ref(model_ref: str) -> tuple[str, str]:
        """model alanını provider/model formatına normalize eder."""
        ref = (model_ref or "").strip()
        if not ref:
            return "mock", "mock-target"

        if "/" in ref:
            provider, model_name = ref.split("/", 1)
            return provider.strip().lower(), model_name.strip()

        lower = ref.lower()
        if lower.startswith("gpt-"):
            return "openai", ref
        if lower.startswith("claude-"):
            return "anthropic", ref
        if lower.startswith("deepseek-"):
            return "deepseek", ref
        if lower.startswith("gemini-"):
            return "gemini", ref
        if lower in {"llama3", "mistral", "phi3", "gemma"}:
            return "ollama", ref
        return "mock", ref

    def _run_test_sync(run_id: str, config: RunConfig):
        """Senkron test çalıştırma (background task içinde) — AI vs AI destekli."""
        from arenix_engine import (
            Orchestrator, SessionConfig,
            AttackerRole, TargetRole, AnalyzerRole, ObserverRole,
            ContextTracker, ArenixAnalyzerV2, build_adapter,
        )
        from compliance_mapper import ComplianceMapper
        from report_generator import ReportGenerator

        with _runs_lock:
            run = _runs.get(run_id)
            if run is None:
                return
            run.status = "running"
            run.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            atk_prov, atk_model, atk_key = config.resolved_attacker()
            tgt_prov, tgt_model, tgt_key = config.resolved_target()

            _set_api_key_env(atk_prov, atk_key)
            if tgt_prov != atk_prov:
                _set_api_key_env(tgt_prov, tgt_key)
            if atk_prov == "custom":
                _set_custom_base_url_env(config.attacker_base_url)
            if tgt_prov == "custom":
                _set_custom_base_url_env(config.target_base_url or config.attacker_base_url)

            _push_event(run_id, {"type": "start", "attacker": f"{atk_prov}/{atk_model}", "target": f"{tgt_prov}/{tgt_model}"})

            session_cfg = SessionConfig(
                session_id=run_id,
                industry=config.industry,
                attack_profile=config.attack_profile,
                attacker_provider=atk_prov,
                attacker_model=atk_model,
                target_provider=tgt_prov,
                target_model=tgt_model,
                analyzer_provider=atk_prov,
                analyzer_model=atk_model,
                observer_provider=atk_prov,
                observer_model=atk_model,
                max_turns=config.turns,
            )

            attacker_adapter = build_adapter(atk_prov, atk_model)
            target_adapter = build_adapter(tgt_prov, tgt_model)
            observer_adapter = build_adapter(atk_prov, atk_model)

            tracker = ContextTracker()
            analyzer_engine = ArenixAnalyzerV2(industry=config.industry)

            attacker = AttackerRole(attacker_adapter, max_retries=3, profile=config.attack_profile)
            target = TargetRole(target_adapter, max_retries=3)
            analyzer = AnalyzerRole(analyzer_engine, tracker=tracker)
            observer = ObserverRole(observer_adapter, max_retries=3)

            def _on_turn(event: Dict[str, Any]) -> None:
                turn_num = int(event.get("turn", 0) or 0)
                with _runs_lock:
                    r = _runs.get(run_id)
                    if r:
                        r.progress = turn_num
                _push_event(run_id, {
                    "type": "turn",
                    "turn": turn_num,
                    "compromise_score": event.get("compromise_score", 0),
                    "resilience_score": event.get("resilience_score", 0),
                    "status": event.get("status", ""),
                })

            orchestrator = Orchestrator(
                config=session_cfg,
                attacker=attacker,
                target=target,
                analyzer=analyzer,
                observer=observer,
                tracker=tracker,
                on_turn=_on_turn,
            )

            report = orchestrator.run()
            analysis_report = report.get("analysis_report", report)

            result: Dict[str, Any] = {"report": report}

            if config.compliance_check:
                mapper = ComplianceMapper(industry=config.industry)
                mapper.analyze_report(analysis_report)
                result["compliance"] = mapper.to_dict()

            if config.generate_report:
                gen = ReportGenerator()
                html = gen.generate_html(
                    analysis_report,
                    compliance_data=result.get("compliance"),
                )
                result["html_report"] = html

            with _runs_lock:
                run = _runs.get(run_id)
                if run is None:
                    return
                run.result = result
                run.status = "completed"
                run.progress = config.turns
        except Exception as exc:
            with _runs_lock:
                run = _runs.get(run_id)
                if run is None:
                    return
                run.status = "failed"
                run.error = str(exc)
        finally:
            with _runs_lock:
                run = _runs.get(run_id)
                if run is not None:
                    run.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # ============================================================
    # ENDPOINTS
    # ============================================================

    @app.get("/", tags=["Meta"])
    async def root():
        return {
            "name": "Arenix AI Red Team API",
            "version": "2.1.0",
            "endpoints": {
                "GET  /health": "Sistem sağlık durumu",
                "GET  /api/v1/models": "Desteklenen provider ve modeller",
                "POST /api/v1/run": "Yeni test başlat (AI vs AI destekli)",
                "GET  /api/v1/run/{run_id}": "Test durumunu sorgula",
                "GET  /api/v1/runs": "Tüm testleri listele",
                "GET  /api/v1/run/{run_id}/report": "HTML rapor al",
                "GET  /api/v1/run/{run_id}/json": "JSON rapor al",
                "DELETE /api/v1/run/{run_id}": "Test sil",
                "POST /api/v1/tournament": "Turnuva başlat",
                "WS  /ws/{run_id}": "Canlı tur olayları (WebSocket)",
            },
        }

    @app.get("/health", tags=["Meta"])
    async def health():
        """Sistem sağlık durumu ve modül bilgisi döndürür."""
        import sys
        modules: Dict[str, bool] = {}
        for mod in ["fastapi", "pydantic", "google.genai", "openai", "anthropic", "pandas", "plotly"]:
            try:
                __import__(mod.replace(".", "/").replace("/", "."))
                modules[mod] = True
            except ImportError:
                modules[mod] = False

        return {
            "status": "ok",
            "version": "2.1.0",
            "python": sys.version,
            "active_runs": sum(1 for r in _runs.values() if r.status == "running"),
            "total_runs": len(_runs),
            "modules": modules,
        }

    @app.get("/api/v1/models", tags=["Meta"])
    async def list_models():
        """Desteklenen provider'lar ve bilinen modelleri döndürür."""
        return {
            "providers": _SUPPORTED_PROVIDERS,
            "models": _KNOWN_MODELS,
            "note": "ollama için OLLAMA_BASE_URL env değişkenini ayarlayın (varsayılan: http://localhost:11434)",
        }

    @app.post("/test-llm", tags=["Test"])
    async def test_llm(payload: TestLLMRequest):
        """Tek prompt ile hedef modeli hızlıca test eder ve özet risk çıktısı döndürür."""
        from arenix_engine import (
            Orchestrator, SessionConfig, ModelResponse,
            TargetRole, AnalyzerRole, ObserverRole,
            ContextTracker, ArenixAnalyzerV2, build_adapter,
        )

        provider, model_name = _resolve_model_ref(payload.model)

        try:
            target_adapter = build_adapter(provider, model_name)
            observer_adapter = build_adapter("mock", "mock-observer")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Model/adaptör oluşturulamadı: {exc}") from exc

        class _SinglePromptAttacker:
            def __init__(self, prompt: str):
                self.prompt = prompt

            def generate(self, conversation: List[Dict[str, str]], temperature: float = 0.2, last_turn_analysis=None) -> ModelResponse:
                return ModelResponse(
                    model_name="single-prompt-attacker",
                    content=self.prompt,
                    latency_ms=0,
                    input_tokens=0,
                    output_tokens=0,
                )

        tracker = ContextTracker()
        analyzer_engine = ArenixAnalyzerV2(industry="default")

        orchestrator = Orchestrator(
            config=SessionConfig(
                session_id=f"test-llm-{uuid.uuid4()}",
                industry="default",
                attacker_provider="mock",
                attacker_model="single-prompt",
                target_provider=provider,
                target_model=model_name,
                analyzer_provider="mock",
                analyzer_model="rule-based",
                observer_provider="mock",
                observer_model="mock-observer",
                max_turns=1,
                stop_on_break=True,
                require_observer_confirmation=False,
            ),
            attacker=_SinglePromptAttacker(payload.prompt),
            target=TargetRole(target_adapter, max_retries=3),
            analyzer=AnalyzerRole(analyzer_engine, tracker=tracker),
            observer=ObserverRole(observer_adapter, max_retries=3),
            tracker=tracker,
        )

        result = orchestrator.run()
        analysis_report = result.get("analysis_report", {}) if isinstance(result, dict) else {}

        risk_score = float(
            analysis_report.get("max_compromise_score", analysis_report.get("overall_compromise_score", 0.0))
        )
        vulnerabilities = analysis_report.get("vulnerabilities_found", []) or []
        status = analysis_report.get("status", "UNKNOWN")
        summary = f"Model {provider}/{model_name} için tek-tur test tamamlandı. Durum: {status}, risk: {risk_score:.1f}."

        return {
            "risk_score": risk_score,
            "vulnerabilities": vulnerabilities,
            "summary": summary,
        }

    @app.post("/api/v1/run", response_model=RunStatus, tags=["Test"])
    async def start_run(config: RunConfig, background_tasks: BackgroundTasks):
        """Yeni bir red team testi başlatır. AI vs AI: saldırgan ve hedef farklı modeller olabilir.
        Test arka planda çalışır, run_id ile durumu sorgulanabilir.
        """
        atk_prov, atk_model, _ = config.resolved_attacker()
        tgt_prov, tgt_model, _ = config.resolved_target()

        run_id = str(uuid.uuid4())[:8]
        run = RunStatus(
            run_id=run_id,
            status="pending",
            total_turns=config.turns,
            attacker=f"{atk_prov}/{atk_model}",
            target=f"{tgt_prov}/{tgt_model}",
        )
        _cleanup_runs()
        with _runs_lock:
            _runs[run_id] = run
            _run_created_at[run_id] = time.time()
            _run_events[run_id] = []
        background_tasks.add_task(_run_test_sync, run_id, config)
        return run

    @app.websocket("/ws/{run_id}")
    async def websocket_events(websocket: WebSocket, run_id: str):
        """Canlı tur olaylarını WebSocket üzerinden akıtır.
        Her tur tamamlandığında JSON olay gönderir.
        Bağlantı test bitince otomatik kapanır.
        """
        await websocket.accept()
        sent_index = 0
        try:
            while True:
                with _runs_lock:
                    run = _runs.get(run_id)
                    events = list(_run_events.get(run_id, []))

                if run is None:
                    await websocket.send_json({"type": "error", "message": f"run_id bulunamadı: {run_id}"})
                    break

                # Yeni olayları gönder
                for event in events[sent_index:]:
                    await websocket.send_json(event)
                sent_index = len(events)

                if run.status in ("completed", "failed"):
                    await websocket.send_json({"type": "done", "status": run.status, "progress": run.progress})
                    break

                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            pass

    @app.get("/api/v1/run/{run_id}", response_model=RunStatus, tags=["Test"])
    async def get_run(run_id: str):
        """Test durumunu sorgular."""
        with _runs_lock:
            run = _runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"Run bulunamadı: {run_id}")
            return run

    @app.get("/api/v1/runs", tags=["Test"])
    async def list_runs():
        """Tüm testleri listeler."""
        _cleanup_runs()
        with _runs_lock:
            return {
                rid: {"status": r.status, "started_at": r.started_at, "completed_at": r.completed_at}
                for rid, r in _runs.items()
            }

    @app.get("/api/v1/run/{run_id}/report", response_class=HTMLResponse, tags=["Rapor"])
    async def get_report(run_id: str):
        """HTML rapor döndürür."""
        with _runs_lock:
            run = _runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"Run bulunamadı: {run_id}")
        if run.status != "completed":
            raise HTTPException(status_code=409, detail=f"Test henüz tamamlanmadı: {run.status}")
        html_report = (run.result or {}).get("html_report")
        if not html_report:
            raise HTTPException(status_code=404, detail="HTML rapor üretilmemiş")
        return HTMLResponse(content=html_report)

    @app.get("/api/v1/run/{run_id}/json", tags=["Rapor"])
    async def get_json_report(run_id: str):
        """JSON rapor döndürür."""
        with _runs_lock:
            run = _runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"Run bulunamadı: {run_id}")
        if run.status != "completed":
            raise HTTPException(status_code=409, detail=f"Test henüz tamamlanmadı: {run.status}")
        result = run.result or {}
        return JSONResponse(content={
            "report": result.get("report"),
            "compliance": result.get("compliance"),
        })

    @app.post("/api/v1/tournament", tags=["Turnuva"])
    async def start_tournament(config: TournamentConfig, background_tasks: BackgroundTasks):
        """Turnuva başlatır — birden fazla modeli aynı senaryoda test eder."""
        run_id = f"tourn-{str(uuid.uuid4())[:6]}"
        run = RunStatus(run_id=run_id, status="pending", total_turns=config.turns)
        _cleanup_runs()
        with _runs_lock:
            _runs[run_id] = run
            _run_created_at[run_id] = time.time()

        def _run_tournament():
            from arenix_engine import (
                Orchestrator, SessionConfig,
                AttackerRole, TargetRole, AnalyzerRole, ObserverRole,
                ContextTracker, ArenixAnalyzerV2, build_adapter,
            )
            from tournament import TournamentEngine

            with _runs_lock:
                run_obj = _runs.get(run_id)
                if run_obj is None:
                    return
                run_obj.status = "running"
                run_obj.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")

            try:
                reports = {}
                for model_cfg in config.models:
                    provider = model_cfg["provider"]
                    model_name = model_cfg["model_name"]
                    api_key = model_cfg.get("api_key")
                    mid = f"{provider}/{model_name}"

                    _set_api_key_env(provider, api_key)

                    session_cfg = SessionConfig(
                        session_id=str(uuid.uuid4()),
                        industry=config.industry,
                        attack_profile=config.attack_profile,
                        attacker_provider=provider,
                        attacker_model=model_name,
                        target_provider=provider,
                        target_model=model_name,
                        analyzer_provider=provider,
                        analyzer_model=model_name,
                        observer_provider=provider,
                        observer_model=model_name,
                        max_turns=config.turns,
                    )

                    attacker_adapter = build_adapter(provider, model_name)
                    target_adapter = build_adapter(provider, model_name)
                    observer_adapter = build_adapter(provider, model_name)

                    tracker = ContextTracker()
                    analyzer_engine = ArenixAnalyzerV2(industry=config.industry)

                    attacker = AttackerRole(attacker_adapter, max_retries=3, profile=config.attack_profile)
                    target = TargetRole(target_adapter, max_retries=3)
                    analyzer = AnalyzerRole(analyzer_engine, tracker=tracker)
                    observer = ObserverRole(observer_adapter, max_retries=3)

                    orchestrator = Orchestrator(
                        config=session_cfg,
                        attacker=attacker,
                        target=target,
                        analyzer=analyzer,
                        observer=observer,
                        tracker=tracker,
                    )

                    report = orchestrator.run()
                    reports[mid] = report.get("analysis_report", report)

                engine = TournamentEngine()
                result = engine.run_from_reports(reports, turns=config.turns, industry=config.industry)
                with _runs_lock:
                    run_obj = _runs.get(run_id)
                    if run_obj is None:
                        return
                    run_obj.result = engine.to_dict(result)
                    run_obj.status = "completed"
            except Exception as exc:
                with _runs_lock:
                    run_obj = _runs.get(run_id)
                    if run_obj is None:
                        return
                    run_obj.status = "failed"
                    run_obj.error = str(exc)
            finally:
                with _runs_lock:
                    run_obj = _runs.get(run_id)
                    if run_obj is not None:
                        run_obj.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        background_tasks.add_task(_run_tournament)
        return run

    @app.delete("/api/v1/run/{run_id}", tags=["Test"])
    async def delete_run(run_id: str):
        """Test sonucunu siler."""
        with _runs_lock:
            run = _runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"Run bulunamadı: {run_id}")
            if run.status in ("pending", "running"):
                raise HTTPException(status_code=409, detail="Çalışan test silinemez")
            del _runs[run_id]
            _run_created_at.pop(run_id, None)
        return {"deleted": run_id}

    # ============================================================
    # CLI ENTRYPOINT
    # ============================================================

    def start_api(host: str = "0.0.0.0", port: int = 8000):
        """API sunucusunu başlatır."""
        import uvicorn
        uvicorn.run(app, host=host, port=port)

else:
    # FastAPI yüklü değilse placeholder
    def start_api(**kwargs):
        print("FastAPI yüklü değil. Kurulum: pip install fastapi uvicorn")


if __name__ == "__main__":
    start_api()
