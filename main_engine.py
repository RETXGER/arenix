"""
Arenix Engine - Standalone Runner
arenix_engine.py'deki ana motoru doğrudan çalıştırır.
"""
import logging
from arenix_engine import main, build_config_from_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ArenixEngine")


if __name__ == "__main__":
    logger.info("🚀 Arenix Engine başlatılıyor...")
    config = build_config_from_env()
    logger.info(f"📋 Sektör: {config.industry} | Hedef: {config.target_provider}/{config.target_model}")
    logger.info(f"📋 Max Turlar: {config.max_turns} | Observer Onay: {config.require_observer_confirmation}")
    main()