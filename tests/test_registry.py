from sexvary.config import build_registry


def test_registry_loads_seed_configs():
    registry = build_registry()
    assert "nlsy79_main" in registry.datasets
    assert "piaac_cycle2" in registry.datasets
    assert "general_intelligence_g" in registry.traits
    assert len(registry.external_datasets()) >= 5
