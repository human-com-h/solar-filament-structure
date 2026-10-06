"""Run with pytest; uses synthetic interfaces, no paper data or GPU training."""
from filament_structure.validation import run_checks

def test_cpu_scientific_contract():
    assert run_checks(with_models=False)['status']=='PASS_CPU_ENGINEERING'
