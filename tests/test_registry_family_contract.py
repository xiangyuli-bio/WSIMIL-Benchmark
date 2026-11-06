from milplus.registry import inspect_method,require_runnable,profiles

def test_technical_family_query_lists_all_complete_architectures():
    family=inspect_method('icmil_coupled')
    assert family['kind']=='family' and family['runnable'] is False
    assert {p['id'] for p in family['execution_profiles']}=={p['id'] for p in profiles() if p['family']=='icmil_coupled'}
    assert len(family['execution_profiles'])==3

def test_family_execution_default_remains_the_complete_abmil_pipeline():
    assert require_runnable('icmil_coupled')['id']==require_runnable('ICMIL Coupled')['id']=='icmil_coupled_abmil_patient_reference'
    for p in inspect_method('icmil_coupled')['execution_profiles']:
        assert require_runnable(p['id'])['id']==p['id']

def test_context_identity_is_never_a_coupled_execution_route():
    assert require_runnable('ICMIL In-context')['family']=='icmil_context'
    assert require_runnable('ICMIL Coupled')['family']=='icmil_coupled'
