import pytest


@pytest.fixture
def fresh_accelerator():
    from accelerate.state import AcceleratorState, GradientState
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    yield
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
