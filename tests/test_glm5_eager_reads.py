"""Stored arrays must finish their reads before GPU transforms; expert placeholders stay lazy."""
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.families.glm5_next import weights


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def reader(tmp_path, tensors):
    shard = tmp_path / "probe.safetensors"
    mx.save_safetensors(str(shard), tensors)
    w = weights.Weights.__new__(weights.Weights)
    w.dir = tmp_path
    w.where = {name: shard.name for name in tensors}
    w._cache = {}
    w._short = lambda name: name
    w.default = (4, 64, "affine")
    w.overrides = {}
    return w


def test_get_evaluates_the_selected_array_before_return(tmp_path, monkeypatch):
    expected = mx.array([[1, -2], [3, 4]], dtype=mx.float16)
    w = reader(tmp_path, {"probe": expected, "other": mx.ones((2, 2))})
    events = []
    original_load, original_eval = mx.load, mx.eval

    def load(*args, **kwargs):
        events.append("read")
        return original_load(*args, **kwargs)

    def evaluate(*arrays):
        assert len(arrays) == 1
        events.append("evaluate")
        return original_eval(*arrays)

    monkeypatch.setattr(mx, "load", load)
    monkeypatch.setattr(mx, "eval", evaluate)
    value = w.get("probe")
    assert events == ["read", "evaluate"]
    assert value.dtype == mx.float16
    assert value.tolist() == expected.tolist()
    w.get("other")
    assert events == ["read", "evaluate", "evaluate"]


def test_explicit_lazy_read_has_no_evaluation(tmp_path, monkeypatch):
    w = reader(tmp_path, {"probe": mx.ones((4,))})
    calls = []
    monkeypatch.setattr(mx, "eval", lambda *arrays: calls.append(arrays))
    value = w.get("probe", eager=False)
    assert value.shape == (4,)
    assert not calls


@pytest.mark.parametrize("stacked", [False, True])
def test_stream_placeholder_does_not_evaluate_experts(tmp_path, monkeypatch, stacked):
    prefix = "layers.0.mlp.switch_mlp.gate_proj" if stacked else "layers.0.mlp.experts.0.gate_proj"
    shape = (2, 64, 64) if stacked else (64, 64)
    q, s, b = mx.quantize(mx.ones(shape, dtype=mx.float16), group_size=64, bits=4)
    w = reader(tmp_path, {prefix+".weight": q, prefix+".scales": s, prefix+".biases": b})
    calls = []
    monkeypatch.setattr(mx, "eval", lambda *arrays: calls.append(arrays))
    placeholder = weights._placeholder(w, "layers.0.mlp", "gate_proj")
    assert placeholder.weight.shape[0] == 0
    assert placeholder.bits == 4 and placeholder.group == 64
    assert not calls


def test_quantized_projection_evaluates_three_stored_arrays(tmp_path, monkeypatch):
    q, s, b = mx.quantize(mx.ones((64, 64), dtype=mx.float16), group_size=64, bits=4)
    w = reader(tmp_path, {"q.weight": q, "q.scales": s, "q.biases": b})
    calls = []
    evaluate = mx.eval

    def record(*arrays):
        calls.append(arrays[0].dtype)
        return evaluate(*arrays)

    monkeypatch.setattr(mx, "eval", record)
    projection = w.q("q")
    assert calls == [mx.float16, mx.float16, mx.uint32]
    assert projection.scales.dtype == mx.float32
    assert projection.weight.tolist() == q.tolist()


def test_file_error_propagates(tmp_path):
    w = reader(tmp_path, {"probe": mx.ones((4,))})
    (tmp_path / "probe.safetensors").unlink()
    with pytest.raises((RuntimeError, ValueError, OSError)):
        w.get("probe")
