from __future__ import annotations

import ctypes
import hashlib
import importlib.util
from pathlib import Path
import struct
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import ca1.sim.gpu_backend as gpu_backend


ROOT = Path(__file__).resolve().parents[1]


class _CFunction:
    def __init__(self, callback: Any = None) -> None:
        self.argtypes: Any = None
        self.restype: Any = None
        self.calls: list[tuple[Any, ...]] = []
        self.callback = callback

    def __call__(self, *args: Any) -> Any:
        self.calls.append(args)
        return 0 if self.callback is None else self.callback(*args)


class _CLibrary:
    def __init__(self) -> None:
        self.functions: dict[str, _CFunction] = {}

    @staticmethod
    def _zero(*_args: Any) -> int:
        return 0

    @staticmethod
    def _is_int_synapse_parameter(value: Any) -> int:
        return int(ctypes.string_at(value) in {b"receptor", b"synapse_group"})

    @staticmethod
    def _is_float_synapse_parameter(value: Any) -> int:
        return int(ctypes.string_at(value) in {b"weight", b"delay"})

    def __getattr__(self, name: str) -> _CFunction:
        callback = None
        if name == "NESTGPU_GetErrorCode":
            callback = self._zero
        elif name == "NESTGPU_SynSpecIsIntParam":
            callback = self._is_int_synapse_parameter
        elif name == "NESTGPU_SynSpecIsFloatParam":
            callback = self._is_float_synapse_parameter
        return self.functions.setdefault(name, _CFunction(callback))


def _load_wrapper(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, _CLibrary]:
    library = _CLibrary()
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: library)
    monkeypatch.setenv("NESTGPU_LIB", "/unused/mock/libnestgpu.so")
    spec = importlib.util.spec_from_file_location(
        "_nestgpu_zero_copy_test", ROOT / "nest-gpu/pythonlib/nestgpu.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, library


def _uint32_values(pointer: ctypes.c_void_p, count: int) -> list[int]:
    array_type = ctypes.c_uint32 * count
    return list(array_type.from_address(pointer.value))


def test_nestgpu_uint32_wrapper_passes_numpy_data_pointers_and_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, library = _load_wrapper(monkeypatch)
    wrapper._CONNECT_GROUP_GROUP_MAX_EDGES = 3
    source = np.arange(7, dtype=np.uint32) + 101
    target = np.arange(7, dtype=np.uint32) + 501

    wrapper.ConnectGroupGroupUInt32(
        source,
        target,
        {"rule": "one_to_one"},
        {"weight": 0.4, "delay": 3.0, "receptor": 2},
    )

    calls = library.functions["NESTGPU_ConnectGroupGroup"].calls
    assert [call[1] for call in calls] == [3, 3, 1]
    assert [call[3] for call in calls] == [3, 3, 1]
    assert [_uint32_values(call[0], call[1]) for call in calls] == [
        [101, 102, 103],
        [104, 105, 106],
        [107],
    ]
    assert [_uint32_values(call[2], call[3]) for call in calls] == [
        [501, 502, 503],
        [504, 505, 506],
        [507],
    ]
    assert calls[0][0].value == source.ctypes.data
    assert calls[0][2].value == target.ctypes.data


def test_nestgpu_fused_wrapper_passes_arrays_and_scalar_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, library = _load_wrapper(monkeypatch)
    wrapper._CONNECT_GROUP_GROUP_MAX_EDGES = 3
    source = np.arange(5, dtype=np.uint32) + 101
    target = np.arange(5, dtype=np.uint32) + 501

    wrapper.ConnectExplicitArrays(source, target, 0.25, 3.0, 7, 2)

    calls = library.functions["NESTGPU_ConnectExplicitArrays"].calls
    assert [call[2].value for call in calls] == [3, 2]
    assert [_uint32_values(call[0], call[2].value) for call in calls] == [
        [101, 102, 103],
        [104, 105],
    ]
    assert [_uint32_values(call[1], call[2].value) for call in calls] == [
        [501, 502, 503],
        [504, 505],
    ]
    assert [call[3].value for call in calls] == [pytest.approx(0.25)] * 2
    assert [call[4].value for call in calls] == [pytest.approx(3.0)] * 2
    assert [call[5].value for call in calls] == [7, 7]
    assert [call[6].value for call in calls] == [2, 2]
    assert calls[0][0].value == source.ctypes.data
    assert calls[0][1].value == target.ctypes.data


@pytest.mark.parametrize(
    ("source", "error"),
    [
        (np.arange(4, dtype=np.int64), TypeError),
        (np.arange(8, dtype=np.uint32)[::2], ValueError),
        (np.arange(4, dtype=np.uint32).reshape(2, 2), ValueError),
    ],
)
def test_nestgpu_uint32_wrapper_rejects_non_abi_ready_buffers(
    monkeypatch: pytest.MonkeyPatch,
    source: np.ndarray[Any, Any],
    error: type[Exception],
) -> None:
    wrapper, _library = _load_wrapper(monkeypatch)
    with pytest.raises(error):
        wrapper.ConnectGroupGroupUInt32(
            source,
            np.arange(4, dtype=np.uint32),
            {"rule": "one_to_one"},
            {"weight": 1.0},
        )


class _Nodes:
    def __init__(self, start: int, count: int) -> None:
        self.start = start
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> int:
        return self.start + index


class _ConnectRecorder:
    def __init__(self, zero_copy: bool, fused: bool = False) -> None:
        self.edges: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], Any]] = []
        if zero_copy:
            self.ConnectGroupGroupUInt32 = self._connect_uint32  # type: ignore[attr-defined]
        if fused:
            self.ConnectExplicitArrays = self._connect_fused  # type: ignore[attr-defined]

    def _connect_uint32(self, source: Any, target: Any, conn: Any, syn: Any) -> None:
        assert source.dtype == target.dtype == np.uint32
        assert source.flags.c_contiguous and target.flags.c_contiguous
        self.edges.append((source.copy(), target.copy(), (conn, syn)))

    def Connect(self, source: Any, target: Any, conn: Any, syn: Any) -> None:  # noqa: N802
        self.edges.append((np.asarray(source), np.asarray(target), (conn, syn)))

    def _connect_fused(
        self,
        source: Any,
        target: Any,
        weight: Any,
        delay: Any,
        receptor: Any,
        synapse_group: Any,
    ) -> None:
        assert source.dtype == target.dtype == np.uint32
        syn = {
            "weight": weight,
            "delay": delay,
            "receptor": receptor,
            "synapse_group": synapse_group,
        }
        self.edges.append(
            (source.copy(), target.copy(), ({"rule": "one_to_one"}, syn))
        )


def _connection_digest(recorder: _ConnectRecorder) -> str:
    digest = hashlib.sha256()
    for source, target, spec in recorder.edges:
        digest.update(np.asarray(source, dtype="<u4").tobytes())
        digest.update(np.asarray(target, dtype="<u4").tobytes())
        digest.update(repr(spec).encode())
    return digest.hexdigest()


def _packed_connection_digest(recorder: _ConnectRecorder, layout: int) -> str:
    """CPU reference for NEST-GPU's default conn12b/conn16b byte layout."""
    digest = hashlib.sha256()
    for sources, targets, (_conn, syn) in recorder.edges:
        delay = round(float(syn.get("delay", 0.0)) / 0.1)
        weight = float(syn.get("weight", 0.0))
        port = int(syn.get("receptor", 0))
        syn_group = int(syn.get("synapse_group", 0))
        for source, target in zip(sources, targets, strict=True):
            if layout == 12:
                key = (int(source) << 12) | (delay - 1)
                value = (int(target) << 12) | (port << 7) | syn_group
                digest.update(struct.pack("<IIf", key, value, weight))
            elif layout == 16:
                key = (
                    (int(source) << 32)
                    | ((delay - 1) << 18)
                    | (port << 11)
                    | syn_group
                )
                digest.update(struct.pack("<QIf", key, int(target), weight))
            else:  # pragma: no cover - test helper misuse
                raise ValueError(layout)
    return digest.hexdigest()


def test_zero_copy_and_fallback_marshal_identical_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_indices = np.array([9, 2, 9, 0, 4], dtype=np.int64)
    target_indices = np.array([0, 0, 3, 4, 4], dtype=np.int64)
    syn_spec = {"weight": 0.25, "delay": 3.0, "receptor": 7, "synapse_group": 2}
    zero_copy = _ConnectRecorder(zero_copy=True)
    fallback = _ConnectRecorder(zero_copy=False)

    gpu_backend._connect_explicit_one_to_one(
        zero_copy,
        _Nodes(100, 10),
        _Nodes(500, 5),
        source_indices,
        target_indices,
        syn_spec,
        label="digest",
    )
    monkeypatch.setenv("CA1_GPU_ZERO_COPY_EXPLICIT_CONNECT", "0")
    gpu_backend._connect_explicit_one_to_one(
        fallback,
        _Nodes(100, 10),
        _Nodes(500, 5),
        source_indices,
        target_indices,
        syn_spec,
        label="digest",
    )

    assert _connection_digest(zero_copy) == _connection_digest(fallback)
    np.testing.assert_array_equal(zero_copy.edges[0][0], fallback.edges[0][0])
    np.testing.assert_array_equal(zero_copy.edges[0][1], fallback.edges[0][1])
    assert zero_copy.edges[0][2] == fallback.edges[0][2]


def test_fused_zero_copy_and_chunked_have_identical_packed_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_indices = np.array([9, 2, 9, 0, 4], dtype=np.int64)
    target_indices = np.array([0, 0, 3, 4, 4], dtype=np.int64)
    syn_spec = {"weight": 0.25, "delay": 3.0, "receptor": 7, "synapse_group": 2}
    fused = _ConnectRecorder(zero_copy=True, fused=True)
    zero_copy = _ConnectRecorder(zero_copy=True)
    chunked = _ConnectRecorder(zero_copy=False)

    monkeypatch.setenv("CA1_GPU_FUSED_EXPLICIT_CONNECT", "1")
    gpu_backend._connect_explicit_one_to_one(
        fused, _Nodes(100, 10), _Nodes(500, 5), source_indices,
        target_indices, syn_spec, label="digest"
    )
    monkeypatch.setenv("CA1_GPU_FUSED_EXPLICIT_CONNECT", "0")
    gpu_backend._connect_explicit_one_to_one(
        zero_copy, _Nodes(100, 10), _Nodes(500, 5), source_indices,
        target_indices, syn_spec, label="digest"
    )
    monkeypatch.setenv("CA1_GPU_ZERO_COPY_EXPLICIT_CONNECT", "0")
    gpu_backend._connect_explicit_one_to_one(
        chunked, _Nodes(100, 10), _Nodes(500, 5), source_indices,
        target_indices, syn_spec, label="digest"
    )

    for layout in (12, 16):
        assert _packed_connection_digest(fused, layout) == _packed_connection_digest(
            zero_copy, layout
        )
        assert _packed_connection_digest(fused, layout) == _packed_connection_digest(
            chunked, layout
        )


def test_zero_copy_global_id_mapping_validates_range() -> None:
    with pytest.raises(ValueError, match="outside"):
        gpu_backend._explicit_node_ids_uint32(_Nodes(100, 3), np.array([3]))
    with pytest.raises(ValueError, match="outside"):
        gpu_backend._explicit_node_ids_uint32(_Nodes(100, 3), np.array([-1]))
    with pytest.raises(OverflowError, match="uint32"):
        gpu_backend._explicit_node_ids_uint32(_Nodes(np.iinfo(np.uint32).max, 2), np.array([1]))
