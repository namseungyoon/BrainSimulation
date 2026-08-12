from __future__ import annotations

from typing import Mapping, Protocol, Sequence, TypeAlias, cast, overload

import numpy as np
import numpy.typing as npt

NestGpuStatusValue: TypeAlias = float | int | str | list[float] | dict[str, list[float]]
NestGpuConnValue: TypeAlias = str | int | float | bool


class NestGpuNodes(Protocol):
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, index: int) -> int: ...
    @overload
    def __getitem__(self, index: slice) -> NestGpuNodes: ...


class NestGpuRemoteNodes(Protocol):
    i_host: int
    node_seq: NestGpuNodes


class NestGpuModule(Protocol):
    def SetRandomSeed(self, seed: int) -> None: ...

    def SetTimeResolution(self, resolution_ms: float) -> None: ...

    def Create(
        self,
        model: str,
        count: int = 1,
        n_ports: int | None = None,
    ) -> NestGpuNodes: ...

    def RemoteCreate(
        self,
        host: int,
        model: str,
        count: int = 1,
        n_ports: int | None = None,
    ) -> NestGpuRemoteNodes: ...

    @overload
    def SetStatus(
        self,
        nodes: NestGpuNodes | NestGpuRemoteNodes,
        params: Mapping[str, NestGpuStatusValue],
    ) -> None: ...
    @overload
    def SetStatus(
        self,
        nodes: NestGpuNodes | NestGpuRemoteNodes,
        params: str,
        val: NestGpuStatusValue,
    ) -> None: ...

    def Connect(
        self,
        pre: NestGpuNodes | Sequence[int],
        post: NestGpuNodes | Sequence[int],
        conn_spec: Mapping[str, NestGpuConnValue],
        syn_spec: Mapping[str, NestGpuConnValue],
    ) -> None: ...

    def ConnectGroupGroupUInt32(
        self,
        pre: npt.NDArray[np.uint32],
        post: npt.NDArray[np.uint32],
        conn_spec: Mapping[str, NestGpuConnValue],
        syn_spec: Mapping[str, NestGpuConnValue],
    ) -> None: ...

    def ConnectMpiInit(self) -> None: ...

    def HostId(self) -> int: ...

    def HostNum(self) -> int: ...

    def CreateHostGroup(self, hosts: list[int]) -> int: ...

    def ConnectDistributedFixedIndegree(
        self,
        source_hosts: list[int],
        source_groups: list[NestGpuNodes],
        target_hosts: list[int],
        target_groups: list[NestGpuNodes],
        indegree: int,
        host_group: int,
        syn_spec: Mapping[str, NestGpuConnValue],
    ) -> None: ...

    def ActivateRecSpikeTimes(self, nodes: NestGpuNodes, max_spikes: int) -> None: ...

    def CreateRecord(
        self,
        file_name: str,
        var_names: list[str],
        nodes: list[int],
        ports: list[int],
    ) -> int: ...

    def SetRecordStride(self, record: int, sampling_stride: int) -> None: ...

    def Simulate(self, duration_ms: float) -> None: ...

    def GetRecSpikeTimes(self, nodes: NestGpuNodes) -> list[list[float] | None] | None: ...

    def GetRecordData(self, record: int) -> list[list[float]]: ...


def nestgpu_module(raw_module: object) -> NestGpuModule:
    return cast(NestGpuModule, raw_module)
