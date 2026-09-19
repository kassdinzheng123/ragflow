#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import networkx as nx
from networkx.readwrite import json_graph

import rag.graphrag.utils as graphrag_utils
from rag.graphrag.utils import SOURCE_SUBGRAPH_VERSION, SOURCE_SUBGRAPH_VERSION_KEY, GraphChange


class FakeDocStore:
    def __init__(self, batches=None):
        self._batches = list(batches or [])
        self.deleted = []
        self.inserted = []

    def search(self, *_args, **_kwargs):
        return object()

    def get_fields(self, _res, _flds):
        return self._batches.pop(0) if self._batches else {}

    def delete(self, condition, *_args):
        self.deleted.append(condition)

    def insert(self, chunks, *_args):
        self.inserted.extend(chunks)


def _subgraph_chunk(doc_id, *, edge_weight, node_description, edge_description, stale_rank=None, source_version=SOURCE_SUBGRAPH_VERSION):
    graph = nx.Graph()
    node_attrs = {"description": node_description, "source_id": [doc_id], "entity_type": "X"}
    if stale_rank is not None:
        node_attrs["rank"] = stale_rank
    graph.add_node("A", **node_attrs)
    graph.add_node("B", description=f"B-{doc_id}", source_id=[doc_id], entity_type="Y", **({"rank": stale_rank} if stale_rank is not None else {}))
    graph.add_edge("A", "B", weight=edge_weight, description=edge_description, source_id=[doc_id], keywords=[f"k-{doc_id}"])
    graph.graph["source_id"] = [doc_id]
    if source_version is not None:
        graph.graph[SOURCE_SUBGRAPH_VERSION_KEY] = source_version
    return {
        "knowledge_graph_kwd": "subgraph",
        "content_with_weight": json.dumps(json_graph.node_link_data(graph, edges="edges"), ensure_ascii=False),
        "source_id": [doc_id],
    }


def _run_rebuild(monkeypatch, chunks, exclude=None, *, source_version=SOURCE_SUBGRAPH_VERSION, aliases=None):
    store = FakeDocStore([chunks, {}])
    settings = MagicMock()
    settings.docStoreConn = store
    monkeypatch.setattr(graphrag_utils, "settings", settings)
    return asyncio.run(
        graphrag_utils.rebuild_graph(
            "tenant",
            "kb",
            exclude,
            source_subgraph_version=source_version,
            entity_resolution_aliases=aliases,
        )
    )


def test_rebuild_accumulates_document_contributions(monkeypatch):
    graph = _run_rebuild(
        monkeypatch,
        {
            "doc1": _subgraph_chunk("doc1", edge_weight=1, node_description="node-1", edge_description="edge-1"),
            "doc2": _subgraph_chunk("doc2", edge_weight=2, node_description="node-2", edge_description="edge-2"),
        },
    )

    assert graph.edges["A", "B"]["weight"] == 3
    assert graph.edges["A", "B"]["source_id"] == ["doc1", "doc2"]
    assert graph.edges["A", "B"]["keywords"] == ["k-doc1", "k-doc2"]
    assert graph.nodes["A"]["description"] == f"node-1{graphrag_utils.GRAPH_FIELD_SEP}node-2"


def test_rebuild_excludes_all_document_contributions(monkeypatch):
    graph = _run_rebuild(
        monkeypatch,
        {
            "doc1": _subgraph_chunk("doc1", edge_weight=1, node_description="node-1", edge_description="edge-1"),
            "doc2": _subgraph_chunk("doc2", edge_weight=2, node_description="node-2", edge_description="edge-2"),
        },
        exclude=["doc2"],
    )

    assert graph.graph["source_id"] == ["doc1"]
    assert graph.edges["A", "B"]["weight"] == 1
    assert graph.edges["A", "B"]["description"] == "edge-1"
    assert graph.edges["A", "B"]["source_id"] == ["doc1"]
    assert graph.nodes["A"]["description"] == "node-1"


def test_snapshot_rebuild_does_not_reaccumulate_aggregated_attributes(monkeypatch):
    graph = _run_rebuild(
        monkeypatch,
        {
            "doc1": _subgraph_chunk("doc1", edge_weight=3, node_description="node-1<SEP>node-2", edge_description="edge-1<SEP>edge-2", source_version=None),
            "doc2": _subgraph_chunk("doc2", edge_weight=3, node_description="node-1<SEP>node-2", edge_description="edge-1<SEP>edge-2", source_version=None),
        },
        source_version=None,
    )

    assert graph.edges["A", "B"]["weight"] == 3
    assert graph.edges["A", "B"]["description"] == "edge-1<SEP>edge-2"


def test_rebuild_recomputes_rank(monkeypatch):
    graph = _run_rebuild(
        monkeypatch,
        {"doc1": _subgraph_chunk("doc1", edge_weight=1, node_description="node-1", edge_description="edge-1", stale_rank=7)},
    )

    assert graph.nodes["A"]["rank"] == 1
    assert graph.nodes["B"]["rank"] == 1


def test_rebuild_applies_alias_only_if_canonical_node_survives(monkeypatch):
    canonical = _subgraph_chunk("doc1", edge_weight=1, node_description="canonical", edge_description="edge-1")
    alias = _subgraph_chunk("doc2", edge_weight=2, node_description="alias", edge_description="edge-2")
    alias_graph = json_graph.node_link_graph(json.loads(alias["content_with_weight"]), edges="edges")
    nx.relabel_nodes(alias_graph, {"A": "ALIAS"}, copy=False)
    alias["content_with_weight"] = json.dumps(json_graph.node_link_data(alias_graph, edges="edges"), ensure_ascii=False)
    aliases = {"ALIAS": "A"}

    graph = _run_rebuild(monkeypatch, {"doc1": canonical, "doc2": alias}, aliases=aliases)
    assert "ALIAS" not in graph
    assert graph.nodes["A"]["source_id"] == ["doc1", "doc2"]

    graph_without_canonical = _run_rebuild(monkeypatch, {"doc2": alias}, aliases=aliases)
    assert "ALIAS" in graph_without_canonical
    assert "A" not in graph_without_canonical


def test_get_graph_uses_persisted_source_subgraph_version(monkeypatch):
    chunks = {
        "doc1": _subgraph_chunk("doc1", edge_weight=1, node_description="node-1", edge_description="edge-1"),
        "doc2": _subgraph_chunk("doc2", edge_weight=2, node_description="node-2", edge_description="edge-2"),
    }
    stored_graph = nx.Graph()
    stored_graph.graph["source_id"] = ["doc1", "doc2"]
    stored_graph.graph[SOURCE_SUBGRAPH_VERSION_KEY] = SOURCE_SUBGRAPH_VERSION
    response = SimpleNamespace(
        total=1,
        ids=["graph"],
        field={
            "graph": {
                "content_with_weight": json.dumps(json_graph.node_link_data(stored_graph, edges="edges")),
                "removed_kwd": "Y",
                "source_id": ["doc1", "doc2"],
            }
        },
    )
    settings = MagicMock()
    settings.retriever.search = AsyncMock(return_value=response)
    settings.docStoreConn = FakeDocStore([chunks, {}])
    monkeypatch.setattr(graphrag_utils, "settings", settings)
    monkeypatch.setattr(graphrag_utils.search, "index_name", lambda _tenant: "index")

    graph = asyncio.run(graphrag_utils.get_graph("tenant", "kb"))

    assert graph.edges["A", "B"]["weight"] == 3


def test_set_graph_preserves_source_subgraphs(monkeypatch):
    store = FakeDocStore()
    settings = MagicMock()
    settings.docStoreConn = store
    monkeypatch.setattr(graphrag_utils, "settings", settings)
    monkeypatch.setattr(graphrag_utils.search, "index_name", lambda _tenant: "index")

    graph = nx.Graph()
    graph.graph["source_id"] = ["doc1"]
    graph.graph[SOURCE_SUBGRAPH_VERSION_KEY] = SOURCE_SUBGRAPH_VERSION
    asyncio.run(graphrag_utils.set_graph("tenant", "kb", MagicMock(), graph, GraphChange(), None))

    assert store.deleted[0] == {"knowledge_graph_kwd": ["graph"]}
    assert [chunk["knowledge_graph_kwd"] for chunk in store.inserted] == ["graph"]


def test_set_graph_rewrites_snapshot_subgraphs(monkeypatch):
    store = FakeDocStore()
    settings = MagicMock()
    settings.docStoreConn = store
    monkeypatch.setattr(graphrag_utils, "settings", settings)
    monkeypatch.setattr(graphrag_utils.search, "index_name", lambda _tenant: "index")

    graph = nx.Graph()
    graph.graph["source_id"] = ["doc1"]
    asyncio.run(graphrag_utils.set_graph("tenant", "kb", MagicMock(), graph, GraphChange(), None))

    assert store.deleted[0] == {"knowledge_graph_kwd": ["graph", "subgraph"]}
    assert [chunk["knowledge_graph_kwd"] for chunk in store.inserted] == ["graph", "subgraph"]
