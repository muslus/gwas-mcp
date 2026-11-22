"""
Regression tests for STRING API tools in clinical_tools module.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import aiohttp
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gwas_mcp.tools import clinical_tools
from gwas_mcp.tools.clinical_tools import (
    get_functional_enrichment,
    get_interaction_network,
    get_protein_interactions,
)


@pytest.fixture(autouse=True)
def clear_clinical_cache():
    """Ensure in-memory cache is empty before and after each test."""
    clinical_tools._cache.clear()
    yield
    clinical_tools._cache.clear()


class _MockStringResponse:
    """Mock aiohttp response mimicking STRING's text/json Content-Type."""

    def __init__(self, body, status: int = 200, content_type: str = "text/json"):
        self._body = body if isinstance(body, str) else json.dumps(body)
        self.status = status
        self.content_type = content_type
        self.headers = {"Content-Type": content_type}

    async def text(self) -> str:
        return self._body

    async def json(self):
        raise aiohttp.ContentTypeError(
            request_info=MagicMock(),
            history=(),
            message=f"Attempt to decode JSON with unexpected mimetype: {self.content_type}",
            headers=self.headers,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_mock_session(response: _MockStringResponse):
    """Create a mock aiohttp.ClientSession recording calls to session.get."""
    recorded = {}

    def _get(url, params=None, **kwargs):
        recorded["url"] = url
        recorded["params"] = params
        return response

    session = MagicMock()
    session.get = MagicMock(side_effect=_get)
    session.__aenter__.return_value = session
    session.__aexit__.return_value = False
    return session, recorded


class TestStringTools:
    """Regression tests for STRING identifier encoding, text/json parsing, and caching."""

    async def test_interaction_network_params_and_text_json_parsing(self):
        sample_edges = [
            {
                "stringId_A": "9606.ENSP00000233838",
                "stringId_B": "9606.ENSP00000295228",
                "preferredName_A": "GGCX",
                "preferredName_B": "INHBB",
                "score": 650,
            }
        ]
        response = _MockStringResponse(sample_edges, content_type="text/json")
        session, recorded = _make_mock_session(response)

        with patch("gwas_mcp.tools.clinical_tools.aiohttp.ClientSession", return_value=session):
            raw = await get_interaction_network(["GGCX", "INHBB", "CCNT2"])

        assert recorded["params"]["identifiers"] == "GGCX\rINHBB\rCCNT2"
        assert "%" not in recorded["params"]["identifiers"]

        result = json.loads(raw)
        assert result["n_edges"] == 1
        assert result["n_nodes"] == 2
        assert "warning" not in result
        assert len(clinical_tools._cache) == 1

    async def test_interaction_network_empty_result_not_cached_and_warns(self):
        response = _MockStringResponse([], content_type="text/json")
        session, recorded = _make_mock_session(response)

        with patch("gwas_mcp.tools.clinical_tools.aiohttp.ClientSession", return_value=session):
            raw = await get_interaction_network(["GGCX", "INHBB", "CCNT2"])

        result = json.loads(raw)
        assert result["n_edges"] == 0
        assert "warning" in result
        assert len(clinical_tools._cache) == 0

    async def test_protein_interactions_text_json_parsing(self):
        sample_edges = [
            {
                "stringId_A": "9606.ENSP00000233838",
                "stringId_B": "9606.ENSP00000295228",
                "preferredName_A": "GGCX",
                "preferredName_B": "VKORC1",
                "score": 990,
            }
        ]
        response = _MockStringResponse(sample_edges, content_type="text/json")
        session, _ = _make_mock_session(response)

        with patch("gwas_mcp.tools.clinical_tools.aiohttp.ClientSession", return_value=session):
            raw = await get_protein_interactions("GGCX")

        result = json.loads(raw)
        assert result["n_interactions"] == 1
        assert result["interacting_proteins"][0]["partner"] == "VKORC1"

    async def test_functional_enrichment_params_and_text_json_parsing(self):
        sample_enrichment = [
            {
                "category": "Process",
                "term": "GO:0007596",
                "description": "Blood coagulation",
                "p_value": 0.001,
                "fdr": 0.01,
                "number_of_genes": 2,
                "inputGenes": ["GGCX", "INHBB"],
                "preferredNames": ["GGCX", "INHBB"],
            }
        ]
        response = _MockStringResponse(sample_enrichment, content_type="text/json")
        session, recorded = _make_mock_session(response)

        with patch("gwas_mcp.tools.clinical_tools.aiohttp.ClientSession", return_value=session):
            raw = await get_functional_enrichment(["GGCX", "INHBB", "CCNT2"])

        assert recorded["params"]["identifiers"] == "GGCX\rINHBB\rCCNT2"
        assert "%" not in recorded["params"]["identifiers"]

        result = json.loads(raw)
        assert result["n_categories"] == 1
        assert "Process" in result["enrichment_by_category"]
        assert result["enrichment_by_category"]["Process"][0]["genes_matched"] == ["GGCX", "INHBB"]

    async def test_functional_enrichment_legacy_string_input_genes(self):
        sample_enrichment = [
            {
                "category": "Process",
                "term": "GO:0007596",
                "description": "Blood coagulation",
                "p_value": 0.001,
                "fdr": 0.01,
                "number_of_genes": 2,
                "inputGenes": "GGCX, INHBB",
            }
        ]
        response = _MockStringResponse(sample_enrichment, content_type="text/json")
        session, _ = _make_mock_session(response)

        with patch("gwas_mcp.tools.clinical_tools.aiohttp.ClientSession", return_value=session):
            raw = await get_functional_enrichment(["GGCX", "INHBB", "CCNT2"])

        result = json.loads(raw)
        assert result["enrichment_by_category"]["Process"][0]["genes_matched"] == ["GGCX", "INHBB"]

    @pytest.mark.skipif(
        not os.environ.get("RUN_LIVE_STRING_TESTS"),
        reason="Set RUN_LIVE_STRING_TESTS=1 to run live STRING API test",
    )
    async def test_live_interaction_network(self):
        raw = await get_interaction_network(["GGCX", "INHBB", "CCNT2"])
        result = json.loads(raw)
        assert "error" not in result
        assert "n_edges" in result
