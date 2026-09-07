"""Full synthetic source-to-report acceptance without substitute producers."""

import pytest

from tests.helpers.research_acceptance import run_a_share_acceptance


@pytest.mark.complete_chain
def test_a_share_real_source_model_portfolio_report_chain(tmp_path):
    run_a_share_acceptance(tmp_path)
