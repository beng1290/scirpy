import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from scirpy import get


def test_obs_context(adata_cdr3):
    adata_cdr3.obs["foo"] = "xxx"
    obs_pre = adata_cdr3.obs.copy()
    with get.obs_context(
        adata_cdr3,
        {
            "foo": "bar",
            "a": "b",
            "c": [1, 2, 3, 4, 5],
            "VJ_1_cdr3": get.airr(adata_cdr3, "junction_aa", "VJ_1"),
        },
    ) as a:
        assert a.obs is adata_cdr3.obs
        assert np.all(a.obs["foo"] == "bar")
        assert a.obs["VJ_1_cdr3"].tolist() == ["AAA", "AHA", pd.NA, "AAA", "AAA"]
    pdt.assert_frame_equal(obs_pre, adata_cdr3.obs)


@pytest.mark.parametrize("kwargs", [{}, {"chain": "VJ_1"}])
def test_airr_context(adata_cdr3, kwargs):
    with get.airr_context(adata_cdr3, "junction_aa", **kwargs):
        assert adata_cdr3.obs["VJ_1_junction_aa"].tolist() == [
            "AAA",
            "AHA",
            pd.NA,
            "AAA",
            "AAA",
        ]


def test_airr_multi_model_explicit_third_chain(adata_multichain):
    obs = get.airr(
        adata_multichain,
        ["junction_aa", "umi_count"],
        ["VJ_1", "VJ_2", "VJ_3", "VDJ_3"],
    )

    assert obs["VJ_3_junction_aa"].tolist() == ["TRA3", pd.NA, pd.NA]
    assert obs["VDJ_3_junction_aa"].tolist() == ["TRB3", pd.NA, pd.NA]
    assert obs["VJ_3_umi_count"].tolist() == [10, pd.NA, pd.NA]


def test_airr_multi_model_all_chains_expands_to_max_chain_count(adata_multichain):
    obs = get.airr(adata_multichain, "junction_aa", chain="all")

    assert obs.columns.tolist() == [
        "VJ_1_junction_aa",
        "VJ_2_junction_aa",
        "VJ_3_junction_aa",
        "VDJ_1_junction_aa",
        "VDJ_2_junction_aa",
        "VDJ_3_junction_aa",
    ]
    assert obs["VJ_3_junction_aa"].tolist() == ["TRA3", pd.NA, pd.NA]
    assert obs["VDJ_3_junction_aa"].tolist() == ["TRB3", pd.NA, pd.NA]


def test_airr_multi_model_invalid_chain_index(adata_multichain):
    with pytest.raises(ValueError, match="Valid chains are VJ_1, VJ_2, VJ_3, VDJ_1, VDJ_2, VDJ_3"):
        get.airr(adata_multichain, "junction_aa", "VJ_5")


def test_airr_context_multi_model_third_chain(adata_multichain):
    obs_pre = adata_multichain.obs.copy()
    with get.airr_context(adata_multichain, "junction_aa", chain=["VJ_3"]):
        assert adata_multichain.obs["VJ_3_junction_aa"].tolist() == ["TRA3", pd.NA, pd.NA]

    pdt.assert_frame_equal(obs_pre, adata_multichain.obs)
