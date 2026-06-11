# pylama:ignore=W0611,W0404
import sys
from typing import cast

import anndata as ad
import awkward as ak
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from mudata import MuData

import scirpy as ir

from . import TESTDATA


@pytest.mark.parametrize("key_added", [None, "my_key"])
@pytest.mark.parametrize("inplace", [True, False])
def test_define_clonotype_clusters_return_values(adata_define_clonotype_clusters_singletons, key_added, inplace):
    """Test that key_added and inplace work as expected"""
    adata = adata_define_clonotype_clusters_singletons
    res = ir.tl.define_clonotype_clusters(
        adata,
        receptor_arms="VJ",
        dual_ir="primary_only",
        metric="identity",
        sequence="aa",
        same_v_gene=True,
        inplace=inplace,
        key_added=key_added,
    )  # type: ignore

    clonotype_expected = np.array([0, 1, 2, 3]).astype(str)
    clonotype_size_expected = np.array([1, 1, 1, 1])

    if inplace:
        assert res is None
        if key_added is None:
            npt.assert_equal(adata.obs["cc_aa_identity"], clonotype_expected)
            npt.assert_equal(adata.obs["cc_aa_identity_size"], clonotype_size_expected)
        else:
            npt.assert_equal(adata.obs["my_key"], clonotype_expected)
            npt.assert_equal(adata.obs["my_key_size"], clonotype_size_expected)

    else:
        npt.assert_equal(res[0], clonotype_expected)
        npt.assert_equal(res[1], clonotype_size_expected)


@pytest.mark.parametrize("receptor_arms", ["VJ", "VDJ", "all", "any"])
@pytest.mark.parametrize("dual_ir", ["primary_only", "all", "any"])
def test_define_clonotypes_diagonal_connectivities(adata_define_clonotype_clusters_singletons, receptor_arms, dual_ir):
    """Regression test for #236. Computing the clonotypes when
    no cells are connected in the clonotype neighborhood graph should not fail.
    """
    clonotype, clonotype_size, _ = ir.tl.define_clonotype_clusters(
        adata_define_clonotype_clusters_singletons,
        receptor_arms=receptor_arms,
        dual_ir=dual_ir,
        metric="identity",
        sequence="aa",
        same_v_gene=True,
        inplace=False,
    )  # type: ignore
    npt.assert_equal(clonotype, np.array([0, 1, 2, 3]).astype(str))
    npt.assert_equal(clonotype_size, np.array([1, 1, 1, 1]))


def test_clonotypes_end_to_end1(adata_define_clonotypes):
    """Test that default parameters of define_clonotypes yields
    clonotypes based on nt-identity.
    """
    ir.pp.ir_dist(adata_define_clonotypes)
    clonotypes, clonotype_size, _ = ir.tl.define_clonotypes(
        adata_define_clonotypes,
        inplace=False,
        within_group=None,
        receptor_arms="all",
        dual_ir="all",
    )  # type: ignore
    print(clonotypes)
    expected = ["0", "0", "1", "2", np.nan]
    expected_size = [2, 2, 1, 1, np.nan]
    pdt.assert_extension_array_equal(clonotypes.array, pd.array(expected), check_dtype=False)
    npt.assert_equal(clonotype_size.values, expected_size)


@pytest.mark.parametrize(
    "adata_fixture,receptor_arms,dual_ir,within_group,uneven_vdj,expected_clonotypes,expected_size_values,expected_size_cols",
    [
        pytest.param(
            "adata_multichain",
            "VJ",
            "any",
            None,
            False,
            [[0, 3, 1, None, None, None], [1, None], [2]],
            [[30.0, 10.0, 0.0, 20.0], [0.0, 40.0, 0.0, 0.0], [0.0, 0.0, 50.0, 0.0]],
            [3],
            id="vj_any",
        ),
        pytest.param(
            "adata_multichain",
            "VJ",
            "any",
            "receptor_type",
            False,
            [[0, 3, 1, None, None, None], [1, None], [2]],
            [[30.0, 10.0, 0.0, 20.0], [0.0, 40.0, 0.0, 0.0], [0.0, 0.0, 50.0, 0.0]],
            [3],
            id="vj_any_within_group",
        ),
        pytest.param(
            "adata_multichain",
            "VJ",
            "primary_only",
            None,
            False,
            [[0, None, None, None, None, None], [1, None], [2]],
            [[30.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 50.0]],
            [2],
            id="vj_primary_only",
        ),
        pytest.param(
            "adata_multichain",
            "VDJ",
            "primary_only",
            None,
            False,
            [[None, None, None, 0, None, None], [None, 1], [None]],
            [[30.0, 0.0], [0.0, 40.0], [0.0, 0.0]],
            [1],
            id="vdj_primary_only",
        ),
        pytest.param(
            "adata_multichain",
            "VDJ",
            "any",
            None,
            False,
            [[None, None, None, 0, 2, 1], [None, 1], [None]],
            [[30.0, 10.0, 20.0], [0.0, 40.0, 0.0], [0.0, 0.0, 0.0]],
            [2],
            id="vdj_any",
        ),
        pytest.param(
            "adata_multichain",
            "any",
            "any",
            None,
            False,
            [
                [[0, 3, 1, None, None, None], [1, None], [2]],
                [[None, None, None, 0, 3, 1], [None, 1], [None]],
            ],
            [[60.0, 20.0, 0.0, 40.0], [0.0, 80.0, 0.0, 0.0], [0.0, 0.0, 50.0, 0.0]],
            [3, 3],
            id="any_any",
        ),
        pytest.param(
            "adata_multichain",
            "all",
            "any",
            None,
            False,
            [
                [[0, 3, 1, None, None, None], [1, None], [2]],
                [[None, None, None, 0, 3, 1], [None, 1], [None]],
            ],
            [[60.0, 20.0, 0.0, 40.0], [0.0, 80.0, 0.0, 0.0], [0.0, 0.0, 50.0, 0.0]],
            [3, 3],
            id="all_any",
        ),
        pytest.param(
            "adata_multichain",
            "any",
            "any",
            None,
            True,
            [
                [[0, 3, 1, None, None, None], [1, None], [2]],
                [[None, None, None, 0, None, None], [None, 1], [None]],
            ],
            [[60.0, 10.0, 0.0, 20.0], [0.0, 80.0, 0.0, 0.0], [0.0, 0.0, 50.0, 0.0]],
            [3, 1],
            id="any_any_uneven_arm_chain_counts",
        ),
        pytest.param(
            "adata_multichain_mixed_receptor_types",
            "any",
            "any",
            "receptor_type",
            False,
            [
                [[0, 1, 3, None, None, None], [0, 2, None], [1, None]],
                [[None, None, None, 1, 0, 4], [None, None, 0], [None, 1]],
            ],
            [[50.0, 50.0, 0.0, 10.0, 10.0], [80.0, 0.0, 10.0, 0.0, 0.0], [0.0, 80.0, 0.0, 0.0, 0.0]],
            [3, 4],
            id="any_any_mixed_receptor_type_within_group",
        ),
    ],
)
def test_define_clonotype_clusters_multi_model_return_values(
    request,
    adata_fixture,
    receptor_arms,
    dual_ir,
    within_group,
    uneven_vdj,
    expected_clonotypes,
    expected_size_values,
    expected_size_cols,
):
    """Multi-chain clonotype output is stable across public receptor-arm modes."""
    adata_multichain = request.getfixturevalue(adata_fixture)
    if uneven_vdj:
        adata_multichain = adata_multichain.copy()
        chain_indices = adata_multichain.obsm["chain_indices"]
        chain_indices["VDJ"] = ak.Array([[x[0]] for x in ak.to_list(chain_indices["VDJ"])])
        adata_multichain.obsm["chain_indices"] = chain_indices

    ir.pp.ir_dist(adata_multichain, metric="identity", sequence="aa")

    clonotypes, clonotype_size, res = ir.tl.define_clonotype_clusters(
        adata_multichain,
        inplace=False,
        within_group=within_group,
        receptor_arms=receptor_arms,
        dual_ir=dual_ir,
    )  # type: ignore
    if receptor_arms in ["any", "all"]:
        assert ak.to_list(clonotypes["VJ"]) == ak.to_list(ak.zip({"clone_id": expected_clonotypes[0]}))
        assert ak.to_list(clonotypes["VDJ"]) == ak.to_list(ak.zip({"clone_id": expected_clonotypes[1]}))
        assert np.max(clonotypes["VJ"]["clone_id"]) == expected_size_cols[0]
        assert np.max(clonotypes["VDJ"]["clone_id"]) == expected_size_cols[1]
    else:
        assert ak.to_list(clonotypes) == ak.to_list(ak.zip({"clone_id": expected_clonotypes}))
        assert np.max(clonotypes["clone_id"]) == expected_size_cols[0]
    #
    expected_size = np.array(expected_size_values)
    assert isinstance(clonotype_size, np.ndarray)
    npt.assert_equal(clonotype_size, expected_size)
    assert set(res) == {
        "distances",
        "cell_indices",
        "VJ_clone_chain_indices",
        "VDJ_clone_chain_indices",
        "clone_umi_count",
    }


def test_define_clonotype_clusters_multi_model_dual_ir_all_not_implemented(adata_multichain):
    ir.pp.ir_dist(adata_multichain, metric="identity", sequence="aa")

    with pytest.raises(NotImplementedError, match="dual_ir='all'"):
        ir.tl.define_clonotype_clusters(
            adata_multichain,
            inplace=False,
            within_group=None,
            receptor_arms="VJ",
            dual_ir="all",
        )


def test_define_clonotype_clusters_multi_model_within_group_requires_airr_field(adata_multichain):
    adata_multichain.obs["receptor_type"] = "TCR"
    adata_multichain.obsm["airr"] = ak.without_field(adata_multichain.obsm["airr"], "receptor_type")
    ir.pp.ir_dist(adata_multichain, metric="identity", sequence="aa")

    with pytest.raises(ValueError, match="column `receptor_type` not found in `airr`"):
        ir.tl.define_clonotype_clusters(
            adata_multichain,
            inplace=False,
            within_group="receptor_type",
            receptor_arms="VJ",
            dual_ir="any",
        )


@pytest.mark.parametrize(
    "receptor_arms,dual_ir,same_v_gene,within_group,expected,expected_size",
    [
        (
            "all",
            "all",
            False,
            None,
            [0, 0, 1, 2, 3, np.nan, 4, 5, 6, 7, np.nan],
            [2, 2, 1, 1, 1, np.nan, 1, 1, 1, 1, np.nan],
        ),
        (
            "any",
            "any",
            False,
            None,
            [0, 0, 0, 0, 0, np.nan, 0, 0, 0, 0, np.nan],
            [9, 9, 9, 9, 9, np.nan, 9, 9, 9, 9, np.nan],
        ),
        (
            "all",
            "any",
            False,
            None,
            [0, 0, 0, 0, 0, np.nan, 0, 1, 0, 2, np.nan],
            [7, 7, 7, 7, 7, np.nan, 7, 1, 7, 1, np.nan],
        ),
        (
            "any",
            "all",
            False,
            None,
            [0, 0, 0, 0, 0, np.nan, 0, 0, 0, 0, np.nan],
            [9, 9, 9, 9, 9, np.nan, 9, 9, 9, 9, np.nan],
        ),
        (
            "all",
            "primary_only",
            False,
            None,
            [0, 0, 1, 2, 0, np.nan, 0, 3, 4, 5, np.nan],
            [4, 4, 1, 1, 4, np.nan, 4, 1, 1, 1, np.nan],
        ),
        (
            "VDJ",
            "primary_only",
            False,
            None,
            [0, 0, 0, 1, 0, np.nan, 0, 2, 3, 3, np.nan],
            [5, 5, 5, 1, 5, np.nan, 5, 1, 2, 2, np.nan],
        ),
        # by receptor type
        (
            "any",
            "any",
            False,
            "receptor_type",
            [0, 0, 0, 1, 1, np.nan, 0, 0, 0, 1, np.nan],
            [6, 6, 6, 3, 3, np.nan, 6, 6, 6, 3, np.nan],
        ),
        # different combinations with same_v_gene
        (
            "all",
            "all",
            True,
            None,
            [0, 1, 2, 3, 4, np.nan, 5, 6, 7, 8, 9],
            [1, 1, 1, 1, 1, np.nan, 1, 1, 1, 1, 1],
        ),
        (
            "any",
            "any",
            True,
            None,
            [0, 0, 0, 1, 0, np.nan, 0, 0, 0, 0, 2],
            [8, 8, 8, 1, 8, np.nan, 8, 8, 8, 8, 1],
        ),
        (
            "VDJ",
            "primary_only",
            True,
            None,
            [0, 0, 0, 1, 0, np.nan, 0, 2, 3, 4, 5],
            [5, 5, 5, 1, 5, np.nan, 5, 1, 1, 1, 1],
        ),
        # v gene and receptor type
        (
            "any",
            "any",
            True,
            "receptor_type",
            [0, 0, 0, 1, 2, np.nan, 0, 0, 0, 2, 3],
            [6, 6, 6, 1, 2, np.nan, 6, 6, 6, 2, 1],
        ),
    ],
)
def test_clonotype_clusters_end_to_end(
    adata_define_clonotype_clusters,
    receptor_arms,
    dual_ir,
    same_v_gene,
    within_group,
    expected,
    expected_size,
):
    """Test define_clonotype_clusters with different parameters"""
    ir.pp.ir_dist(
        adata_define_clonotype_clusters,
        cutoff=0,
        sequence="aa",
    )
    clonotypes, clonotype_size, _ = ir.tl.define_clonotype_clusters(
        adata_define_clonotype_clusters,
        inplace=False,
        within_group=within_group,
        receptor_arms=receptor_arms,
        dual_ir=dual_ir,
        same_v_gene=same_v_gene,
    )  # type: ignore
    print(clonotypes)
    npt.assert_equal(list(clonotypes.values), [str(x) if not np.isnan(x) else x for x in expected])
    npt.assert_almost_equal(clonotype_size.values, expected_size)


@pytest.mark.extra
@pytest.mark.xfail(
    sys.platform == "win32",
    reason="Inconsistent coordinates with igraph on windows (got introduced only after release of python-igraph 0.9.11)",
)
@pytest.mark.parametrize(
    "min_cells,min_nodes,layout,size_aware,expected",
    (
        [
            1,
            1,
            "components",
            True,
            [
                [1.0, 76.29518321],
                [1.0, 76.29518321],
                [62.66666667, 63.91071673],
                [72.8511768, 62.66666667],
                [96.0, 46.84496659],
                [np.nan, np.nan],
                [3.6992887, 79.33333333],
                [56.69600481, 28.51130046],
                [16.75280038, 17.66666667],
                [67.66666667, 34.33333333],
                [np.nan, np.nan],
            ],
        ],
        [
            2,
            3,
            "fr",
            False,
            [
                [0.8168581312048484, -0.5998359351272509],
                [0.8168581312048484, -0.5998359351272509],
                [0.5102479235509972, 0.03402664585673332],
                [-2.1361996252554656, -0.00451326321092127],
                [-1.5094203782869742, -0.5570739804378876],
                [np.nan, np.nan],
                [1.4631495716508085, -0.37520604565881344],
                [0.8743828283132641, 0.6267343727751706],
                [1.4923560407323928, 0.32479780384697404],
                [-1.5113744919098704, 0.5510704019559949],
                [np.nan, np.nan],
            ],
        ],
    ),
)
def test_clonotype_network(adata_conn, min_cells, min_nodes, layout, size_aware, expected):
    coords = ir.tl.clonotype_network(
        adata_conn,
        sequence="aa",
        metric="alignment",
        min_cells=min_cells,
        min_nodes=min_nodes,
        size_aware=size_aware,
        layout=layout,
        inplace=False,
        mask_obs=None,
    )
    npt.assert_almost_equal(coords.values, np.array(expected), decimal=1)


@pytest.mark.extra
def test_clonotype_network_mask_obs(adata_conn):
    expected = [
        [34.854376, 96.0],
        [34.854376, 96.0],
        [93.076805, 27.643534],
        [np.nan, np.nan],
        [np.nan, np.nan],
        [np.nan, np.nan],
        [22.889353, 1.0],
        [1.0, 65.74282],
        [96.0, 51.195539],
        [np.nan, np.nan],
        [np.nan, np.nan],
    ]

    boolean_mask = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0], dtype=bool)
    adata_conn.obs["boolean_mask"] = boolean_mask

    coords1 = ir.tl.clonotype_network(
        adata_conn,
        sequence="aa",
        metric="alignment",
        min_cells=1,
        min_nodes=1,
        size_aware=True,
        layout="components",
        inplace=False,
        mask_obs=boolean_mask,
    )

    coords2 = ir.tl.clonotype_network(
        adata_conn,
        sequence="aa",
        metric="alignment",
        min_cells=1,
        min_nodes=1,
        size_aware=True,
        layout="components",
        inplace=False,
        mask_obs="boolean_mask",
    )

    npt.assert_almost_equal(coords1.values, np.array(expected), decimal=1)
    npt.assert_almost_equal(coords2.values, np.array(expected), decimal=1)


@pytest.mark.extra
def test_clonotype_network_igraph(adata_clonotype_network):
    g, lo = ir.tl.clonotype_network_igraph(adata_clonotype_network)
    print(lo.coords)
    print(g.vcount())
    assert g.vcount() == 8
    npt.assert_almost_equal(
        np.array(lo.coords),
        np.array(
            [
                [1.0, 3.0381501206112773],
                [62.66666666666667, 15.422616607972202],
                [3.699288696489563, 0.0],
                [56.696004811098774, 50.82203287163998],
                [16.75280038357383, 61.66666666666668],
                [72.85117680388525, 16.66666666666667],
                [96.0, 32.48836674202366],
                [67.66666666666667, 45.0],
            ]
        ),
    )


def test_clonotype_convergence(adata_clonotype):
    res = cast(
        pd.Series,
        ir.tl.clonotype_convergence(
            adata_clonotype,
            key_coarse="clonotype_cluster",
            key_fine="clone_id",
            inplace=False,
        ),
    )
    ir.tl.clonotype_convergence(
        adata_clonotype,
        key_coarse="clonotype_cluster",
        key_fine="clone_id",
        inplace=True,
        key_added="is_convergent_",
    )
    if isinstance(adata_clonotype, MuData):
        pdt.assert_series_equal(res, adata_clonotype.obs["airr:is_convergent_"], check_names=False)
        pdt.assert_series_equal(res, adata_clonotype["airr"].obs["is_convergent_"], check_names=False)
    else:
        pdt.assert_series_equal(res, adata_clonotype.obs["is_convergent_"], check_names=False)
    pdt.assert_extension_array_equal(
        res.values,
        pd.Categorical(
            ["not convergent"] * 3 + [np.nan] * 2 + ["not convergent"] + ["convergent"] * 2 + ["not convergent"] * 2,
            categories=["convergent", "not convergent"],
        ),
    )

    res = cast(
        pd.Series,
        ir.tl.clonotype_convergence(
            adata_clonotype,
            key_fine="clonotype_cluster",
            key_coarse="clone_id",
            inplace=False,
        ),
    )
    pdt.assert_extension_array_equal(
        res.values,
        pd.Categorical(
            ["not convergent"] * 3 + [np.nan] * 2 + ["not convergent"] * 5,
            categories=["convergent", "not convergent"],
        ),
    )


def test_j_gene_matching():
    data = ad.read_h5ad(TESTDATA / "clonotypes_test_data/j_gene_test_data.h5ad")

    ir.tl.define_clonotype_clusters(
        data,
        sequence="nt",
        metric="normalized_hamming",
        receptor_arms="all",
        dual_ir="any",
        same_j_gene=True,
        key_added="test_j_gene",
    )

    clustering = data.obs["test_j_gene"].tolist()
    expected = ["0", "0", "0", "0", "0", "1", "1", "1", "1", "1", "1", "1", "1", "2", "2", "2", "2", "2"]
    assert np.array_equal(clustering, expected)


def test_gene_segment_matching_index_error_issue_625():
    """Regression test for #625, where an IndexError occured during gene segment matching."""
    adata = ir.io.read_airr(TESTDATA / "airr" / "issue_625.tsv")
    ir.pp.index_chains(adata)
    ir.tl.chain_qc(adata)
    ir.pp.ir_dist(adata, metric="normalized_hamming", cutoff=15, sequence="nt", histogram=False)
    ir.tl.define_clonotype_clusters(
        adata,
        sequence="nt",
        metric="normalized_hamming",
        receptor_arms="all",
        dual_ir="any",
        same_v_gene=True,
        same_j_gene=True,
        partitions="fastgreedy",
        key_added="clone_id_similarity",
    )
