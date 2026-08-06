# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from opendde.data.inference.json_parser import build_polymer, rdkit_mol_to_atom_array
from opendde.data.inference import json_to_feature
from opendde.data.msa.msa_utils import map_to_standard


def test_build_polymer_rejects_out_of_range_ptm_position():
    with pytest.raises(ValueError, match="ptmPosition 5 is out of range"):
        build_polymer(
            {
                "proteinChain": {
                    "sequence": "AC",
                    "modifications": [{"ptmPosition": 5, "ptmType": "CCD_MSE"}],
                }
            }
        )


def test_map_to_standard_rejects_unknown_chain():
    meta = {0: {"sequence": "AAA"}, 1: {"sequence": "BB"}}
    with pytest.raises(ValueError, match="could not map residues"):
        map_to_standard(np.array([9]), np.array([1]), meta)


def test_constraint_field_warns_that_it_is_ignored(monkeypatch, caplog):
    monkeypatch.setattr(
        json_to_feature,
        "add_entity_atom_array",
        lambda sample: {"sequences": sample["sequences"]},
    )

    with caplog.at_level("WARNING"):
        sample = json_to_feature.SampleDictToFeatures(
            {"sequences": [], "constraint": {"contact": []}}
        )

    assert sample.entity_poly_type == {}
    assert "constraint" in caplog.text
    assert "ignored" in caplog.text


def test_rdkit_atom_array_preserves_double_bond_order():
    mol = Chem.AddHs(Chem.MolFromSmiles("C=C"))
    assert AllChem.EmbedMolecule(mol, randomSeed=0) == 0

    atom_array = rdkit_mol_to_atom_array(mol)
    bond_orders = atom_array.bonds.as_array()[:, 2].tolist()

    assert 2 in bond_orders


def test_rgi_metadata_is_preserved_in_features(monkeypatch):
    monkeypatch.setattr(
        "opendde.data.core.ccd.get_mol_type", lambda _res_name: "ligand"
    )
    monkeypatch.setattr(
        "opendde.data.core.ccd.get_one_letter_code", lambda _res_name: "X"
    )
    restraints_config = {
        "gpu": False,
        "conformer_restraints_config": {"bond": {}},
    }
    converter = json_to_feature.SampleDictToFeatures(
        {
            "name": "rgi_metadata",
            "sequences": [
                {
                    "ligand": {
                        "ligand": "C/C=C\\C",
                        "count": 1,
                        "id": ["L"],
                        "conformer_restraints": True,
                    }
                }
            ],
            "restraints_config": restraints_config,
        }
    )

    features, atom_array, _ = converter.get_feature_dict()

    assert features["atom_array"] is atom_array
    assert features["restraints_config"] == restraints_config
    assert features["smiles_by_chain"] == {"L": "C/C=C\\C"}
    assert atom_array.conformer_restraints.all()
