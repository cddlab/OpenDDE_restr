# Restraint-Guided Inference

The `rgi-integration` branch of
[`cddlab/OpenDDE_restr`](https://github.com/cddlab/OpenDDE_restr) integrates
OpenDDE with [`rgi_utils`](https://github.com/cddlab/rgi_utils). RGI optimizes
the denoised coordinate estimate during every active diffusion step. It supports
distance, angle, dihedral, improper, plane, RMSD, base-pair, ligand/polymer
conformer, VdW, and custom restraints through the shared `restraints_config`
schema.

## Installation

The fork declares `rgi_utils` as a dependency:

```bash
git clone git@github.com:cddlab/OpenDDE_restr.git
cd OpenDDE_restr
git switch rgi-integration
uv venv --python 3.12
uv pip install --python .venv --torch-backend cu126 -e ".[gpu]"
```

For engine development, install a sibling checkout after installing OpenDDE:

```bash
uv pip install --python .venv -e ../rgi_utils
```

## Input

Put one `restraints_config` object beside `name` and `sequences` in each
job. Set `conformer_restraints: true` only on entities whose local geometry
should be restrained. RGI uses chain IDs from `id`; `resid` is the
per-chain, 1-based token ordinal.

```json
[
  {
    "name": "opendde_rgi_example",
    "modelSeeds": [0],
    "sequences": [
      {
        "proteinChain": {
          "sequence": "ACDEFGHIKLMNPQRSTVWY",
          "count": 1,
          "id": ["A"]
        }
      },
      {
        "ligand": {
          "ligand": "C/C=C\\C",
          "count": 1,
          "id": ["L"],
          "conformer_restraints": true
        }
      }
    ],
    "restraints_config": {
      "gpu": true,
      "method": "CG",
      "max_iter": 100,
      "verbose": true,
      "distance_restraints_config": [
        {
          "atom_selection1": "chain A and resid 1",
          "atom_selection2": "chain L",
          "harmonic": {"target_distance": 8.0},
          "start_sigma": 1.0
        }
      ],
      "conformer_restraints_config": {
        "start_sigma": 1.0,
        "bond": {"weight": 1.0},
        "angle": {"weight": 1.0},
        "chiral": {"weight": 1.0},
        "cistrans": {"weight": 1.0},
        "plane": {"weight": 1.0},
        "vdw": {"weight": 1.0}
      }
    }
  }
]
```

The complete schema, selection DSL, activation windows, reference restraints,
and custom-expression vocabulary are documented in the
[`rgi_utils` configuration reference](https://github.com/cddlab/rgi_utils/blob/main/doc/config.md).

## Run

```bash
uv run --no-project --python .venv opendde pred \
  -i rgi_input.json \
  -o output_rgi \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 10
```

No RGI-specific CLI flag is required. Omitting `restraints_config` preserves
the upstream sampling path.

OpenDDE's native Training-Free Guidance remains available with
`--use_tfg_guidance true`. When both systems are enabled, each step applies
native TFG refinement first and RGI second.

Completed seed/sample outputs are detected from their CIF and summary-confidence
files and skipped on rerun, so interrupted array jobs can be resubmitted safely.

With `verbose: true`, verify that the `built spec:` line reports non-zero
counts for every requested restraint. The output layout is unchanged:

```text
<out_dir>/<job_name>/seed_<seed>/predictions/
```
