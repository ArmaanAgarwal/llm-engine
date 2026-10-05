# SketchBranch

Package query evaluation algorithms: SketchRefine, Branching, and our hybrid SketchBranch.

## Structure

```
SketchBranch/
├── shared/                     Shared code used by all algorithms
│   ├── package_query.py          PaQL data model (PackageQuery, constraints, objective)
│   ├── ilp_solver.py             ILP translation rules + Direct solver
│   └── data_gen.py               Synthetic data generators
│
├── sketchrefine/               Algorithm 1: SketchRefine (Brucato et al., VLDB 2018)
│   ├── partitioning.py           Offline quad-tree partitioning
│   ├── sketchrefine.py           Algorithms 1, 2, 3 from the paper
│   └── test_sketchrefine.py      Tests
│
├── branching/                  Algorithm 2: Branching (Rohwedder & Wegrzycki, 2024)
│   ├── branching.py              Algorithm 1 from the paper
│   └── test_branching.py         Tests
│
├── sketchbranch/               Algorithm 3: SketchBranch (our proposal, TODO)
│   ├── sketchbranch.py           Sketch + Branch hybrid
│   └── test_sketchbranch.py      Tests
│
├── data/                       CSV data for experiments
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
# All tests
pytest */test_*.py -v

# Just one algorithm
pytest sketchrefine/test_sketchrefine.py -v
pytest branching/test_branching.py -v
pytest sketchbranch/test_sketchbranch.py -v
```
