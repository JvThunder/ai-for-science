#!/usr/bin/env bash
set -e

cd /home/ubuntu/Volume-Virginia/ai-for-science

python src/reconstruct_polymer_nylon66.py
python src/reconstruct_polymer_peg.py
python src/reconstruct_polymer_polyester.py
python src/reconstruct_polymer_polyethelyn.py
python src/reconstruct_polymer_polytetrafluoroethylene.py
python src/reconstruct_polymer_polystyrene.py
python src/reconstruct_cyclohexane.py
