#/bin/bash
set -e

./install/bin/run_ptz_ba -i data/synthetic/offline/scene_01 -f data/synthetic/offline_matches/scene_01 -a data/synthetic/offline/scene_01/scene_01.json -o output-synthetic-offline
./install/bin/run_ptz_ba -i data/synthetic/offline/scene_02 -f data/synthetic/offline_matches/scene_02 -a data/synthetic/offline/scene_02/scene_02.json -o output-synthetic-offline
./install/bin/run_ptz_ba -i data/synthetic/offline/scene_03 -f data/synthetic/offline_matches/scene_03 -a data/synthetic/offline/scene_03/scene_03.json -o output-synthetic-offline
./install/bin/run_ptz_ba -i data/synthetic/offline/scene_04 -f data/synthetic/offline_matches/scene_04 -a data/synthetic/offline/scene_04/scene_04.json -o output-synthetic-offline
./install/bin/run_ptz_ba -i data/synthetic/offline/scene_05 -f data/synthetic/offline_matches/scene_05 -a data/synthetic/offline/scene_05/scene_05.json -o output-synthetic-offline
./install/bin/run_ptz_ba -i data/synthetic/offline/scene_06 -f data/synthetic/offline_matches/scene_06 -a data/synthetic/offline/scene_06/scene_06.json -o output-synthetic-offline
./install/bin/run_ptz_ba -i data/synthetic/offline/scene_07 -f data/synthetic/offline_matches/scene_07 -a data/synthetic/offline/scene_07/scene_07.json -o output-synthetic-offline
./install/bin/run_ptz_ba -i data/synthetic/offline/scene_08 -f data/synthetic/offline_matches/scene_08 -a data/synthetic/offline/scene_08/scene_08.json -o output-synthetic-offline
./install/bin/run_ptz_ba -i data/synthetic/offline/scene_09 -f data/synthetic/offline_matches/scene_09 -a data/synthetic/offline/scene_09/scene_09.json -o output-synthetic-offline
./install/bin/run_ptz_ba -i data/synthetic/offline/scene_10 -f data/synthetic/offline_matches/scene_10 -a data/synthetic/offline/scene_10/scene_10.json -o output-synthetic-offline

python scripts/eval_synthetic.py --pred output-synthetic-offline/scene_01.json --gt data/synthetic/gt/scene_01.json
python scripts/eval_synthetic.py --pred output-synthetic-offline/scene_02.json --gt data/synthetic/gt/scene_02.json
python scripts/eval_synthetic.py --pred output-synthetic-offline/scene_03.json --gt data/synthetic/gt/scene_03.json
python scripts/eval_synthetic.py --pred output-synthetic-offline/scene_04.json --gt data/synthetic/gt/scene_04.json
python scripts/eval_synthetic.py --pred output-synthetic-offline/scene_05.json --gt data/synthetic/gt/scene_05.json
python scripts/eval_synthetic.py --pred output-synthetic-offline/scene_06.json --gt data/synthetic/gt/scene_06.json
python scripts/eval_synthetic.py --pred output-synthetic-offline/scene_07.json --gt data/synthetic/gt/scene_07.json
python scripts/eval_synthetic.py --pred output-synthetic-offline/scene_08.json --gt data/synthetic/gt/scene_08.json
python scripts/eval_synthetic.py --pred output-synthetic-offline/scene_09.json --gt data/synthetic/gt/scene_09.json
python scripts/eval_synthetic.py --pred output-synthetic-offline/scene_10.json --gt data/synthetic/gt/scene_10.json