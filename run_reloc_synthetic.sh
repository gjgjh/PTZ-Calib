#/bin/bash
set -e

./install/bin/run_ptz_reloc --ref_images data/synthetic/offline/scene_01 --ref_features data/synthetic/offline_matches/scene_01 --ref_params output-synthetic-offline/scene_01.json --test_images data/synthetic/online/scene_01 --test_features data/synthetic/online_matches/scene_01 --output output-synthetic-online
./install/bin/run_ptz_reloc --ref_images data/synthetic/offline/scene_02 --ref_features data/synthetic/offline_matches/scene_02 --ref_params output-synthetic-offline/scene_02.json --test_images data/synthetic/online/scene_02 --test_features data/synthetic/online_matches/scene_02 --output output-synthetic-online
./install/bin/run_ptz_reloc --ref_images data/synthetic/offline/scene_03 --ref_features data/synthetic/offline_matches/scene_03 --ref_params output-synthetic-offline/scene_03.json --test_images data/synthetic/online/scene_03 --test_features data/synthetic/online_matches/scene_03 --output output-synthetic-online
./install/bin/run_ptz_reloc --ref_images data/synthetic/offline/scene_04 --ref_features data/synthetic/offline_matches/scene_04 --ref_params output-synthetic-offline/scene_04.json --test_images data/synthetic/online/scene_04 --test_features data/synthetic/online_matches/scene_04 --output output-synthetic-online
./install/bin/run_ptz_reloc --ref_images data/synthetic/offline/scene_05 --ref_features data/synthetic/offline_matches/scene_05 --ref_params output-synthetic-offline/scene_05.json --test_images data/synthetic/online/scene_05 --test_features data/synthetic/online_matches/scene_05 --output output-synthetic-online
./install/bin/run_ptz_reloc --ref_images data/synthetic/offline/scene_06 --ref_features data/synthetic/offline_matches/scene_06 --ref_params output-synthetic-offline/scene_06.json --test_images data/synthetic/online/scene_06 --test_features data/synthetic/online_matches/scene_06 --output output-synthetic-online
./install/bin/run_ptz_reloc --ref_images data/synthetic/offline/scene_07 --ref_features data/synthetic/offline_matches/scene_07 --ref_params output-synthetic-offline/scene_07.json --test_images data/synthetic/online/scene_07 --test_features data/synthetic/online_matches/scene_07 --output output-synthetic-online
./install/bin/run_ptz_reloc --ref_images data/synthetic/offline/scene_08 --ref_features data/synthetic/offline_matches/scene_08 --ref_params output-synthetic-offline/scene_08.json --test_images data/synthetic/online/scene_08 --test_features data/synthetic/online_matches/scene_08 --output output-synthetic-online
./install/bin/run_ptz_reloc --ref_images data/synthetic/offline/scene_09 --ref_features data/synthetic/offline_matches/scene_09 --ref_params output-synthetic-offline/scene_09.json --test_images data/synthetic/online/scene_09 --test_features data/synthetic/online_matches/scene_09 --output output-synthetic-online
./install/bin/run_ptz_reloc --ref_images data/synthetic/offline/scene_10 --ref_features data/synthetic/offline_matches/scene_10 --ref_params output-synthetic-offline/scene_10.json --test_images data/synthetic/online/scene_10 --test_features data/synthetic/online_matches/scene_10 --output output-synthetic-online

python scripts/eval_synthetic.py --pred output-synthetic-online/scene_01.json --gt data/synthetic/gt/scene_01.json
python scripts/eval_synthetic.py --pred output-synthetic-online/scene_02.json --gt data/synthetic/gt/scene_02.json
python scripts/eval_synthetic.py --pred output-synthetic-online/scene_03.json --gt data/synthetic/gt/scene_03.json
python scripts/eval_synthetic.py --pred output-synthetic-online/scene_04.json --gt data/synthetic/gt/scene_04.json
python scripts/eval_synthetic.py --pred output-synthetic-online/scene_05.json --gt data/synthetic/gt/scene_05.json
python scripts/eval_synthetic.py --pred output-synthetic-online/scene_06.json --gt data/synthetic/gt/scene_06.json
python scripts/eval_synthetic.py --pred output-synthetic-online/scene_07.json --gt data/synthetic/gt/scene_07.json
python scripts/eval_synthetic.py --pred output-synthetic-online/scene_08.json --gt data/synthetic/gt/scene_08.json
python scripts/eval_synthetic.py --pred output-synthetic-online/scene_09.json --gt data/synthetic/gt/scene_09.json
python scripts/eval_synthetic.py --pred output-synthetic-online/scene_10.json --gt data/synthetic/gt/scene_10.json