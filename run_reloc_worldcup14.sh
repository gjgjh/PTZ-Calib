#/bin/bash
set -e

./install/bin/run_ptz_reloc --ref_images data/worldcup14/offline/GER_ARG --ref_features data/worldcup14/offline_matches/GER_ARG --ref_params output-worldcup14-offline/GER_ARG.json --test_images data/worldcup14/online/ESP_CHI --test_features data/worldcup14/online_matches/ESP_CHI --output output-worldcup14-online
./install/bin/run_ptz_reloc --ref_images data/worldcup14/offline/GER_ARG --ref_features data/worldcup14/offline_matches/GER_ARG --ref_params output-worldcup14-offline/GER_ARG.json --test_images data/worldcup14/online/FRA_GER --test_features data/worldcup14/online_matches/FRA_GER --output output-worldcup14-online
./install/bin/run_ptz_reloc --ref_images data/worldcup14/offline/GER_POR --ref_features data/worldcup14/offline_matches/GER_POR --ref_params output-worldcup14-offline/GER_POR.json --test_images data/worldcup14/online/SUI_FRA --test_features data/worldcup14/online_matches/SUI_FRA --output output-worldcup14-online
./install/bin/run_ptz_reloc --ref_images data/worldcup14/offline/NED_ARG --ref_features data/worldcup14/offline_matches/NED_ARG --ref_params output-worldcup14-offline/NED_ARG.json --test_images data/worldcup14/online/ARG_SUI --test_features data/worldcup14/online_matches/ARG_SUI --output output-worldcup14-online
./install/bin/run_ptz_reloc --ref_images data/worldcup14/offline/NED_ARG --ref_features data/worldcup14/offline_matches/NED_ARG --ref_params output-worldcup14-offline/NED_ARG.json --test_images data/worldcup14/online/BRA_CRO --test_features data/worldcup14/online_matches/BRA_CRO --output output-worldcup14-online
./install/bin/run_ptz_reloc --ref_images data/worldcup14/offline/NED_ARG --ref_features data/worldcup14/offline_matches/NED_ARG --ref_params output-worldcup14-offline/NED_ARG.json --test_images data/worldcup14/online/URU_ENG --test_features data/worldcup14/online_matches/URU_ENG --output output-worldcup14-online
./install/bin/run_ptz_reloc --ref_images data/worldcup14/offline/USA_GER --ref_features data/worldcup14/offline_matches/USA_GER --ref_params output-worldcup14-offline/USA_GER.json --test_images data/worldcup14/online/CRO_MEX --test_features data/worldcup14/online_matches/CRO_MEX --output output-worldcup14-online

python scripts/eval_worldcup.py --gt_dir data/worldcup14/gt/test --pred output-worldcup14-online/ARG_SUI.json
python scripts/eval_worldcup.py --gt_dir data/worldcup14/gt/test --pred output-worldcup14-online/BRA_CRO.json
python scripts/eval_worldcup.py --gt_dir data/worldcup14/gt/test --pred output-worldcup14-online/CRO_MEX.json
python scripts/eval_worldcup.py --gt_dir data/worldcup14/gt/test --pred output-worldcup14-online/ESP_CHI.json
python scripts/eval_worldcup.py --gt_dir data/worldcup14/gt/test --pred output-worldcup14-online/FRA_GER.json
python scripts/eval_worldcup.py --gt_dir data/worldcup14/gt/test --pred output-worldcup14-online/SUI_FRA.json
python scripts/eval_worldcup.py --gt_dir data/worldcup14/gt/test --pred output-worldcup14-online/URU_ENG.json