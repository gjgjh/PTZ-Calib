#/bin/bash
set -e

./install/bin/run_ptz_ba -i data/worldcup14/offline/GER_ARG -f data/worldcup14/offline_matches/GER_ARG -a data/worldcup14/offline/GER_ARG/GER_ARG.json -o output-worldcup14-offline
./install/bin/run_ptz_ba -i data/worldcup14/offline/GER_POR -f data/worldcup14/offline_matches/GER_POR -a data/worldcup14/offline/GER_POR/GER_POR.json -o output-worldcup14-offline
./install/bin/run_ptz_ba -i data/worldcup14/offline/NED_ARG -f data/worldcup14/offline_matches/NED_ARG -a data/worldcup14/offline/NED_ARG/NED_ARG.json -o output-worldcup14-offline
./install/bin/run_ptz_ba -i data/worldcup14/offline/USA_GER -f data/worldcup14/offline_matches/USA_GER -a data/worldcup14/offline/USA_GER/USA_GER.json -o output-worldcup14-offline