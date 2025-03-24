#!/bin/bash
set -e
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )" # root directory, for example: /root/code/PTZ-Calib
cwd=$SCRIPTPATH
OPTPATH=$cwd/install
rm -rf build
mkdir -p build
source $OPTPATH/bashrc

echo "Configuring and building PTZ-Calib ..."
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$OPTPATH \
      ..

make -j32 install
cd $cwd