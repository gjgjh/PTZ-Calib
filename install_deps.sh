#!/bin/bash
set -e
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )" # root directory, for example: /root/code/PTZ-Calib
cwd=$SCRIPTPATH
mkdir -p install 3rdparty
OPTPATH=$cwd/install
echo 'Installing tools and libraries to: ' $OPTPATH

function init_bashrc {
    mkdir -p $OPTPATH/lib
    mkdir -p $OPTPATH/lib64
    mkdir -p $OPTPATH/bin
    mkdir -p $OPTPATH/include

    if ! grep "OPTPATH" $OPTPATH/bashrc 1> /dev/null 2> /dev/null; then
        echo '# added for new opt-installation prefix' >> $OPTPATH/bashrc
        echo export OPTPATH=$OPTPATH >> $OPTPATH/bashrc
        echo export PATH=\$OPTPATH/bin:"\$PATH" >> $OPTPATH/bashrc
        echo export LD_LIBRARY_PATH=\$OPTPATH/lib:\$OPTPATH/lib64:\$LD_LIBRARY_PATH >> $OPTPATH/bashrc
        echo export LIBRARY_PATH=\$OPTPATH/lib:\$OPTPATH/lib64:\$LIBRARY_PATH >> $OPTPATH/bashrc
        echo export C_INCLUDE_PATH=\$OPTPATH/include:\$C_INCLUDE_PATH >> $OPTPATH/bashrc
        echo export CPLUS_INCLUDE_PATH=\$OPTPATH/include:\$CPLUS_INCLUDE_PATH >> $OPTPATH/bashrc
        echo export PKG_CONFIG_PATH=\$OPTPATH/lib/pkgconfig:\$OPTPATH/lib64/pkgconfig:\$PKG_CONFIG_PATH >> $OPTPATH/bashrc
        echo export CMAKE_PREFIX_PATH=\$OPTPATH:\$CMAKE_PREFIX_PATH >> $OPTPATH/bashrc
    fi

    source $OPTPATH/bashrc
}

function install_json {
    echo "Installing nlohmann-json ..."
    cd $cwd/3rdparty
    curl -L -o json.zip https://github.com/nlohmann/json/archive/refs/tags/v3.9.1.zip
    unzip -qq json.zip && rm -rf json.zip
}

function install_plog {
    echo "Installing plog ..."
    cd $cwd/3rdparty
    curl -L -o plog.zip https://github.com/SergiusTheBest/plog/archive/refs/tags/1.1.5.zip
    unzip -qq plog.zip && rm -rf plog.zip
}

function install_opencv {
    echo "Installing opencv-4.5.3 ..."
    cd $cwd/3rdparty
    
    rm -rf opencv-4.5.3
    curl -L -o opencv-4.5.3.zip https://github.com/opencv/opencv/archive/refs/tags/4.5.3.zip
    unzip -qq opencv-4.5.3.zip && rm -rf opencv-4.5.3.zip

    cd $cwd/3rdparty/opencv-4.5.3
    mkdir -p build
    cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX=$OPTPATH \
        -DBUILD_SHARED_LIBS=OFF \
        -DWITH_FFMPEG=ON \
        -DBUILD_DOCS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_TESTS=OFF \
        -DENABLE_PRECOMPILED_HEADERS=OFF \
        -DWITH_IPP=OFF \
        -DWITH_EIGEN=OFF \
        -DWITH_PNG=ON \
        -DWITH_LAPACK=OFF
    make -j32 install
}

function install_gflags {
    echo "Installing gflags-2.2.2 ..."
    cd $cwd/3rdparty
    rm -rf gflags-2.2.2
    curl -L -o gflags-2.2.2.tar.gz https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.tar.gz
    tar xzf gflags-2.2.2.tar.gz && rm -rf gflags-2.2.2.tar.gz

    cd $cwd/3rdparty/gflags-2.2.2
    mkdir -p build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$OPTPATH -DBUILD_SHARED_LIBS=ON ..
    make -j32 install
}

function install_glog {
    echo "Installing glog-0.3.4 ..."
    cd $cwd/3rdparty
    rm -rf glog-0.3.4
    curl -L -o glog-0.3.4.tar.gz https://github.com/google/glog/archive/refs/tags/v0.3.4.tar.gz
    tar xzf glog-0.3.4.tar.gz && rm -rf glog-0.3.4.tar.gz

    cd $cwd/3rdparty/glog-0.3.4
    ./configure  --enable-shared --disable-static --prefix=$OPTPATH
    make -j32 install
}

function install_eigen {
    echo "Installing eigen-3.3.7 ..."
    cd $cwd/3rdparty
    curl -L -o eigen-3.3.7.tar.gz https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
    tar xzf eigen-3.3.7.tar.gz && rm -rf eigen-3.3.7.tar.gz

    cd $cwd/3rdparty/eigen-3.3.7
    mkdir -p build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$OPTPATH
    make -j32 install
}

function install_ceres {
    install_gflags
    install_glog
    install_eigen

    echo "Installing ceres-1.14.0 ..."
    cd $cwd/3rdparty
    curl -L -o ceres-solver-1.14.0.tar.gz https://github.com/ceres-solver/ceres-solver/archive/refs/tags/1.14.0.tar.gz
    tar xzf ceres-solver-1.14.0.tar.gz && rm -rf ceres-solver-1.14.0.tar.gz

    cd $cwd/3rdparty/ceres-solver-1.14.0
    mkdir -p build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$OPTPATH -DBUILD_SHARED_LIBS=ON
    make -j32 install
}

function install_all_deps {
    init_bashrc
    install_json
    install_plog
    install_opencv
    install_ceres
}

$1