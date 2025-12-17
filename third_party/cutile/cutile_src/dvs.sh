#!/bin/bash
# NVIDIA_COPYRIGHT_BEGIN
#
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# NVIDIA_COPYRIGHT_END

#
# This shell script is intended to be used by DVS to invoke a UNIX
# build of the nvOmega library and create an intermediate
# tarball containing the files to be propagated to the machine that
# will create the final package.
#
# See the comments at the top of drivers/common/build/unix/dvs-util.sh for
# usage details.

# determine NV_SOURCE by cd'ing to the directory containing this script,
# and then backing up the appropriate number of directories
cd `dirname $0`
cd ../../../../..

# Default values
export SBSA=0
export BUILD_TILEIRAS_LIBRARY=0

# Check for flags in arguments
for arg in "$@"; do
  case "$arg" in
    SBSA=1) export SBSA=1 ;;
    BUILD_TILEIRAS_LIBRARY=1) export BUILD_TILEIRAS_LIBRARY=1 ;;
  esac
done

nv_source=`pwd`
# include the helper functions; this also parses the commandline
. ${nv_source}/drivers/common/build/unix/dvs-util.sh

# assign variables needed below and in the helper functions called below
if [ "$BUILD_TILEIRAS_LIBRARY" = "1" ] ; then
    assign_common_variables tileiraslib
    tileiraslib_dir=drivers/compiler/cuda_tile/tileir
    tileiraslib_outputdir=`get_component_outputdir "${tileiraslib_dir}"`
    # build the library
    run_nvmake ${nv_source}/${tileiraslib_dir} BUILD_TILEIRAS_LIBRARY=1
else
    assign_common_variables tileiras
    tileiras_dir=drivers/compiler/cuda_tile/tileir
    tileiras_outputdir=`get_component_outputdir "${tileiras_dir}"`
    # build the library
    run_nvmake ${nv_source}/${tileiras_dir} BUILD_TILEIRAS_LIBRARY=0
fi

# check if we ran nvmake on WSL
wsl_build=0

for nvmake_arg in ${nvmake_args} ; do
    if [ "$nvmake_arg" == "winnext" ] ; then wsl_build=1 ; fi
    if [ "$nvmake_arg" == "winfuture" ] ; then wsl_build=1 ; fi
    if [ "$nvmake_arg" == "wddm2" ] ; then wsl_build=1 ; fi
done
echo "nvmake arg is $nvmake_arg"

if [ $wsl_build -eq 0 ] && [ "$BUILD_TILEIRAS_LIBRARY" = "1" ] ; then
  tar_output_files --xz ${tileiraslib_outputdir}/lib/libnvidia-tileiras.so*
fi

# success
exit 0
