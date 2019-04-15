#!/usr/bin/env bash

set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$SCRIPT_DIR/../"

cd $REPO_ROOT
mpic++ -std=c++14 -I $REPO_ROOT -O2 natural-parallelism/sequential.cc common/test_functions.cc -o sequential.out
mpic++ -std=c++14 -I $REPO_ROOT -O2 natural-parallelism/mpi_version.cc common/test_functions.cc -o mpi_version_improved.out
