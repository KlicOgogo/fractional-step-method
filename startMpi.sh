#!/usr/bin/env bash
/home/alexandr/magistratura/mpi/installationDir/bin/mpic++ -o mainMpiVersion mainMpiVersion.cpp  startConditions.cpp startConditions.h

/home/alexandr/magistratura/mpi/installationDir/bin/mpiexec -n 2 mainMpiVersion


#/home/alexandr/magistratura/mpi/installationDir/bin/mpic++ -o mainMpiVersion test.cpp
#
#/home/alexandr/magistratura/mpi/installationDir/bin/mpiexec -n 2 mainMpiVersion
