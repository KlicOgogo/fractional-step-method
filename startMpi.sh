#!/usr/bin/env bash
#/home/alexandr/magistratura/mpi/installationDir/bin/mpic++ -o mainMpiVersion mainMpiVersion.cpp  startConditions.cpp startConditions.h
#
#for N in 9 49 99 199 299 399
#do
#    for processesCount in 1 2 3 4
#    do
#        /home/alexandr/magistratura/mpi/installationDir/bin/mpiexec -n $processesCount mainMpiVersion $N output$processesCount
#    done
#done


/data/local/bin/mpic++ -o mainMpiVersion mainMpiVersion.cpp  startConditions.cpp startConditions.h
for N in 9 49 99 199 299 399
do
    for processesCount in 1 4 8 12 16 20
    do
        /data/local/bin/mpiexec --allow-run-as-root -n $processesCount --hostfile /data/fractional-step-method/hosts mainMpiVersion $N output$processesCount
#        /home/alexandr/magistratura/mpi/installationDir/bin/mpiexec -n $processesCount mainMpiVersion $N output$processesCount
    done
done


#/home/alexandr/magistratura/mpi/installationDir/bin/mpic++ -o mainMpiVersion test.cpp
#
#/home/alexandr/magistratura/mpi/installationDir/bin/mpiexec -n 2 mainMpiVersion


#
#/data/fractional-step-method/installationDir/bin/mpic++ -o mainMpiVersion mainMpiVersion.cpp  startConditions.cpp startConditions.h
#
#/data/fractional-step-method/installationDir/bin/mpiexec --allow-run-as-root -n 2 -host master,node001 mainMpiVersion
#/data/fractional-step-method/installationDir/bin/mpiexec --allow-run-as-root -n 1 mainMpiVersion

#export PATH=/data/fractional-step-method/installationDir/bin:$PATH


#/data/local/bin/mpic++ -o mainMpiVersion mainMpiVersion.cpp  startConditions.cpp startConditions.h
#/data/local/bin/mpiexec --allow-run-as-root -n 1 mainMpiVersion
#/data/local/bin/mpiexec --allow-run-as-root -n 1 --hostfile /data/fractional-step-method/hosts mainMpiVersion
#/data/local/bin/mpiexec --allow-run-as-root -n 3 --host 172.31.27.51,172.31.27.25 mainMpiVersion

