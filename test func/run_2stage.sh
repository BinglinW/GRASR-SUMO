#!/bin/bash
#SBATCH -p paratera
#SBATCH -N 2
#SBATCH -n 48
#SBATCH --tasks-per-node=24
#SBATCH -J SR_py
source /PARA/app/scripts/cn-module.sh
module load anaconda2/2019.07
source activate py373
#module load R/4.1.0 gcc/8.3.0
# module load gcc/8.3.0

cd ~/wbl/new_test_fun
if [ -e result_2stage.csv ];then
    rm slurm*
    rm result*
    rm output*
    rm max*
    rm min*
    rm MPV*
fi
#chmod u+x copy.sh
#./copy.sh
chmod u+x SUMO_main_2stage.py
echo $(date)
trap "exec 10>&-;exec 10<&-;exit 0" 2
tmp_fifo=$$.fifo
mkfifo $tmp_fifo
exec 10<>$tmp_fifo
rm -rf $tmp_fifo
let thread_num=2*$(grep "processor" /proc/cpuinfo | sort -u | wc -l)
for ((i=0;i<$thread_num;i++))
do
    echo >&10
done
for ((j=0;j<40;j++))# file_i
do
  for ((i=1;i<21;i++))# n_min ######
  do
    read -u10
    {
        srun -n 1 -c 1 ~/.conda/envs/py373/bin/python ./SUMO_main_2stage.py -n ${j} -i ${i} &> output$i
        echo >&10
    } &
  done
done
wait
exec 10>&-
# if [ -e RMSE_2stage.txt ];then
#     rm RMSE_2stage.txt
#     rm max_2stage.txt
#     rm min_2stage.txt
#     rm MPV_2stage.txt
# fi

# ls result*.txt | sort -V | xargs paste -d ' '> RMSE_2stage.txt
# ls result*.txt | xargs rm

# ls max*.txt | sort -V | xargs paste -d ' '> max_2stage.txt
# ls max*.txt | grep -v "max_2stage.txt" | xargs rm

# ls min*.txt | sort -V | xargs paste -d ' '> min_2stage.txt
# ls min*.txt | grep -v "min_2stage.txt" | xargs rm

# ls MPV*.txt | sort -V | xargs paste -d ' '> MPV_2stage.txt
# ls MPV*.txt | grep -v "MPV_2stage.txt" | xargs rm

echo $(date)
