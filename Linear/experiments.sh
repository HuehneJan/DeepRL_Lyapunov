mkdir -p ./loggs
printf "Starting Linear System Analysis Experiments \n\n" > ./loggs/progress_information.txt

# DDPG Algorithm

echo "DDPG with System 1. Started at: " >> ./loggs/progress_information.txt
echo $(date) >> ./loggs/progress_information.txt
printf "\n" >> ./loggs/progress_information.txt
python3 train.py --save_dir=ddpg_1 --run_name=ddpg_1 --env=1 --epoch_number=40000 --tau=0.001 --algorithm=DDPG > ./loggs/ddpg_1.txt
python3 plotter.py --save_dir=ddpg_1 --reference=1
echo "DDPG with System 1. Ended at: " >> ./loggs/progress_information.txt
echo $(date) >> ./loggs/progress_information.txt
printf "\n" >> ./loggs/progress_information.txt

echo "DDPG with System 2. Started at: " >> ./loggs/progress_information.txt
echo $(date) >> ./loggs/progress_information.txt
printf "\n" >> ./loggs/progress_information.txt
printf "DDPG with System 2. Started at: $s \n "%now >> ./loggs/progress_information.txt
python3 train.py --save_dir=ddpg_2 --run_name=ddpg_2 --env=2 --epoch_number=40000 --tau=0.001 --algorithm=DDPG > ./loggs/ddpg_2.txt
python3 plotter.py --save_dir=ddpg_2 --reference=2
echo "DDPG with System 2. Ended at: " >> ./loggs/progress_information.txt
echo $(date) >> ./loggs/progress_information.txt
printf "\n" >> ./loggs/progress_information.txt


# TD3 Algorithm

echo "TD3 with System 1. Started at: " >> ./loggs/progress_information.txt
echo $(date) >> ./loggs/progress_information.txt
printf "\n" >> ./loggs/progress_information.txt
python3 train.py --save_dir=td3_1 --run_name=td3_1 --env=1 --epoch_number=40000 --algorithm=TD3 > ./loggs/td3_1.txt
python3 plotter.py --save_dir=td3_1 --reference=1
echo "TD3 with System 1. Ended at: " >> ./loggs/progress_information.txt
echo $(date) >> ./loggs/progress_information.txt
printf "\n" >> ./loggs/progress_information.txt

echo "TD3 with System 2. Started at: " >> ./loggs/progress_information.txt
echo $(date) >> ./loggs/progress_information.txt
printf "\n" >> ./loggs/progress_information.txt
printf "TD3 with System 2. Started at: $s \n "%now >> ./loggs/progress_information.txt
python3 train.py --save_dir=td3_2 --run_name=td3_2 --env=2 --epoch_number=40000 --algorithm=TD3 > ./loggs/td3_2.txt
python3 plotter.py --save_dir=td3_2 --reference=2
echo "TD3 with System 2. Ended at: " >> ./loggs/progress_information.txt
echo $(date) >> ./loggs/progress_information.txt
printf "\n" >> ./loggs/progress_information.txt

printf "Finished Linear System Analysis Experiments at" > ./loggs/progress_information.txt
echo $(date) >> ./loggs/progress_information.txt
