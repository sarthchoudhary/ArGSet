# this code didn't work
import sys
for i in range(int(sys.argv[4]), int(sys.argv[5])+1):
    run_folder_name = '_'.join(['Run', str(i)])
    exec(f"/home/sarthak/miniconda3/envs/simple_env/bin/python 
    /home/sarthak/my_projects/argset/process_over_entire_run({run_folder_name}, sys.argv[2], sys.argv[3])")