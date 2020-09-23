import os
import sys
import time
import subprocess


while True:

    all_files = os.listdir('.')

    policies = map(lambda policy: policy[7:] , filter(lambda file: file.startswith('policy'), all_files))

    #all_images  = os.listdir(''

    all_images = map(lambda file: file[file.rindex('_')+1:file.index('.gif')], os.listdir('./images/rl/'))

    difference = list(set(policies) - set(all_images))

    print(difference)

    for policy in difference:
        subprocess.call(['python3.8', 'playBreakout.py', policy])
    time.sleep(5 * 60)



#print(list(all_images))
