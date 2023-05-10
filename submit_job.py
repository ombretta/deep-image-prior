import os

submit_on_cluster = True

cluster_text = ''
if submit_on_cluster:
    cluster_text = '#!/bin/sh\n'+\
    '#SBATCH --partition=general\n'+\
    '#SBATCH --qos=short\n'+\
    '#SBATCH --time=1:00:00\n'+\
    '#SBATCH --ntasks=1\n'+\
    '#SBATCH --mail-type=ALL\n'+\
    '#SBATCH --cpus-per-task=4\n'+\
    '#SBATCH --mem=4000\n'+\
    '#SBATCH --gres=gpu:1\n'+\
    'module use /opt/insy/modulefiles\n'+\
    'module load cuda/11.2 cudnn/11.2-8.1.1.33\n'+\
    'srun '

task = 'inpainting'
model = 'ce_skip_depth4' # one of skip_depth6|skip_depth4|skip_depth2|UNET|ResNet
image = 'kate' # one of vase|library|kate

cluster_text += "python "+task+".py --model "+model+" --image "+image
text = cluster_text

print(text)

if submit_on_cluster:
    with open(task+".sbatch", "w") as file:
        file.write(text)
    os.system("sbatch "+task+".sbatch")
else:
    os.system(text)




