#TODO: Setup abci group name
GROUP_NAME="gcd50664"

LOG_DIR="$PBS_O_WORKDIR/logs"
OUTPUT_DIR="$PBS_O_WORKDIR/output/$PBS_JOBID"
ENV_FILE="$PBS_O_WORKDIR/.env"
CONFIG_FILE="$PBS_O_WORKDIR/configs/base.json"

DEF_NAME="ms-swift_container.def"
SIF_NAME="ms-swift_container.sif"

# export HF_HOME="/groups/$GROUP_NAME/share/.cache/huggingface"
# export MODELSCOPE_CACHE="/groups/$GROUP_NAME/share/.cache/modelscope"
# export MNT_WORKSPACE="/groups/$GROUP_NAME/share/.cache/"

# trainingディレクトリ直下の.env内にwandbのトークンを記載する
# WANDB_API_KEY="your_wandb_api_key"