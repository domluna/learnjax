# v5litepod-1	ct5lp-hightpu-1t
# v5litepod-4	ct5lp-hightpu-4t
# v5litepod-8	ct5lp-hightpu-8t
# v5litepod-16	ct5lp-hightpu-4t	4x4
# v5litepod-32	ct5lp-hightpu-4t	4x8
# v5litepod-64	ct5lp-hightpu-4t	8x8
# v5litepod-128	ct5lp-hightpu-4t	8x16
# v5litepod-256	ct5lp-hightpu-4t	16x16 (256 chips)
export TPU_NAME="brrr1"
export ZONE="us-west4-a"

# v5litepod-1 is 1x1 (1 chip)
gcloud compute tpus tpu-vm create ${TPU_NAME} \
--zone=${ZONE} \
--accelerator-type=v5litepod-1 \
--version=v2-alpha-tpuv5-lite \
--preemptible \
--spot 

# gets and error because you need to contact Google for v5 use

# --version=v2-alpha-tpuv5-lite \
# gcloud alpha compute tpus queued-resources create ${QUEUED_RESOURCE_ID} \
#    --node-id=${TPU_NAME} \
#    --project=${PROJECT_ID} \
#    --zone=${ZONE} \
#    --accelerator-type=${ACCELERATOR_TYPE} \
#    --runtime-version=${RUNTIME_VERSION} \
#    --valid-until-duration=${VALID_UNTIL_DURATION} \
#    --service-account=${SERVICE_ACCOUNT} \
#    --${QUOTA_TYPE}
#

pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html


gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --worker="0"
# gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --worker=all
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --worker="all" --command='pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html'

gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --worker="all" --command='python3 -c "import jax; print(jax.device_count()); print(jax.local_device_count())"'


gcloud compute tpus tpu-vm ssh ${TPU_NAME} -L 8080:localhost:8080

gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE}
