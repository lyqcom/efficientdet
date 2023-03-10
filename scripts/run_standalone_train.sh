#!/bin/bash

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# != 1 ]
then
    echo "Usage: bash run_standalone_train.sh [DEVICE_ID]"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=$1
export RANK_SIZE=1
export RANK_ID=0

rm -rf ./train_standalone
mkdir ./train_standalone
cp ../*.py ./train_standalone
cp *.sh ./train_standalone
cp -r ../src ./train_standalone
cd ./train_standalone || exit
echo "start training for rank $RANK_ID, device $DEVICE_ID"
env > env.log
python train.py > log.txt 2>&1 &

cd ..
