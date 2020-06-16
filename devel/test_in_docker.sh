#!/bin/bash

set -ex

allargs="$@"
exec docker run \
  -v $PWD:/workspaces/nwb_datajoint \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /run/docker.sock:/run/docker.sock \
  -e DOCKER_HOST=unix:///run/docker.sock \
  --group-add docker \
  -u $(id -u):$(id -g) \
  -v /tmp:/tmp \
  --net=host \
  -w /workspaces/nwb_datajoint \
  -it magland/nwb-datajoint-dev \
  /bin/bash -c "PYTHONPATH=/workspaces/nwb_datajoint devel/test.sh $allargs"
