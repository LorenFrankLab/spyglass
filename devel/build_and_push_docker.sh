#!/bin/bash

set -ex

docker build -f .devcontainer/Dockerfile -t magland/nwb-datajoint-dev .
docker push magland/nwb-datajoint-dev
