#!/bin/bash

set -ex

docker run \
    --name datajoint-test-server \
    -p 3306:3306 \
    -e MYSQL_ROOT_PASSWORD=tutorial \
    datajoint/mysql