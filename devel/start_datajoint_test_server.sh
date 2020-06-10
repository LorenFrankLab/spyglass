#!/bin/bash

set -ex

docker run \
    -p 3306:3306 \
    -e MYSQL_ROOT_PASSWORD=tutorial \
    datajoint/mysql