#!/bin/bash

set -ex

docker kill datajoint-test-server || true
docker rm datajoint-test-server