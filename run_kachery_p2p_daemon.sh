#!/bin/bash

logdir="_log"
mkdir -p $logdir
logfile="${logdir}/kachery_p2p_daemon.log"

nohup kachery-p2p-start-daemon --label franklab --static-config franklab_kachery-p2p_config.yaml > "$logfile" 2>&1 &

