# #!/bin/bash

set -ex

# install nwb_datajoint in develop mode
cd /workspaces/nwb_datajoint
pip install -e .

mkdir -p $KACHERY_STORAGE_DIR

# Put some convenient git aliases into .bashrc
cat <<EOT >> ~/.bashrc
alias gs="git status"
alias gpl="git pull"
alias gps="git push"
alias gpst="git push && git push --tags"
alias gc="git commit"
alias ga="git add -u"
EOT
