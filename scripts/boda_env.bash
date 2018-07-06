# this base script is meant to be sourced to add boda to the PATH and
# enable bash completion support. sourcing it is optional. it should
# not need to be modified, but if the 'BODA_HOME' magic below doesn't
# work for you, you can just hard-code the boda directory instead, or
# just put boda in your PATH however else you please.
if [ -n "$ZSH_VERSION" ]; then
   # make zsh compatible with bash completion scripts
   autoload bashcompinit
   bashcompinit
   SOURCE=${(%):-%N}
elif [ -n "$BASH_VERSION" ]; then
   SOURCE=${BASH_SOURCE}
else
   echo "boda_env.bash: This script is only compatible with bash and zsh."
   exit 1
fi

BODA_HOME="$( cd "$(dirname "${SOURCE}")" ; cd .. ; pwd -P )"
export PATH=${BODA_HOME}/lib:$PATH
. ${BODA_HOME}/scripts/boda_completion.bash

