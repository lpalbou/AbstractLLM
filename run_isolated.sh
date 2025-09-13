#!/bin/bash
# Script to run Python with isolated library paths, avoiding Anaconda conflicts

# Unset any Anaconda-related environment variables
unset DYLD_LIBRARY_PATH
unset DYLD_FALLBACK_LIBRARY_PATH

# Remove Anaconda from PATH temporarily
export PATH=$(echo $PATH | tr ':' '\n' | grep -v anaconda | tr '\n' ':')

# Use only the virtual environment's libraries
export DYLD_LIBRARY_PATH=""
export DYLD_FALLBACK_LIBRARY_PATH="$HOME/lib:/usr/local/lib:/usr/lib"

# Run the command with isolated environment
exec "$@"