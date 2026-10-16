#!/bin/sh
# v1 origin: tasks/permit-forge/instances/003bd328a392-s47/environment/build-models.sh
set -eu
build_root=${1:-/build}
model_output=${2:-/opt/models}
mkdir -p "$model_output"
cd "$build_root/aermod"
for source in modules grsm aermod setup coset soset reset meset ouset inpsum metext iblval siggrid tempgrid windgrid calc1 calc2 prise arise prime sigmas pitarea uninam output evset evcalc evoutput rline bline; do
    gfortran -c -O2 -static -fbounds-check -Wuninitialized "$source.f"
done
gfortran -static -O2 -o "$model_output/aermod" ./*.o
cd "$build_root/bpipprm"
# The released DOS source ends in SUB (0x1a), not a Fortran statement.
# Preserve the bound source; normalize only the compiler's scratch input.
tr -d '\032' < Bpipprm.for > bpipprm.f
gfortran -O2 -static -std=legacy -o "$model_output/bpipprm" bpipprm.f
