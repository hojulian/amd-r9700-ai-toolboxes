#!/usr/bin/env bash
# Only show in interactive shells
[[ $- != *i* ]] && return

oem_info() {
  local vendor product
  if [[ -r /sys/class/dmi/id/sys_vendor ]]; then
    vendor=$(cat /sys/class/dmi/id/sys_vendor 2>/dev/null || echo "Unknown")
    product=$(cat /sys/class/dmi/id/product_name 2>/dev/null || echo "")
    echo "${vendor} ${product}" | xargs
  else
    echo "Unknown"
  fi
}

gpu_name() {
  if command -v rocm-smi &>/dev/null; then
    rocm-smi --showproductname 2>/dev/null | grep -i 'card\|gpu\|gfx' | head -1 | sed 's/.*: *//' | xargs || echo "gfx1201"
  elif command -v rocminfo &>/dev/null; then
    rocminfo 2>/dev/null | grep -A2 'Marketing Name' | grep -v 'Marketing Name\|--' | head -1 | xargs || echo "gfx1201"
  else
    lspci 2>/dev/null | grep -i 'VGA\|Display\|3D' | head -1 | sed 's/.*: //' | xargs || echo "gfx1201"
  fi
}

rocm_version() {
  if [[ -f /opt/venv/lib/python3.12/site-packages/torch/version.py ]]; then
    python3 -c "import torch; print(torch.version.hip or 'N/A')" 2>/dev/null || echo "N/A"
  elif [[ -f /opt/rocm/.info/version ]]; then
    cat /opt/rocm/.info/version 2>/dev/null || echo "N/A"
  else
    echo "N/A"
  fi
}

cat <<'BANNER'

  ██████╗  █████╗ ██████╗ ████████╗ ██████╗ ██████╗      █████╗ ██╗
  ██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗    ██╔══██╗██║
  ██████╔╝███████║██████╔╝   ██║   ██║   ██║██████╔╝    ███████║██║
  ██╔══██╗██╔══██║██╔═══╝    ██║   ██║   ██║██╔══██╗    ██╔══██║██║
  ██║  ██║██║  ██║██║        ██║   ╚██████╔╝██║  ██║    ██║  ██║██║
  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝

       ██╗   ██╗██╗     ██╗     ███╗   ███╗
       ██║   ██║██║     ██║     ████╗ ████║
       ██║   ██║██║     ██║     ██╔████╔██║
       ╚██╗ ██╔╝██║     ██║     ██║╚██╔╝██║
        ╚████╔╝ ███████╗███████╗██║ ╚═╝ ██║
         ╚═══╝  ╚══════╝╚══════╝╚═╝     ╚═╝
  AMD R9700 (gfx1201) Toolbox
BANNER

printf '  GPU    : %s\n' "$(gpu_name)"
printf '  ROCm   : %s\n' "$(rocm_version)"
printf '  Machine: %s\n' "$(oem_info)"
printf '  Image  : docker.io/kyuz0/amd-r9700-vllm-toolboxes\n'
echo
echo '  Tools available:'
echo '    start-vllm          interactive model launcher (TUI)'
echo '    vllm serve          run the OpenAI-compatible API server'
echo '    python /opt/run_vllm_bench.py   run benchmarks'
echo
echo '  Remote access (port forwarding):'
echo '    ssh -L 8000:localhost:8000 user@host'
echo '    then: curl http://localhost:8000/v1/models'
echo

PS1='\u@\h:\w\$ '
