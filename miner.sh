#!/bin/bash
miners=$(jq -c '.[]' .miner)

for m in $miners; do
  NAME=$(echo $m | jq -r '.NAME')
  SCRIPT=$(echo $m | jq -r '.SCRIPT')
  WALLET=$(echo $m | jq -r '.WALLET')
  HOTKEY=$(echo $m | jq -r '.HOTKEY')
  COPY_UID=$(echo $m | jq -r '.COPY_UID')
  TRANSFORM=$(echo $m | jq -r '.TRANSFORM')
  CONTINUOUS=$(echo $m | jq -r '.CONTINUOUS')

  CMD="python $SCRIPT --wallet-name $WALLET --hotkey-name $HOTKEY --copy-uid $COPY_UID"

  # Add transform only if non-empty
  if [[ -n "$TRANSFORM" && "$TRANSFORM" != "null" ]]; then
    CMD="$CMD --transform $TRANSFORM"
  fi

  # Add continuous flag
  [[ "$CONTINUOUS" == "true" ]] && CMD="$CMD --continuous"

  pm2 start --name "$NAME" "$CMD"
done