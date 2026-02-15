#!/bin/bash
set -euo pipefail

DIR="/mnt/ASC1664/unifolm-wma-0-dual/converted-data/videos/Z1_StackBox_Dataset/observation.images.cam_high"
LOG="/tmp/reencode_av1.log"
echo "Start re-encode AV1 -> H264 at $(date)" > "$LOG"

for f in "$DIR"/*.mp4; do
  # skip if not a file
  [ -f "$f" ] || continue

  codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name \
         -of default=noprint_wrappers=1:nokey=1 "$f") || codec="?"

  if [ "$codec" != "av1" ]; then
    echo "[SKIP] $f (codec=$codec)" | tee -a "$LOG"
    continue
  fi

  out="${f%.mp4}.re.mp4"
  echo "[RE-ENC] $f -> $out (codec=$codec)" | tee -a "$LOG"

  # re-encode; tune preset/crf if you want
  ffmpeg -v error -y -i "$f" -c:v libx264 -preset veryfast -crf 20 -c:a aac -b:a 64k "$out" 2>>"$LOG"
  if [ $? -ne 0 ]; then
    echo "[ERR] ffmpeg failed for $f" | tee -a "$LOG"
    rm -f "$out"
    continue
  fi

  # replace original (atomic-ish)
  mv "$out" "$f"
  echo "[OK] replaced $f" | tee -a "$LOG"
done

echo "Done at $(date)" >> "$LOG"
