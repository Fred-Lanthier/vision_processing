#!/bin/bash
sleep 5  # attendre que RViz et tout soit prêt
DURATION=${1:-60}  # durée en secondes, 60 par défaut
OUTPUT="/tmp/cbf_test_$(date +%Y%m%d_%H%M%S).mp4"
echo "Enregistrement démarré: $OUTPUT (durée: ${DURATION}s)"
ffmpeg -y -video_size 1920x1080 -framerate 30 -f x11grab -i $DISPLAY \
       -c:v libx264 -preset ultrafast -pix_fmt yuv420p \
       -t $DURATION "$OUTPUT"
echo "Enregistrement terminé: $OUTPUT"