#!/bin/bash
# Resumable download of the real sequence databases. Runs on a COMPUTE node
# (login nodes are not to be used for the big files) under QOS embers, which is
# PREEMPTIBLE -- so every transfer uses `curl -C -` and every file is size- and
# md5-checked against the publisher's own metalink before being declared good.
# Re-running this script after a preemption resumes; a file that already
# verifies is skipped.
set -u
D=/storage/scratch1/7/avandevoorde3/p1/data
cd "$D" || exit 1
echo "node $(hostname)  start $(date -Is)"
df -h "$D" | tail -1

# name | url | expected_md5 | expected_size
#
# The md5s and sizes are the PUBLISHER'S OWN, copied out of
#   .../knowledgebase/complete/RELEASE.metalink   (UniProt release 2026_02)
#   .../uniref/uniref{50,90}/RELEASE.metalink     (same release)
#   ftp.ebi.ac.uk/pub/databases/Pfam/current_release/md5_checksums
# Those four small files are fetched alongside the data so the check can be
# re-derived rather than trusted to this comment. Pfam publishes an md5 but no
# size, hence the 0 -- the md5 is the stronger check anyway.
FILES=(
"uniprot_sprot.fasta.gz|https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz|797dad11a33b1b58e3c140649a74d6b6|93706469"
"Pfam-A.hmm.gz|https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz|7ab3c4e215d0daaea3004e37c4e24f8a|0"
"uniref50.fasta.gz|https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz|3228886e9d749f050f60e9a0ce1f727d|8770260598"
)
# UniRef90 is 32 GB compressed / ~100 GB raw and is only worth pulling once the
# UniRef50 numbers look consistent. Opt in with WANT_UNIREF90=1.
if [ "${WANT_UNIREF90:-0}" = "1" ]; then
  FILES+=("uniref90.fasta.gz|https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz|abdd341aeafa7fa060c8d6639d594990|32059052376")
fi

for m in \
  "RELEASE.knowledgebase.metalink|https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/RELEASE.metalink" \
  "RELEASE.uniref50.metalink|https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/RELEASE.metalink" \
  "RELEASE.uniref90.metalink|https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/RELEASE.metalink" \
  "pfam_md5_checksums|https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/md5_checksums" ; do
  IFS='|' read -r mf murl <<< "$m"
  [ -s "$mf" ] || curl -sS --retry 3 -o "$mf" "$murl"
done

status=0
for spec in "${FILES[@]}"; do
  IFS='|' read -r f url want_md5 want_size <<< "$spec"
  echo "=============== $f ==============="

  # Already verified in a previous (possibly preempted) invocation?
  if [ -f "$f.md5ok" ]; then echo "  already verified, skipping"; continue; fi

  for attempt in 1 2 3 4 5 6; do
    have=0; [ -f "$f" ] && have=$(stat -c %s "$f")
    if [ "$want_size" != "0" ] && [ "$have" = "$want_size" ]; then
      echo "  size already $have == expected, no transfer needed"
      break
    fi
    echo "  attempt $attempt (have $have bytes) $(date -Is)"
    curl -L -C - --fail --no-progress-meter --retry 5 --retry-delay 15 --connect-timeout 30 \
         --speed-time 120 --speed-limit 10240 -o "$f" "$url"
    rc=$?
    echo "  curl rc=$rc  now $(stat -c %s "$f" 2>/dev/null) bytes"
    # rc 33 = server has no range support for a complete file; 416 -> rc 22.
    if [ $rc -eq 0 ] || [ $rc -eq 33 ] || [ $rc -eq 22 ]; then break; fi
    sleep 20
  done

  got_size=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$want_size" != "0" ] && [ "$got_size" != "$want_size" ]; then
    echo "  SIZE MISMATCH: got $got_size want $want_size"; status=1; continue
  fi
  echo "  md5sum ($got_size bytes) ..."
  got_md5=$(md5sum "$f" | cut -d' ' -f1)
  if [ "$got_md5" = "$want_md5" ]; then
    echo "  MD5 OK   $got_md5   (publisher metalink / md5_checksums)"
    echo "$got_md5  $f" > "$f.md5ok"
  else
    echo "  MD5 MISMATCH: got $got_md5 want $want_md5"; status=1
  fi
done

# Small files decompressed here so the measurement job need not; uniref50 stays
# gzipped on scratch and is expanded onto node-local NVMe inside the bench job.
for f in uniprot_sprot.fasta Pfam-A.hmm; do
  if [ ! -f "$f" ] && [ -f "$f.gz.md5ok" ]; then
    echo "decompressing $f.gz"; gunzip -kf "$f.gz" && ls -la "$f"
  fi
done

echo "=== final ==="; ls -la "$D"
echo "exit status $status  end $(date -Is)"
exit $status
