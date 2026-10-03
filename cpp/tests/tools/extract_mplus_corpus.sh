#!/usr/bin/env bash
set -euo pipefail
# Extraction is local evidence only; originals and zip bytes stay untracked.
python3 - "$@" <<'PY'
import pathlib, sys, zipfile, hashlib, tempfile, time
root=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else 'external/textbook-corpus')
out=pathlib.Path(sys.argv[2] if len(sys.argv)>2 else '~/.cache/magmaan-mplus-corpus').expanduser()
out.mkdir(parents=True,exist_ok=True)
files=pathlib.Path(tempfile.mkdtemp(prefix='files_staging_',dir=out))
entries=[]
for p in sorted(root.glob('cases/*/*/source/original.inp')): entries.append((str(p),p.read_bytes()))
for p in sorted((root/'raw').rglob('*.inp')): entries.append((str(p),p.read_bytes()))
for p in sorted((root/'raw').rglob('*.zip')):
    if not zipfile.is_zipfile(p):
        print(f"Skipping non-zip mirror: {p}",file=sys.stderr); continue
    with zipfile.ZipFile(p) as z:
        for name in sorted(z.namelist()):
            if name.lower().endswith('.inp'): entries.append((str(p)+'!'+name,z.read(name)))
manifest=['file\tsource\tsha256']
for i,(name,data) in enumerate(entries):
    target=f'{i:05d}.inp'; (files/target).write_bytes(data)
    manifest.append(target+'\t'+name+'\t'+hashlib.sha256(data).hexdigest())
destination=out/'files'
if destination.exists(): destination.rename(out/f'files_previous_{time.time_ns()}')
files.rename(destination)
files=destination
(out/'manifest.tsv').write_text('\n'.join(manifest)+'\n')
print(f'Extracted {len(entries)} inputs to {files}')
PY
