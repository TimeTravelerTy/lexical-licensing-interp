# TSUBAME Workflow

Use `tsubame` as the SSH host alias and
`/gs/fs/tga-sip_arase/tyrone/lexical_licensing_interp` as the default remote
project directory.

## Sync Policy

For official runs, git is the canonical sync path:

1. Check local changes.
2. Commit the run-relevant code, configs, and small metadata artifacts.
3. Push to GitHub.
4. Pull on TSUBAME before creating or submitting jobs.

For this repo, sync commits are pre-approved when they are needed to run or
verify the current TSUBAME task. The commit should be narrow, include only
run-relevant files, and use a descriptive message. No separate confirmation is
needed for that sync commit.

Do not include large generated model outputs, logs, caches, or scratch artifacts
in sync commits. Use `scp` or `rsync` for those when needed.

## Rsync And Scratch Files

`rsync` is useful for large generated inputs, one-off scratch files, or outputs
that should not live in git. Prefer a dry run first:

```bash
rsync -av --dry-run path/ tsubame:/gs/fs/tga-sip_arase/tyrone/lexical_licensing_interp/path/
```

Then run the same command without `--dry-run` if the file list is correct.

For official job provenance, write the local git commit hash and any rsynced
file checksums or counts into the job log or manifest.

## Job Rule

Do not submit TSUBAME jobs against stale remote code. Before `qsub`, run:

```bash
ssh tsubame "cd /gs/fs/tga-sip_arase/tyrone/lexical_licensing_interp && git pull"
```

Submit jobs with:

```bash
qsub -g tga-sip_arase jobs/<script>.sh
```

