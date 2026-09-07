# Upgrade and rollback

## Upgrade to v1.0.0

1. Stop Job Copilot and make a private backup of `data/local_workspace/`.
2. Check out the `v1.0.0` tag in a clean source directory.
3. Create a new Python 3.11 or 3.12 virtual environment and install `requirements.txt`.
4. Copy the private workspace into the new checkout only after the clean Demo smoke test.
5. Start with `python run_dashboard.py` and verify the candidate, saved-job, generated-file,
   and tracker views before making changes.

The release preserves existing workspace data, extension messages, model artifacts, and
document formats. Never commit the workspace backup.

## Roll back

1. Stop Job Copilot.
2. Restore the private workspace backup if the newer version wrote incompatible local data.
3. Return to the previously used immutable tag or commit, create a fresh virtual
   environment, and reinstall its requirements.
4. Start with that version's documented command and verify the workspace before deleting
   the backup.

Published tags are immutable. A post-release blocker is handled by a patch release or a
revert commit, not by moving `v1.0.0`.
