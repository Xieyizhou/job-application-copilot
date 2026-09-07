# JobCopilot browser companion

This unpacked Chrome/Edge extension imports the job posting already open in your browser into the local Personal workspace. It reads `JobPosting` JSON-LD first and falls back to the visible job-description container. The captured text is sent only to `127.0.0.1:8765`.

For v1.0.0, native acceptance covers Chrome on macOS. Edge compatibility is experimental
and has not been verified. A Safari extension is not included.

## Install locally

1. Start JobCopilot with `python run_dashboard.py`.
2. Open `chrome://extensions` and enable **Developer mode**. For experimental Edge use,
   the equivalent page is `edge://extensions`.
3. Choose **Load unpacked** and select this `browser_companion/` directory.
4. In JobCopilot, open **Settings → Job sources** and copy the local connection token.
5. Open the extension once, paste the token, and save it.

## Import a posting

Save the job in JobCopilot first, open its original posting in the same browser, and choose **Import and verify this posting** in the extension. JobCopilot accepts the capture only when it matches the saved URL or the saved company and role, and only replaces the saved text after completeness checks pass. Choose **Open JobCopilot** to return directly to the local review screen and inspect the refreshed score.

The extension is deliberately a capture companion, not a remote scoring service. It does not upload resume data and does not communicate with a remote JobCopilot service; matching and scoring remain in the local app.
