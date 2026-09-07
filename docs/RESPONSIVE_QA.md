# Narrow Review Jobs regression

Reproduced on macOS, 2026-09-07, candidate `65ac9fd`, at 390 × 844 CSS pixels.

1. Open Demo → Review Jobs → Northstar Metrics Studio → Evidence.
2. Inspect the original-posting action and requirement filter toolbar.

Before the fix, “Open job” wraps into individual letter fragments and the evidence
heading overlaps the compressed filter labels. The analysis action is clipped.
Expected: the posting action and filter toolbar use separate full-width rows when
the viewport is narrow, with readable labels and accessible actions. Desktop keeps
its existing columns. Repeat at 390 × 844 and the default 1280 × 720 viewport.
