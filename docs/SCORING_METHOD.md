# Scoring Method

Job Application Copilot uses deterministic requirement matching to support human
review. It does not use a trained model and does not predict hiring decisions.
Deterministic rules keep the same text reproducible, make each match inspectable,
and avoid presenting historical hiring patterns as judgments about a candidate.

## Role Fit

The parser identifies recognized requirements that are explicitly present in the
job description. Required terms have full weight. Preferred, bonus, and
nice-to-have terms have lower weight, so a preferred gap has less influence than
a required gap.

A direct match is explicit candidate evidence for the requested term. A partial
match is documented adjacent evidence, such as UAV route-planning work for a
robotics requirement. Partial evidence receives less credit and is labeled; it is
never reported as a direct claim.

The scorer first calculates **Observed Requirement Coverage** using active-category
normalization. Only scoring categories mentioned by the job description enter the
coverage denominator. Categories absent from the description are marked not
applicable rather than treated as candidate gaps. Required and preferred weights
affect terms within each active category; active categories retain their configured
category weight.

Observed coverage is not automatically the final Role Fit. When fewer than four
reliable requirement signals are available, high coverage is calibrated toward a
neutral 50-point prior. The evidence factor is `reliable signals / 4`; a 1-of-1
match therefore has 100% observed coverage but a 62/100 provisional Role Fit.
Calibration never increases a weak observed score. Saved API descriptions that
are short or visibly truncated are limited to at most three reliable signals even
if several keywords happen to appear in the snippet.

For saved jobs, the explicit role title contributes one separate **role-focus
alignment** signal. Specialized titles such as physics, audio engineering, data
migration, or data entry require candidate evidence in that core domain; a generic
machine-learning overlap cannot by itself produce the same fit as a matching ML
or data role. Role-focus alignment may reduce a superficial coverage score and may
add one reliable signal, but it never raises the score above observed requirement
coverage or fills an explicitly missing requirement.

## Eligibility, Confidence, and Recommendation

Role Fit is separate from Eligibility. Explicit minimum experience, seniority,
graduate-degree, and work-authorization constraints are evaluated as gates. An
unmet hard constraint can fail Eligibility without rewriting the Role Fit score;
unknown or equivalent-experience cases require manual review.
Minimum-years, degree, and work-authorization failures therefore do not reduce
Role Fit a second time; the Recommendation gate prevents an unsafe Apply result.

Confidence describes evidence coverage, not candidate quality. It reflects job
text completeness, the number of recognized requirements, and whether usable
candidate evidence exists. A very short description or a narrow 1-of-1 match is
low confidence even when its observed coverage is 100%. The dashboard labels the
calibrated number as a **Provisional Role Fit** and shows observed coverage
separately, so users can inspect the extracted terms without mistaking a narrow
match for a reliable recommendation. Low confidence still
forces **Manual Review** until the full job description provides enough evidence.

The final Recommendation applies gates in this order:

1. Failed Eligibility produces **Skip / Not Eligible**.
2. Manual-review Eligibility or low Confidence produces **Manual Review**.
3. Otherwise, calibrated score bands produce **Apply**, **Apply / Maybe Apply**,
   **Maybe Apply**, or **Skip or Low Priority**.

## Public Benchmark

The benchmark uses fictional roles and candidate summaries so it is reproducible,
safe to publish, and independent of personal application records. Cases cover six
role families plus direct, partial, required, preferred, incomplete-evidence, hard
constraint, confidence, and recommendation boundaries.

The evaluator reports:

- score-range agreement;
- Eligibility, Confidence, and Recommendation agreement;
- hard-constraint false negatives;
- unsafe high-score false positives;
- results by role family and concise failed-case diagnostics.

Run it with:

```bash
python scripts/evaluate_scoring.py
python scripts/evaluate_scoring.py --markdown /tmp/scoring-benchmark.md
```

Passing means the deterministic implementation agrees with the reviewed ranges
and labels in this fixture. The ranges intentionally allow reasonable variation
instead of encoding one exact score per case. They are human-reviewed expectations
for fictional evidence, not targets that justify duplicating an Eligibility penalty
inside the production Role Fit score.

## Interpretation and Limits

The vocabulary and adjacent-match map are intentionally small. Unusual wording,
unrecognized technologies, negation, ambiguous seniority, and incomplete source
text can still require human review. Keyword presence does not prove proficiency,
recency, depth, or truthful ownership of an accomplishment.

A Role Fit score is not a probability of interview or offer. The benchmark is
not a statistical validation against real recruiting outcomes, does not measure
employer behavior, and should not be used as an automated hiring decision. It is
a regression and calibration suite for this project's explainable rules.
