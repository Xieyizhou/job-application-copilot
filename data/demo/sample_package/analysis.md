# Demo Job Match Analysis

- Job description file: `data/demo/jobs/data_analyst.md`
- Candidate source: `data/resume/resume_source.example.md`
- Snapshot: Fictional Demo output generated with the canonical analyzer

## Summary

- Role Fit Score: **92/100**
- Eligibility: **Passed**
- Scoring Confidence: **High**
- Recommendation: **Apply**
- Why: Apply because the role-fit score is 92/100 after normalizing only the categories the job description actually mentions.
- Eligibility reasons: None
- Confidence reasons: At least eight scored requirements and candidate evidence were extracted.

## Parsed Job Requirements

- Required skills: Python, SQL, data visualization, data analysis, statistical analysis, documentation, communication
- Preferred / plus skills: None
- Experience level: intern, recent graduate
- Degree requirements: None
- Domain keywords: statistical analysis

## Score Breakdown

- Core technical skills: **40.0/40**
  - JD terms scored: Python, SQL, data visualization, data analysis
  - Matched: Python, SQL, data visualization, data analysis
  - Partial / adjacent: None
  - Missing required or preferred terms: None
  - Note: Strong fit: all requested core technical skills terms are supported by the resume.
- Domain fit: **25.0/25**
  - JD terms scored: statistical analysis
  - Matched: statistical analysis
  - Partial / adjacent: None
  - Missing required or preferred terms: None
  - Note: Strong fit: all requested domain fit terms are supported by the resume.
- Experience level fit: **7.5/15**
  - JD terms scored: intern, recent graduate
  - Matched: intern
  - Partial / adjacent: None
  - Missing required or preferred terms: recent graduate
  - Note: Good fit: the resume supports several requested experience level fit terms, with any adjacent matches labeled as partial.
- Project relevance: **N/A**
  - N/A: job description does not ask for these terms.
- Communication / collaboration fit: **10.0/10**
  - JD terms scored: communication, documentation
  - Matched: communication, documentation
  - Partial / adjacent: None
  - Missing required or preferred terms: None
  - Note: Strong fit: all requested communication / collaboration fit terms are supported by the resume.

## Penalties

- None found

## Matched Skills

- Python
- SQL
- data visualization
- data analysis
- statistical analysis
- intern
- communication
- documentation

## Partial / Adjacent Matches

- None found

## Missing Skills

- recent graduate

## Relevant Experience Themes

- Python data analysis
- Econometrics and statistical reasoning
- Teaching and communication

## Red Flags

- None found

## Relevant Resume Evidence

- Candidate source contains keywords related to Python data analysis.
- Candidate source contains keywords related to Econometrics and statistical reasoning.
- Candidate source contains keywords related to Teaching and communication.

## Demo Output

- Demo package generated from fictional, read-only source data.
- Demo only; review all files and do not submit them.

## Human Review Notes

- This report uses weighted keyword matching and simple penalty rules.
- Required and preferred JD terms use symmetric requirement weights; preferred gaps have less impact.
- Categories the JD does not mention are marked N/A and are not counted against the final score.
- Eligibility and scoring confidence are separate from the role-fit score and may override the recommendation.
- It should be reviewed by a person before preparing application materials.
- It does not invent experience, skills, degree level, metrics, visa status, or work authorization.
- Confirm the resume source's degree level before relying on education-related statements.
- It does not submit applications or interact with job platforms.
