# MiDas - Multi-granularity Detector for Vulnerability Fixing Commits

MiDas is a transformer-based novel techinique for detecting vulnerability-fixing commits. MiDas extract information of commit in respect to multiple levels of granularity (i.e. commit level, file level, hunk level, line level)

MiDas consists of seven base models, regarding the combination of granularity and CodeBERT representation:


| Base model index | Granularity | CodeBERT representation |
|------------------|-------------|-------------------------|
| 1                | Commit      | Bimodal                 |
| 2                | File        | Bimodal                 |
| 3                | Hunk        | Bimodal                 |
| 5                | Commit      | Unimodal                |
| 6                | File        | Unimodal                |
| 7                | Hunk        | Unimodal                |
| 8                | Line        | Unimodal                |
