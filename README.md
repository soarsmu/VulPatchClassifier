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


To replicate the training process of MiDas, please follow the below steps:

        1. Finetune CodeBERT for each base model
        2. Save commit embedding vectors represented by CodeBERT
        3. Train base models
        4. Infer base models to extract commit's features
        5. Train ensemble model
        6. Apply adjustment function 
        7. Evaluate MiDas 
