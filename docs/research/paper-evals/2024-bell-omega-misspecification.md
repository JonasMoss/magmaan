# The Impact of Measurement Model Misspecification on Coefficient Omega Estimates of Composite Reliability

**Cite.** Bell, Chalmers & Flora (2024). *Educational and Psychological Measurement*, 84(1):5-39. DOI: 10.1177/00131644231155804.
**PDF.** `external/refs/Bell et al. 2024 - The impact of measurement model misspecification on coefficient omega estimates of composite reliability.pdf`
**Read.** 2026-06-26  ·  **Verdict.** → speculative

## TL;DR
Monte Carlo study of finite-sample bias in model-based reliability coefficients omega-unidimensional (omega_u, from a one-factor CFA) and omega-hierarchical (omega_H, from a bifactor CFA) under structural-model misspecification. omega_u is strongly positively biased when error correlations are ignored or the population is multidimensional; omega_H stays nearly unbiased even when the bifactor model is wrong (one-factor-with-correlated-errors or higher-order populations).

## Contribution
- Population reliability is defined as the proportion of composite-score variance due to a factor common to all items, computed from the correct data-generating model. Sample omega is `(sum lambda)^2 / sigma_X^2`, with the general-factor loadings in the numerator (omega_H) or the single-factor loadings (omega_u), and sigma_X^2 either model-implied (`1' Sigma-hat 1`) or observed (`1' S 1`).
- omega_u and omega_H are unbiased under correct specification across composite length (8/16 items), population reliability (.60/.85), and N (100/250/1000).
- omega_u is badly positively biased (mean bias up to .25) when (a) true non-zero error covariances are left unmodeled, or (b) the population is bifactor / higher-order and a one-factor model is fit. Bias grows with longer scales and lower reliability.
- omega_H is robust to model-type misspecification: it stays within ~.02 even when the population is one-factor-with-correlated-errors or higher-order (not bifactor). So omega_H estimates "variance due to a general factor" well even when the bifactor structure is itself wrong.
- Denominator choice barely matters: observed `1' S 1` gives a tiny edge over model-implied under severe misspecification, RMSE differences negligible. Contradicts Bentler's (2009) efficiency claim for model-implied.
- Fit indices only weakly-to-moderately track omega bias (CFI/TLI correlations |.14| or weaker under correct spec; RMSEA flat once RMSEA < ~.10), so a passing fit does not certify an unbiased omega.
- Open future work the authors flag: omega for **categorical/ordinal items via polychoric CFA** scaled into the observed total-score metric (Green-Yang 2009b, Flora 2020), and the finite-sample behavior of omega-higher-order (omega_ho) under misspecification.

## Relevance to magmaan
Not core parity. semTools::reliability (over a lavaan CFA fit) is the tool the paper uses; matching it is a `measures::frontier` addition, not a lavaan-surface obligation.

The paper is the canonical recent reference and simulation design for **model-based (CFA-parameter) omega** (omega_u / omega_H / omega_ho), which is a distinct object from the **S-based** coefficients now in flight in `measures::frontier::reliability` (Cronbach alpha, Guttman lambda6, Spearman-Guttman omega computed directly from S, with delta-method SEs; grounded in `docs/research/notes/guttman_cfa_asymptotics.tex`). It is the same object as the exp-20 omega-alpha thread (`docs/backlog/todo.md`: omega as `(sum lambda)^2 / sigma_X^2` from a one-factor ML fit, alpha as the omega of a ULS tau-equivalent fit) and the roadmap `infer_gamma_nt` omega.

magmaan already produces every input: a fitted one-factor CFA gives `Lambda` and model-implied `Sigma-hat`, so `omega_u = (1' Lambda)^2 / (1' Sigma-hat 1)` (or `/ (1' S 1)`) is a one-liner over a `ModelEvaluator`; omega_H reads the general-factor loadings off a bifactor fit; omega_ho needs the `lambda_jk * gamma_k` products off a higher-order fit. Delta-method SEs would reuse the same `gradient`-times-Gamma machinery the in-flight S-based module already carries.

Load-bearing caveat for the author's "on our turf re. misspec" framing: the misspecification here is **structural-model-form** misspecification that biases a **point estimate** (fit a one-factor model to multidimensional/correlated-error data, omega goes up). That is a different beast from the distributional / moment-weight misspecification of experiment 35 and `papers/estimated-weight-se`, which leaves the model structure correct and corrects the **standard errors** (the omitted weight-meat term, [[misspec-robust-se-weight-influence]]). This paper does not feed the estimated-weight-se sandwich; it feeds the reliability module. Do not conflate.

The genuinely novel cell for magmaan is the ordinal one. Bell generates only continuous multivariate-normal data and explicitly leaves polychoric-CFA omega under misspecification (and omega_ho finite-sample properties) unexamined. magmaan's ordinal/DWLS/polychoric estimators plus the `sim` sublibrary make an ordinal extension of Bell's bias-under-misspecification design an author-flagged gap, not manufactured work.

## Verdict
→ speculative. Model-based omega (omega_u / omega_H / omega_ho) as a `measures::frontier` reliability surface, with Bell as the bias-under-misspecification oracle and the ordinal extension as the novel paper cell. Landed in [speculative.md](../../backlog/speculative.md#model-based-omega-omega_u--omega_h--omega_ho-reliability) under Measures / reporting; cross-links the in-flight S-based module and the exp-20 omega-alpha todo.
