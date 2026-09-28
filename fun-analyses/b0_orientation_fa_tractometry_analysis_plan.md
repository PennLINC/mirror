# Sanity-Check Analysis Plan: Fiber Orientation Relative to B0 and DTI Metrics in NIBS

## 1. Objective

Perform a **small, observational sanity check** for an association between white-matter fiber
orientation relative to the scanner main field (`B0`) and diffusion-tensor metrics (FA, AD, RD,
MD) in the NIBS pilot dataset.

Prior controlled work that re-scanned the same participants at multiple head positions reported
that tensor metrics vary with fiber-to-`B0` angle: fibers closer to perpendicular to `B0` showed
higher FA, higher AD, and lower RD. The proposed mechanism is orientation-dependent compartmental
T2 (intra- vs extra-axonal), which changes the relative weighting of compartments at a given TE
and b-value. The expected direction and magnitude in Sections 2 and 6 are taken from the
reference studies in Section 1.1.

### 1.1 Reference studies

Both studies use the same data: 5 healthy adults, 3 T Connectom, a receive coil tilted by 0 and
18 degrees about the left-right axis, 3 mm isotropic voxels, TE = 54, 75, 100, 125, and 150 ms.

- **Tax et al. (2021)**, *NeuroImage* 236, 117967, doi:10.1016/j.neuroimage.2021.117967.
  Compartmental T2 only. Extra-axonal `R2 = 17.4 + 2.4 sin^4(theta)` s^-1; intra-axonal R2 varies
  much less with orientation. This is the mechanism.
- **Kleban, Jones & Tax (2023)**, *Imaging Neuroscience* 1, doi:10.1162/imag_a_00012. Tensor
  metrics from `b = 0, 750, 1500 s/mm^2`, corrected for gradient nonlinearity. Single-fiber voxels
  only (second/first fODF peak below 10%, low dispersion). Two analyses:
  1. **Pooled voxels** across all white matter and both head orientations. FA, AD, and RD vary by
     up to about 20% of their mean across angles (FA range about 0.13 at TE 75-100 ms). This
     analysis mixes anatomy with orientation, like the pooled plots in Section 11. It is **not**
     the reference for the primary analysis.
  2. **Tract-segment differences** between the two head orientations, regressed on the change in
     `sin^4(theta)`. This has the same logic as Section 9 and is the reference for it.

Reference effect sizes (analysis 2; slope per unit `sin^4(theta)`), read from the bars and error
bars of the published Fig. 4B, so accurate to about 0.005:

| Metric | TE 75 ms | TE 100 ms | Interpolated to TE 88 ms (NIBS) |
|--------|----------|-----------|---------------------------------|
| FA | 0.058 (0.034 to 0.082) | 0.064 (0.035 to 0.093) | **0.061 (0.034 to 0.088)** |
| AD (um^2/ms) | 0.112 (0.063 to 0.161) | 0.124 (0.070 to 0.178) | 0.118 (0.067 to 0.170) |
| RD (um^2/ms) | -0.035 (-0.059 to -0.011) | -0.040 (-0.064 to -0.016) | -0.038 (-0.062 to -0.014) |
| MD (um^2/ms) | 0.015 (0.001 to 0.029) | 0.015 (-0.004 to 0.034) | 0.015 (-0.002 to 0.032) |

Caveats when comparing with NIBS:

- The paper does not state what its error bars are, and its segments are treated as independent
  although they come from 5 participants. Its uncertainty is probably understated.
- Its voxels are much more coherent than NIBS bundle nodes (Section 8.2). Its own simulations show
  that fiber dispersion reduces the FA effect. The reference slope is therefore an **upper bound**
  on the slope expected in NIBS.
- The FA slope increases with TE (0.039 at 54 ms to 0.126 at 150 ms), so the interpolation to
  88 ms matters.
- NIBS uses `b <= 1200` for the primary tensor fit; the reference reports slightly larger effects
  at lower maximum b-value.

NIBS is **not** a single-head-position dataset. Each participant was scanned in **two sessions**,
each with its own natural head positioning. That gives a within-subject, within-tract contrast
that has the same logic as a multi-head-position experiment, just with small angle changes. This
contrast is the **primary analysis**. The cross-sectional (between-subject) contrast is secondary
because anatomy dominates it.

This is still not a controlled manipulation: head pose is not assigned, and angle changes will be
small. The result should be framed as a feasibility check and a basis for power calculations for
a prospective tilted-head study.

---

## 2. Hypotheses

### 2.1 Primary hypothesis

Within participant, tract, and node, a between-session increase in the angular sensitivity term
`f(theta)` (see Section 7) is associated with a between-session increase in FA. In other words,
fibers moved closer to perpendicular to `B0` show higher FA.

### 2.2 Secondary directional hypotheses

Using the same model:

- more perpendicular -> higher AD;
- more perpendicular -> lower RD;
- MD: no pre-specified direction (the reference reports a small positive slope at short TE and
  about zero at long TE).

Concordance across FA, AD, and RD counts for more than a nominal FA-only association. It requires
each interval to exclude zero, not only the expected signs.

### 2.3 Positive control

R2* (from MEGRE) has a large, well-established dependence on fiber orientation relative to `B0`,
approximately linear in `sin^2(theta)` and `sin^4(theta)`. Using the same pipeline, R2* should
increase with `f(theta)`. **If the pipeline cannot recover the R2* effect, a null FA result is
uninterpretable** (Section 13). The control validates the geometry (B0 vector, sign conventions,
node pairing). It does not show that the design is sensitive enough for the much smaller FA
effect; Section 6 does that.

---

## 3. Analysis Unit and Data

The observation is **participant x session x tract x node**. For each, retain:

- FA, AD, RD, MD (primary tensor fit, Section 5.4);
- local fiber direction `v` in ACPC space (Section 5.3);
- session-specific `B0` unit vector in ACPC space (Section 4);
- `theta`, `f(theta)`, and `theta_ref` (Section 5.5);
- tract identity, node index, participant, session;
- fiber-coherence metrics: number of MSMT-CSD fixels above threshold, peak ratio;
- node support (streamline count, voxel count);
- node-level group-average FA (for outcome-independent filtering, Section 8);
- QC flags (existing manual QC ratings, eddy motion summaries).

Existing data used:

- QSIPrep outputs (ACPC space, `--subject-anatomical-reference unbiased`, so both sessions share
  one ACPC anatomical reference);
- QSIRecon TORTOISE tensors (inner-shell `bval_cutoff: 1200` and full-shell), DIPY DKI, MSMT-CSD
  FODs, DSI Studio AutoTrack bundles;
- MEGRE R2* for the positive control.

---

## 4. Head Pose and the B0 Vector (Step 0; see `extract_b0_pose.py`)

### 4.1 Definition

For head-first supine (HFS) acquisitions on the Siemens Prisma, `B0` is parallel to the world
z-axis of the raw dcm2niix NIfTI (scanner/patient coordinates). **Verify that `PatientPosition ==
HFS` for every run**; stop and review any run where it is not.

All tractometry happens in ACPC space. ACPC image z is **not** `B0`. For each participant and
session, bring the scanner z-axis into ACPC space using the rotation that relates the raw
acquisition to the ACPC reference:

```text
B0_acpc[s, ses] = R[s, ses]^T @ [0, 0, 1]
```

where `R[s, ses]` is the rotation part of the rigid map from ACPC to raw scanner coordinates.

### 4.2 Sources of the rotation, in order of preference

1. **Direct DWI registration (preferred):** rigid registration of the raw DWI b0 (scanner world
   coordinates) to the session's `space-ACPC_dwiref`. This measures pose during the diffusion scan
   itself.
2. **Anatomical chain (cross-check):** compose the session's QSIPrep transforms
   `from-orig_to-anat` (session MPRAGE -> unbiased template) and `from-anat_to-ACPC`. This measures
   pose during the MPRAGE, which was acquired several series before the DWI. The head may have
   moved in between.

Compute both routes and report how far apart their `B0_acpc` vectors are. Large disagreement
(e.g. > 2 degrees) marks a session for review.

### 4.3 Summaries

- Head **tilt** relative to `B0`: `arccos(|B0_acpc . z_acpc|)`.
- **Pitch** (nodding, sagittal plane) and **roll** (coronal plane) components of `B0_acpc`.
- Yaw about `B0` is irrelevant because it does not change `theta`.
- **Between-session change** in the `B0_acpc` direction for each participant (the key feasibility
  quantity).
- Within-scan rotation range from the eddy motion parameters, as QC.
- Table-position translation along the bore between sessions, to use as a covariate for spatially
  varying gradient-nonlinearity effects.

### 4.4 Other modalities

The R2* positive control needs `B0` at the **MEGRE** acquisition. Pilot results (23 participants,
45 sessions; see Section 4.6):

- DWI-registration and anatomical-chain `B0_acpc` agreed to a median of 0.7 degrees (max 2.6),
  even though the DWI was acquired several series after the MPRAGE. Within-session head motion is
  therefore about 1 degree.
- Cross-contrast rigid registration of the MEGRE RMS to the ACPC T1w is under-constrained. On
  sub-22449 it did not improve NMI or brain-mask Dice over the anatomical-chain pose, yet moved
  2-3 degrees away from it. Its result also depended on the ANTs random seed (moving up to about
  5 degrees). Across participants it moved a median of 1.5 degrees (max 7.2) from the chain pose.

Therefore, **use the same-session anatomical-chain pose for MEGRE** by default, and accept about 1
degree of pose uncertainty (similar in size to the within-session DWI-vs-chain disagreement). Keep
the MEGRE registration only as a diagnostic. A better MEGRE pose would need a same-contrast target,
e.g. registering to the other session's MEGRE, or to a T2*-weighted template in ACPC space.

### 4.6 Pilot pose results (`extract_b0_pose.py`, 23 participants)

- All runs `PatientPosition == HFS`. All transforms passed the forward-vs-inverse direction check.
- Head tilt (angle between `B0` and ACPC z) from DWI: mean 11.9 degrees, SD 5.5, range 2.4-21.5.
  Nearly all of it is pitch. This is the cross-sectional spread available to Section 10.
- AP vs PA DWI runs disagree by a median of 1.4 degrees (distortion bias in the rigid fit). The
  `dwi` source averages the two.
- **Between-session change in `B0_acpc` (DWI): median 2.6 degrees, IQR 2.0-4.6, range 0.6-6.8**
  (22 participants with two sessions). The anatomical chain gives similar values (median 2.7).
- Implication for Section 6: at a 45-degree node, a 2.6-degree change moves `sin^2(theta)` by only
  about 0.045, and `sin^4(theta)` by about the same amount. The within-subject contrast is real
  but small. The power check decides whether it is usable.

### 4.5 Feasibility gate

Before any modelling, report the distribution of between-session `B0_acpc` angle changes. If
nearly all are below ~1-2 degrees, the within-subject analysis has almost no leverage. Report this
as the main finding, and fall back to Sections 10-11 as descriptive analyses only.

---

## 5. Orientation and Angle

### 5.1 Coordinate system

All vectors (`v`, `B0_acpc`) are expressed in ACPC RAS world coordinates. Voxel-axis directions
must be converted to world directions through the image affine before use.

### 5.2 Axial symmetry

Fiber direction is axial, so `v` and `-v` are equivalent. Likewise `B0` and `-B0`. Use
`abs(dot(v, B0))`.

### 5.3 Local fiber direction (pre-specified primary)

**Primary:** the MSMT-CSD fixel peak best aligned with the local streamline tangent of the bundle
at that node. This keeps orientation independent of the tensor fit that produces the outcome.

**Sensitivity (Section 12.4):** bundle tangent alone; DTI principal eigenvector.

Compute `theta` per voxel/streamline point and then summarize to the node, rather than
summarizing `v` first, because the angular function is nonlinear. Report within-node angular
dispersion.

In practice each node stores weighted orientation moments, so that the node mean of `f(theta)` is
exact for **any** `B0` vector (actual, reference, or permuted) without revisiting the images:

```text
M = <v v^T>                         # scatter tensor, 6 components
Q = <v_x^a v_y^b v_z^c>, a+b+c = 4  # fourth-order moments, 15 components
<sin^2 theta> = 1 - b^T M b
<sin^4 theta> = 1 - 2 b^T M b + sum_abc [4! / (a! b! c!)] b_x^a b_y^b b_z^c Q_abc
```

`(<sin^2 theta>)^2` is **not** a substitute for `<sin^4 theta>`: with the observed within-node
dispersion the two differ.

### 5.4 Tensor fit (pre-specified primary)

**Primary:** TORTOISE tensor with `bval_cutoff: 1200` (inner shells), which matches the DTI
literature. **Secondary:** full-shell tensor and DKI-derived tensor metrics. The expected effect
depends on b-value and TE (TE = 88 ms here). Note this when comparing with the reference study.

### 5.5 Angle variables

```text
theta      = arccos(|v . B0_acpc[s, ses]|)       # actual angle
theta_ref  = arccos(|v . B0_ref|)                # angle to a fixed reference B0
```

`B0_ref` is the mean `B0_acpc` over all participants and sessions: the direction `B0` would have in
ACPC space for an average head position. `theta_ref` carries the **anatomical** orientation.
`theta - theta_ref` is driven only by **head pose**.

---

## 6. Power Check

1. From the reference study (Section 1.1), take the FA change per unit `sin^4(theta)`:
   0.061 (0.034 to 0.088) at TE = 88 ms.
2. From Section 4.5, compute the achievable between-session `Δ f(theta)` at each retained node.
   Note that `d sin^2(theta) / d theta = sin(2 theta)` and
   `d sin^4(theta) / d theta = 2 sin^2(theta) sin(2 theta)`: small pose changes have essentially
   no leverage at nodes near 0 or 90 degrees. For `sin^4` the leverage peaks near 60 degrees.
3. Take the standard error of `β` from the participant-level bootstrap of the primary model. It
   depends on the design and the noise, not on the estimated `β`.
4. Report the **design-based power** to detect the reference estimate (two-sided 0.05), and the
   minimal detectable `|β|` at 80% power.

The standard error in step 3 comes from the fitted model, so this check is computed together with
the model and not before it. It must still be reported whatever the primary result is. Do not
compute "observed power" from the estimated `β`.

If the power for the reference estimate is low, report that as the result. It is directly useful
for designing a prospective tilted-head study.

---

## 7. Angular Function (Pre-specified)

**Primary:** `f(theta) = sin^4(theta)`. It has one degree of freedom and is monotone on [0, 90]
degrees. It is the function the reference studies use for the re-orientation analysis (Section
1.1): extra-axonal R2 follows `sin^4(theta)`, and the tensor metrics are nearly flat below about
50 degrees and change at larger angles. Using the same function makes `β` directly comparable
with the reference slope.

**Sensitivity:** `f(theta) = sin^2(theta)`, which equals `(2/3)(1 - P2(cos theta))`. This was the
primary function in the first version of this plan (Section 24).

**Secondary:** `sin^2(theta)` and `sin^4(theta)` together (equivalent to P2 + P4 in
`cos(theta)`). With small pose changes the two terms are highly collinear; report their
correlation and do not interpret the coefficients separately when it is near 1.

**Descriptive only:** linear in degrees; low-df spline.

When a nonlinear basis is used in a deviation or difference model, **difference or center the
basis, not the angle**. Use `Δ sin^4(theta)` and `sin^4(theta) - mean sin^4(theta)`, never
`sin^4(Δtheta)`.

---

## 8. Node Construction and Inclusion

### 8.1 Tractometry

The current QSIRecon outputs are **bundle-level** `scalarstats` (`masked_mean`, `masked_median`).
Node-wise profiles are new work:

1. from each AutoTrack bundle (`.tck`), resample streamlines to a fixed number of nodes (e.g. 50)
   with consistent orientation across participants and sessions;
2. sample tensor metrics, fixel directions, and fixel counts at each node (weighted mean;
   median as sensitivity);
3. keep node support and dispersion.

Because both sessions share one unbiased ACPC anatomical reference, node correspondence between a
participant's two sessions should be much better than between participants.

### 8.2 Inclusion criteria (outcome-independent)

Pre-specify, and **do not threshold on subject-level FA**. Selecting on the outcome truncates
exactly the variation being tested. Use:

- a single dominant fixel: second-peak / first-peak amplitude ratio < 0.5 at the node, in both
  sessions;
- adequate node support: >= 20 streamlines in both sessions;
- group-average inner-shell FA at the node >= 0.25 (an anatomical, not per-subject, criterion);
- exclusion of the first and last 10% of nodes (endpoints);
- exclusion of the bundles that the test-retest analyses exclude for inconsistent recognition.

The peak-ratio criterion is applied per session, so it is session-specific. Sensitivity variants
(Section 12.1): ratio < 0.3; ratio < 0.1 (the reference study's criterion); the **group-average**
ratio < 0.5 (purely anatomical); no coherence criterion.

NIBS nodes are weighted averages over streamlines, so they are less coherent than the reference
study's single-fiber voxels even at the same peak ratio. Report the within-node angular dispersion
(Section 14).

Crossing-fiber nodes may be analyzed separately as exploratory.

---

## 9. Primary Analysis: Within-Subject Between-Session Differences

For each participant `s`, tract `t`, node `n`:

```text
ΔFA[s,t,n]  = FA[s,ses2,t,n] - FA[s,ses1,t,n]
Δf[s,t,n]   = f(theta[s,ses2,t,n]) - f(theta[s,ses1,t,n])
```

Model:

```text
ΔFA ~ β · Δf + (1 | subject) + (1 | subject:tract)
```

**Pose isolation (added after pilot analysis).** The naive `Δf` above changes not only with
head pose but also with session-to-session differences in the *estimated* fiber orientation.
Those come from tractography and noise, and they co-vary with ΔFA. In the pilot data (with
`f = sin^2`) the naive FA slope (-0.021) matched the mean of its own pose-permutation null
(-0.021). In other words, it was entirely an orientation-estimate artifact, even though its
bootstrap CI excluded zero. The primary model therefore splits `Δf` into two parts, using each
node's orientation moments `M` (Section 5.3), in which `f` is linear:

```text
Δf_pose = f(M̄, b2) - f(M̄, b1)        # M̄ = (M1 + M2)/2; varies only with head pose
Δf_est  = f(M2, b̄) - f(M1, b̄)        # b̄ = mean session B0; orientation-estimate change
ΔFA ~ β · Δf_pose + γ · Δf_est + subject:tract fixed effects
```

The naive model is still reported as a sensitivity analysis, to show the artifact.

- **Fixed effects instead of random intercepts.** The model as implemented absorbs
  `subject:tract` **fixed** effects. They remove global and tract-specific session shifts
  (scanner drift, SNR, motion, tract-specific tractography differences), so `β` is identified
  only by how `Δf_pose` varies along each bundle. This costs precision. A model with
  subject fixed effects only is a sensitivity analysis (Section 12.9).
- Between-session changes in bore translation and mean motion are constant within participant.
  The fixed effects absorb them, so they cannot enter as covariates. Spatially varying
  gradient-nonlinearity effects within a bundle remain a limitation (Section 23).
- Anatomy cancels by construction: the same person and tract, with the same ACPC template.
- A fixed `tract:node` term is unnecessary in the differenced model, but can be added to check that
  results don't change.

**Inference:** the effective sample size is close to the number of participants, not the number of
observations. Pose is a subject-level rotation, and neighbouring nodes are strongly
autocorrelated. Primary uncertainty comes from a **participant-level cluster bootstrap** (resampling
participants). Report model-based intervals alongside it.

**Pose-permutation null:** permute the between-session `B0` changes across participants (keeping
each participant's anatomy and FA), recompute `Δf`, and refit. This gives a null distribution for
`β` that keeps all anatomical and noise structure and breaks only the pose link. Report the
permutation p-value as the primary test, and the mean of the null. A null that is not centered
on zero shows that part of the slope does not depend on which participant's pose is used.

---

## 10. Secondary Analysis: Cross-Sectional, Pose-Isolated

Using one session per participant (or both, with a session term):

```text
FA ~ f(theta) + f(theta_ref) + tract:node + (1 | subject) + (1 | subject:tract)
```

- `tract:node` fixed effects absorb anatomical location. Fit via within-transformation (demeaning
  by tract-node) because there are thousands of levels.
- `f(theta_ref)` absorbs the subject-specific **anatomical** orientation, including tract-shape and
  node-misalignment variation that also affects FA (for example curvature at bends).
- The coefficient on `f(theta)` is then identified by head pose alone.
- Do **not** add a `theta_mean[t,n]` term alongside `tract:node` fixed effects. It is constant
  within tract-node and therefore perfectly collinear.

The same pose-permutation null (permuting `B0_acpc` across participants) applies.

A naive model without `f(theta_ref)`, in which the across-subject angle deviation at a node is
treated as informative, should be reported only to show how much anatomical confounding it
carries.

---

## 11. Descriptive Analyses

- Dataset summary: participants, sessions, tracts, nodes, retention after each inclusion step.
- Pose summary (Section 4.3): tilt, pitch, roll per session; between-session change; agreement
  between the DWI-registration and anatomical-chain routes.
- Per-node leverage: distribution of `Δ sin^2(theta)` across participants; map of nodes with the
  highest leverage.
- Along-tract profiles of FA and `theta_ref` for representative bundles (shows the anatomical
  confound that pooled plots would mistake for an effect).
- Pooled FA vs `theta` (hexbin, faceted by tract): **descriptive only**; do not interpret pooled
  correlations.

---

## 12. Sensitivity Analyses

1. **Fiber coherence:** peak ratio < 0.3 and < 0.1 (reference-like); group-average ratio < 0.5;
   no coherence criterion (Section 8.2).
2. **Node exclusion:** endpoints kept; participant x bundle pairs dropped when their nodes moved
   by more than 5 mm (median) between sessions.
3. **Tract-by-tract:** separate fits for tracts with adequate leverage. Look at direction
   consistency, not p-values.
4. **Orientation estimator:** fixel peak (primary) vs bundle tangent vs DTI V1. Note that V1 shares
   noise with FA.
5. **Tensor fit:** inner-shell (primary) vs full-shell vs DKI.
6. **Node summary:** weighted mean vs median.
7. **Pose source:** DWI registration vs anatomical chain `B0_acpc`. For R2*, also the DWI pose and
   the MEGRE registration pose (diagnostic, Section 4.4).
8. **Leave-one-subject-out:** report the range of `β`; name any participant who drives the result.
9. **Fixed effects:** subject only, instead of `subject:tract`.
10. **Angular function:** `sin^2(theta)` instead of `sin^4(theta)`.

---

## 13. Positive Control: R2*

Repeat Section 9 (and Section 10) with MEGRE R2* as the outcome, using the MEGRE-specific `B0`
(Section 4.4) and R2* sampled at the same bundle nodes. Also fit it cross-sectionally with
`tract:node` fixed effects. The well-established R2* orientation effect should be recoverable
there.

Interpretation:

- R2* effect detected, FA not: the FA effect is below this design's sensitivity, or absent at this
  TE/b.
- R2* effect not detected: the pipeline or design lacks leverage; the FA result is uninformative.
- MESE R2 can serve as a weaker second control.

---

## 14. Measurement-Error Note

Noise in `v` (and hence `theta`) biases `β` toward zero (errors-in-variables). The bias is worse for
difference and deviation terms, whose true variance is small. Report within-node angular
dispersion and the between-estimator agreement (Section 12.4) to bound this. Do not interpret small
`β` as evidence of a small true effect without it.

---

## 15. Multiple Comparisons and Forking Paths

Primary test (one test):

- outcome FA (inner-shell TORTOISE tensor);
- orientation: fixel peak; `f = sin^4(theta)`;
- Section 8.2 inclusion criteria at their stated cutoffs;
- model: Section 9 with pose isolation and `subject:tract` fixed effects, pose-permutation
  p-value, cluster-bootstrap CI.

Everything else (AD/RD/MD, Section 10, sensitivity analyses, node-wise maps) is secondary or
exploratory. Node-wise tests, if shown, are FDR-controlled and labelled exploratory.

**This primary test was not fixed before the data were seen.** Pose isolation and the angular
function were both changed after earlier runs (Section 24). Report it as such. No change was
made because of the direction or significance of the FA result, which was null before and
after each change.

---

## 16. Effect-Size Reporting

- `β` per unit `sin^4(theta)`, with bootstrap CI;
- the implied FA difference between `theta = 0` and `90` degrees (extrapolated; say so), and between
  the observed between-session angle changes (in-sample);
- comparison with the reference study's effect size (Section 1.1): whether the CI contains the
  reference estimate, whether it overlaps the reference interval, and the design-based power
  (Section 6);
- incremental R² of the angular term;
- direction consistency across FA/AD/RD, tracts, and sensitivity analyses.

---

## 17. Diagnostics

- Residual distribution, heteroscedasticity (by node support and FA level).
- Influential participants and tract-nodes.
- Random-effect singularity. Simplify the random structure rather than reporting a singular fit.
- Collinearity between `f(theta)` and `f(theta_ref)` (Section 10). If very high, the pose-isolated
  estimate has no leverage; say so.

---

## 18. Interpretation Criteria

### 18.1 Encouraging

1. Section 4.5 shows meaningful between-session pose variation.
2. The R2* positive control is recovered.
3. The primary FA `β` is in the expected direction, the permutation test rejects the null, and
   the bootstrap CI excludes zero.
4. AD and RD move in the expected complementary directions, and both bootstrap CIs exclude zero.
5. The result is robust to stricter single-fiber criteria and is not driven by one participant or
   tract.
6. The magnitude is compatible with the reference study: the bootstrap CI contains the reference
   estimate (Section 1.1).

### 18.2 Weak

- Effect only in the cross-sectional analysis, not the within-subject one.
- Effect only in the naive model, disappearing once `f(theta_ref)` is included.
- FA only, with no compatible AD/RD pattern.
- Effect confined to crossing-fiber or low-support nodes.
- Driven by one participant.

### 18.3 Null

A null result is **not** evidence that the effect is absent, especially if Section 4.5 shows little
pose variation, the power check (Section 6) predicts low power, or the R2* control fails. Report the
detectable effect size instead.

If the power check predicts adequate power for the reference estimate and the CI excludes it, say
that the result is **in tension with** the reference estimate, not that it refutes it. The
reference slope is an upper bound for NIBS nodes (dispersion, Section 1.1), pose error attenuates
`β` (Section 14), and the reference's own uncertainty is large.

---

## 19. Figures

1. Pose distribution: per-session tilt/pitch/roll; between-session change; agreement between the
   two pose-estimation routes.
2. Per-node leverage map (`Δ sin^2(theta)`).
3. Along-tract FA and `theta_ref` profiles for representative bundles.
4. Primary: `ΔFA` vs `Δ sin^2(theta)` (binned, with fitted line and bootstrap band), with the
   permutation null distribution for `β`.
5. Same for AD, RD, and the R2* positive control.
6. Sensitivity forest plot of `β` across Section 12 variants.
7. Descriptive pooled FA vs `theta` hexbin by tract.

## 20. Tables

1. Dataset, pose, and QC summary (including inclusion counts).
2. Primary result: `β`, bootstrap CI, permutation p, incremental R², power-check expectation.
3. Secondary metrics and R2* control.
4. Sensitivity analyses and leave-one-subject-out range.

---

## 21. Pipeline Steps

0. **Pose extraction** (`extract_b0_pose.py`): `B0_acpc` per participant x session (DWI
   registration and anatomical chain), pose summaries, QC. -> Feasibility gate (Section 4.5).
1. Power check (Section 6).
2. Node-wise tractometry from AutoTrack bundles, with fixel directions and counts (Section 8).
3. Angle variables (Section 5.5) and inclusion (Section 8.2).
4. R2* positive control (Section 13).
5. Primary model (Section 9).
6. Secondary metrics, cross-sectional analysis, sensitivity analyses.

---

## 22. Recommended Primary Result Statement

> Within participants, between-session changes in fiber orientation relative to B0, arising from
> natural differences in head positioning, were / were not associated with changes in FA at the
> same tract locations. The direction was / was not consistent with prior controlled
> multi-head-position studies. The positive control (R2*) was / was not recovered. Because head
> pose was not experimentally manipulated and angle changes were small (median X degrees), this
> analysis estimates feasibility and effect-size bounds rather than a causal effect.

Avoid wording such as "B0 orientation caused the observed FA differences."

---

## 23. Main Limitations

- Head pose varies only a little and is not assigned. Pose could correlate with head size or neck
  anatomy. The within-subject design removes stable anatomy but not session-specific confounds
  that happen to covary with the pose change.
- Node correspondence, even within participant, is imperfect.
- Orientation-estimation noise attenuates effects.
- Leverage is concentrated in the few participants with large pose changes, so the effective
  number of participants is about half the nominal number.
- Tensor fits are not corrected for gradient nonlinearity, and the head position along the bore
  differs between sessions by up to about 30 mm.
- Not implemented: residual and heteroscedasticity diagnostics (Section 17), node-wise maps.

The analysis is most useful as a feasibility test, a pipeline validation, a qualitative consistency
check, and a basis for power calculations for a prospective study in which head orientation is
deliberately changed relative to `B0`.

---

## 24. Amendment Log

Changes made after data had been analysed. None was made because of the direction or
significance of the FA result.

| Date | Change | Reason |
|------|--------|--------|
| 2026-09-28 | Pose isolation: `Δf` split into `Δf_pose` and `Δf_est` (Section 9). | The naive slope matched its own permutation null: it was an orientation-estimate artifact. |
| 2026-09-28 | `subject:tract` fixed effects instead of random intercepts (Section 9). | Implementation choice; documented after review. Subject-only fixed effects added as a sensitivity analysis. |
| 2026-09-28 | Reference studies and effect sizes added (Section 1.1); power check defined as design-based power for the reference estimate (Section 6). | The reference had not been specified. |
| 2026-09-28 | Primary angular function changed from `sin^2(theta)` to `sin^4(theta)` (Section 7); `sin^2` kept as a sensitivity analysis; joint model added. | The reference studies model re-orientation with `sin^4`. |
| 2026-09-28 | Nodes store fourth-order orientation moments (Section 5.3). | Needed for the exact node mean of `sin^4(theta)`. |
| 2026-09-28 | Criterion 18.1.4 requires AD and RD CIs to exclude zero. | Signs alone were reported as "met" with p = 0.76 and 0.38. |
| 2026-09-28 | Sensitivity analyses added: peak ratio < 0.1, group-average peak ratio, node shift <= 5 mm, R2* pose sources (Section 12). | Review findings. |
