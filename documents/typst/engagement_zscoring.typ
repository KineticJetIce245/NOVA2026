#set page(paper: "us-letter", margin: (x: 2cm))
#set text(size: 10pt)
#set par(justify: true)
#set document(
  title: [Baseline-Normalised Engagement Index for PVT Lapse Detection],
)
#show raw: set text(size: 8pt)
#set math.equation(numbering: "(1)")

#let med = math.op("median")
#let MAD = math.op("MAD")

#align(center)[
  #text(size: 20pt, weight: "bold")[Baseline-Normalised Engagement Index]
  #v(0.2em)
  #text(size: 12pt)[Mathematical framing of the AttentivU pipeline and the $z$-scoring stage]
  #v(0.2em)
  #text(size: 9pt)[NOVA Buildathon 2026 --- COG-BCI / PVT]
]
#v(0.6em)

#columns(2)[

= Scope

This document formalises the algorithmic (non-learned) arm of the attention-lapse
pipeline: the AttentivU engagement index #cite(<s19235200>) computed on COG-BCI
PVT trials #cite(<hinss_2022_6874129>), and in particular the *baseline
normalisation* that makes that index comparable across electrodes, sessions and
participants.

The learned arm (EEGNet #cite(<Lawhern_2018>)) consumes the same preprocessed
tensor and is described in the main research document. The two arms are kept on
*identical windows* so that their outputs are directly comparable; §2.1 explains
why that constraint is not cosmetic.

Every numeric constant quoted below was measured on the local COG-BCI copy, not
taken from the literature. Measurements labelled *sub-01* are single-session and
illustrative; measurements labelled with a sample size are aggregates.

== Notation

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: (x: 0pt, y: 2.2pt),
  [$i$], [trial index, $1 dots N$],
  [$c$], [channel index, $1 dots C$, $C = 62$],
  [$w$], [baseline window index, $1 dots W$, $W approx 56$],
  [$s, e$], [subject and session; $s(i), e(i)$ denote those of trial $i$],
  [$f$], [frequency, Hz],
  [$f_s$], [sampling rate after resampling, $128$ Hz],
  [$T$], [window length in samples, $T = 256$ ($2000$ ms)],
  [$H$], [hop between baseline windows, $H = T\/2$],
)

Three indices do three distinct jobs and are never interchangeable: $c$ selects
an electrode, $w$ sweeps the baseline windows of one recording, and the pair
$(s,e)$ selects which recording. A subject alone does not identify a baseline ---
each participant sat three sessions and the cap was re-mounted between them, so
there are $25 times 3 = 75$ distinct baselines and $75 times 62 = 4650$
calibration pairs.

= The pipeline as a whole

== Preprocessing operator

Let $x in RR^(C times n)$ be a continuous recording. AttentivU steps 1--7 define
an operator $cal(P)$ applied *identically* to every recording, task and resting
alike:

$ cal(P) = cal(K) compose cal(B) compose cal(R) compose cal(B) compose cal(N) $

with $cal(N)$ a $60$ Hz notch ($10$ Hz width), $cal(B)$ a $4$--$20$ Hz
zero-phase Butterworth band-pass, $cal(R)$ resampling $500 -> 128$ Hz, and
$cal(K)$ the channel-wise centre--scale--clip
$v |-> min(max((v - macron(v))\/8, -4), 4)$.

Two properties of $cal(P)$ matter downstream. $cal(N)$, $cal(B)$ and $cal(R)$ are
linear and time-invariant, so any distortion they introduce is a *fixed
multiplicative factor* on band power and cancels in the difference of logarithms
taken in §2. $cal(K)$ is not: clipping is a memoryless nonlinearity whose effect
depends on the instantaneous amplitude, and it therefore does *not* cancel. §5
quantifies both.

== Two sampling domains

The operator output is sampled in two different ways.

*Trials.* For trial $i$ with stimulus onset $t_i$ and guard $delta = 100$ ms,
extract the sample set ending $delta$ before onset:
$ cal(T)_i = {t_i - delta - T + 1, dots, t_i - delta} $
$ X_i = [cal(P)x]_(dot, cal(T)_i) in RR^(C times T) $
admitted only if the preceding response cleared by more than $T\/f_s + 200$ ms,
so that no post-response activity enters the window. This is exactly the tensor
the network receives.

*Baseline.* For the eyes-open pre-task rest run `RS_Beg_EO` of the same session,
discard $tau f_s$ samples at each end ($tau = 1$ s) and tile the remainder:

$ a_w = (w - 1) H + tau f_s, quad cal(B)_w = {a_w + 1, dots, a_w + T} $
$ R_(s,e,w) = [cal(P)r]_(dot, cal(B)_w) $

for $w = 1 dots W$ with $W = floor((n - 2 tau f_s - T)\/H) + 1$, so the baseline
windows have exactly the length and sample count of a trial window.

The trim is not arbitrary: the impulse response of the zero-phase Butterworth
cascade decays below $10^(-3)$ of peak within $0.88$ s at $128$ Hz, so the
terminal windows of a continuous run are part filter transient. Trials are
exempt --- the first PVT stimulus occurs at $approx 10.9$ s.

== Spectral operator

A single estimator $cal(W)$ is applied to both domains: Welch
#cite(<welch_1967>) with Hamming taper, segment length $L = f_s$ ($1$ s),
$50%$ overlap, zero-padded to $N_"fft" = T$, averaged over the $K = 3$ resulting
segments, DC removed, evaluated on $[4, 20]$ Hz at $Delta f = f_s\/N_"fft" =
0.5$ Hz.

Fixing $L$ *independently of* $T$ decouples the estimator from window length.
Averaging $K = 3$ segments reduces estimator variance: measured window-to-window
spread of $ln E$ falls from $0.428$ ($K=1$, a single periodogram) to $0.309$.

== Engagement functional

Band powers by trapezoidal integration over *half-open* intervals:

$ P_b = integral_([f_1, f_2)) hat(P)(f) dif f $

for $theta = [4,7)$, $alpha = [7,11)$, $beta = [11,20)$ Hz. Half-open matters:
closed intervals on both edges assign $7$ Hz to both $theta$ and $alpha$ and
$11$ Hz to both $alpha$ and $beta$, inflating $theta$ by $21%$ on a realistic
$1\/f$-plus-alpha spectrum.

The index of Pope et al. #cite(<pope_1995>), in log space:

$ ell = ln P_beta - ln (P_alpha + P_theta) $ <eq-ell>

Two remarks. $ell$ is *scale-invariant*: multiplying a channel by any constant
leaves it unchanged, so the $div 8$ inside $cal(K)$ has no effect whatsoever ---
only the clip does. And $ell$ rather than $E$ is the quantity to standardise:
$E$ is a ratio of near-log-normal powers, hence right-skewed with dispersion
dominated by the upper tail, whereas $ell$ is near-symmetric and admits a
location--scale description.

Applying $cal(W)$ then @eq-ell gives $ell_(i,c)$ for trials and
$ell^"rest"_(s,e,w,c)$ for baseline windows.

== The reduction, dimension by dimension

Each $ell$ is an entire window collapsed to one number. The time axis is
destroyed at the spectral step and never reappears:

```
rest run        (62, 7674)     ch x samples
  | tile
windows         (56, 62, 256)  +window axis
  | Welch
spectra         (56, 62, 33)   time -> freq
  | integrate,log
log engagement  (56, 62)       1 per win-ch
  | fix channel c
x               (56,)          the median set
```

Concretely, for $w = 0$, channel Pz of sub-01/ses-S1: the $256$ voltages give
$P_theta = 0.2214$, $P_alpha = 0.2969$, $P_beta = 0.6723$, hence $E = 1.2970$
and $ell = 0.2600$. That single scalar is one element of the $56$-element set
standardised below.

= The $z$-score

== Why standardise at all

$E$ has no absolute meaning. Band power scales with skull thickness, hair,
electrode impedance and cap placement, so the same numeric value at two
electrodes, or on two days, does not denote the same neural state. The pipeline
therefore never asks *what is the value* but *how far is this value from what
this electrode reads, in this session, at rest* --- which requires both a
location and a scale estimated from a reference distribution.

== Stratification

Location and scale are estimated *per $(s,e,c)$*: per subject, per session, per
channel.

Per channel, because resting $P_beta\/(P_alpha + P_theta)$ is genuinely
site-dependent. Per session, because impedance and placement do not persist
across days. Not pooled across subjects, because between-subject variance is
precisely the nuisance the baseline exists to remove; pooling $sigma$ would leave
it in the numerator and let a classifier learn subject identity instead of state.
`variance_decomposition()` reports the ICC that quantifies this.

== Location and scale

For fixed $(s,e,c)$, write $x_w = ell^"rest"_(s,e,w,c)$, $w = 1 dots W$. Then

$ mu_(s,e,c) = med_w x_w $ <eq-mu>
$ sigma_(s,e,c) = 1.4826 dot med_w abs(x_w - med_w x_w) $ <eq-sigma>

with the guard $sigma <- max(sigma, 10^(-3) med_c sigma)$ against degenerate
channels. Note the median in @eq-sigma is nested: the inner one supplies the
centre, the outer one summarises the distances from it. That structure is not
exotic --- it is the structure of a standard deviation with two substitutions:

#set math.equation(numbering: none)
$ s = sqrt(op("mean")_w (x_w - op("mean") x)^2) $
$ MAD = med_w abs(x_w - med x) $
#set math.equation(numbering: "(1)")

Both say: pick a centre, measure each point's distance from it, summarise those
distances. The standard deviation uses the mean in both roles and squares; the
MAD uses the median in both roles and takes absolute values.

The constant is $1.4826 = 1\/Phi^(-1)(0.75)$, which renders the MAD Fisher-consistent
with $sigma$ under normality #cite(<rousseeuw_croux_1993>). Without it, every
$z$ would be inflated by that factor.

== Standardisation

$ z_(i,c) = (ell_(i,c) - mu_(s(i),e(i),c)) / sigma_(s(i),e(i),c) $ <eq-z>

Only $mu$ and $sigma$ survive the baseline; the $W$ windows are discarded.
$z_(i,c) = 0$ means the trial's engagement at that electrode matched the
session's resting median; $z = -1$ means one resting MAD-$sigma$ below it.

== Worked example

Channel Pz, sub-01/ses-S1. The $W = 56$ values of $ell^"rest"$:

#[
#show raw: set text(size: 6.4pt)
```
 0.260  0.942  0.498  0.414 -0.020 -0.225  0.625  0.053
-0.652 -1.004 -0.843  0.481  0.041  0.370  0.178 -0.551
-1.514 -1.683 -0.603 -0.534 -0.503 -0.816 -0.887 -0.813
-0.228  0.141  0.155 -0.267  0.077  0.745  0.032 -0.278
-0.404 -1.096 -1.411 -1.624 -1.878 -1.278 -0.645 -1.290
-1.193 -0.782 -1.175 -1.304 -1.609 -0.610  0.414 -0.662
-0.641 -0.566  0.019 -0.727 -0.729 -0.066  0.200 -0.331
```
]

*(A)* Sort; with $W$ even the median is the mean of entries $27$ and $28$:
$mu = (-0.5514 + -0.5341)\/2 = -0.5427$. The arithmetic mean would give
$-0.4606$, pulled by the low cluster in windows $32$--$47$.

*(B, C)* Deviations $x_w - mu$, then absolute values:
$0.803, 1.484, 1.041, 0.957, 0.523, 0.318, dots$

*(D)* Median of those distances: $MAD = (0.5616 + 0.5745)\/2 = 0.5681$. Half the
resting windows lie within $0.568$ of the centre.

*(E)* $sigma = 1.4826 times 0.5681 = 0.8422$. Empirically $75%$ of baseline
windows fall inside $plus.minus 1 sigma$ (Gaussian: $68%$).

*(F)* Any trial can now be scored:

#block(breakable: false)[
#table(
  columns: (auto, 1fr, auto),
  stroke: (x, y) => if y == 0 { (bottom: 0.5pt) },
  inset: (x: 3pt, y: 3pt),
  align: (right, center, right),
  [trial $ell$], [arithmetic], [$z$],
  [$-0.543$], [$(-0.543 + 0.543)\/0.842$], [$0.00$],
  [$-1.385$], [$(-1.385 + 0.543)\/0.842$], [$-1.00$],
  [$-1.600$], [$(-1.600 + 0.543)\/0.842$], [$-1.26$],
  [$+0.300$], [$(+0.300 + 0.543)\/0.842$], [$+1.00$],
)
]

== Why median and MAD rather than mean and standard deviation

Corrupting a single one of the $56$ windows with a blink-sized artefact
($+12$ log units) and refitting:

#block(breakable: false)[
#table(
  columns: (1fr, auto, auto, auto),
  stroke: (x, y) => if y == 0 { (bottom: 0.5pt) },
  inset: (x: 3pt, y: 3pt),
  align: (left, right, right, right),
  [], [centre], [$1.4826 MAD$], [$s$],
  [clean], [$-0.5427$], [$0.8422$], [$0.6821$],
  [one bad window], [$-0.5427$], [$0.8422$], [$1.8306$],
  [change], [$0.0%$], [$0.0%$], [$+168%$],
)
]

The robust pair is unmoved --- a single outlier cannot change which value sits in
the middle. The standard deviation nearly triples, and since every trial is
divided by it, that one artefact would compress all $90$ of this channel's trial
$z$-scores toward zero by a factor of $2.7$.

*Caveat.* The constant $1.4826$ assumes approximate normality. On this channel
the resting distribution is platykurtic (windows $32$--$47$ sit systematically
lower), so $1.4826 MAD = 0.842$ exceeds $s = 0.682$ by $23%$, where on the
channel-median they agree ($0.556$ vs $0.555$). Neither is "the true $sigma$";
the argument for the robust pair is the table above, not superior accuracy.

= Channel aggregation

Aggregation happens *after* standardisation, never before: the $mu_c$ differ
systematically by site, so averaging raw $ell$ across channels would combine
incommensurable offsets.

The channel mean of $C$ unit-dispersion variates is not itself unit-dispersion
unless the channels are independent. Its scale is therefore estimated on the same
baseline windows:

$ g_(s,e,w) = 1/C sum_c (ell^"rest"_(s,e,w,c) - mu_(s,e,c))/sigma_(s,e,c) $
$ sigma^G_(s,e) = 1.4826 dot MAD_w g_(s,e,w) $

giving one score per trial,

$ macron(Z)_i = 1/(sigma^G_(s(i),e(i))) dot 1/C sum_(c=1)^C z_(i,c) $ <eq-Z>

Measured $sigma^G in [0.69, 1.00]$ across four sessions, against $C^(-1\/2) =
0.127$ under independence, because the mean inter-channel correlation of $z$ is
$0.47$. The effective dimensionality is

$ n_"eff" = (sigma^G)^(-2) approx 1 "to" 2 $

*Consequence for weighting.* With $1$--$2$ effective dimensions there is
essentially nothing for a weighting scheme to exploit, and the schemes agree
accordingly (correlation across $90$ trials): uniform mean vs median $r = 0.988$,
vs $20%$ trimmed mean $r = 0.998$, vs the Pope electrode subset $r = 0.818$.
The uniform mean is therefore used. Fitting weights is deliberately avoided here
--- a parameter-free algorithmic arm is what makes a later comparison against
attention pooling interpretable.

= Validation

Three checks, in order of precedence.

+ *Construct validity.* $macron(Z)$ of `RS_Beg_EC` scored against the
  `RS_Beg_EO` baseline. Eyes-closed alpha enters the denominator of @eq-ell, so
  this must be clearly negative. Measured sub-01: $-0.84 sigma$, against
  $-0.01 sigma$ for held-out eyes-open windows. If this fails, nothing
  downstream is interpretable.

+ *Stratification necessity.* ICC of raw $ell$ across subjects, from
  `variance_decomposition()`. A high value confirms that pooling $sigma$ across
  subjects would have been inadmissible.

+ *Primary contrast.* $EE[macron(Z) | y = 1] - EE[macron(Z) | y = 0]$, paired
  within session and aggregated across sessions, where $y = 1$ marks the
  within-session slowest decile of reaction time.

= Assumption audit

*Band-edge attenuation.* The $4$--$20$ Hz band-pass is applied twice with zero
phase, giving measured power transfer of $-11.6$ dB at $4$ Hz and $-11.8$ dB at
$20$ Hz --- roughly $7%$ of power surviving at both ends of the analysis range.
Because this is LTI and identical on both sides of @eq-z, it cancels in the
difference of logarithms to first order; the residual effect on $E$ is
$times 1.07$. The cost is signal-to-noise, not bias.

*Window-length coupling.* $E$ is strongly dependent on segmentation: the same
resting signal yields $macron(E) = 0.4219$ measured in $2$ s windows and
$0.2397$ in $5$ s windows, a $43%$ shift from segmentation alone. This is why
the baseline is tiled at exactly $T$ and not at the $5$ s of the original
AttentivU formulation. Restoring $5$ s windows would additionally require a
$5200$ ms clear gap, retaining only $68%$ of trials ($61.3$ vs $89.8$ per
session, $n = 29$).

*Clipping.* $cal(K)$ is nonlinear and amplitude-dependent: the correlation
between window peak amplitude and induced $Delta ell$ is $+0.44$, so this bias
does *not* cancel. It engages on $62%$ of PVT trials and $10.7%$ of
(trial, channel) pairs, concentrated occipitally where alpha is largest. Mean
effect $+0.009$ in high-alpha windows versus $0.000$ in low-alpha; worst case
$0.25 sigma$. This is the one preprocessing step whose bias reaches the
$z$-score.

*Baseline non-stationarity.* Channels that degrade *between* the rest block and
the PVT shift $macron(Z)$ substantially ($-0.75 sigma$ at $12$ affected
channels, $-1.47 sigma$ at $24$), but the correlation with the uncontaminated
composite remains $gt.eq 0.995$: the effect is a near-constant offset, which
cancels in a within-session contrast. Channels that are simply bad *throughout*
are absorbed almost exactly ($+0.009 sigma$ at $24$ affected channels), because
the baseline sees the same corruption the trials do.

= Implementation map

All of §2--§4 lives in `scripts/dataproc/prototype/engagement.py`.

#show raw: set text(size: 7pt)
#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: (x: 0pt, y: 2.2pt),
  column-gutter: 4pt,
  [`welch`], [$cal(W)$, §1.3 --- shared by both domains],
  [`band_power`], [$P_b$, half-open integration],
  [`engagement_index`], [@eq-ell],
  [`epoch`], [baseline tiling + edge trim, §1.2],
  [`baseline_from_rest`], [@eq-mu, @eq-sigma, $sigma^G$],
  [`fit_baselines`], [one `Baseline` per $(s,e)$],
  [`normalize`], [@eq-z, returns $N times 62$],
  [`normalize_composite`], [@eq-Z, returns $N$],
  [`baseline_sanity_check`], [validation 1],
  [`variance_decomposition`], [validation 2],
  [`lapse_contrast`], [validation 3],
)

Window length and hop are derived from `config.SAMPLE_SIZE` rather than restated,
so the baseline cannot silently desynchronise from the trial tensor that
`build_pvt.py` produces. `build_rest.py` must be run with
`DATASET = "RS_Beg_EO"` for the baseline and `"RS_Beg_EC"` for validation 1.

#bibliography("../../refs.bib", title: "References")
]
