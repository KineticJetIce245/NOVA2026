#set page(paper: "us-letter", margin: (x: 2cm))
#set text(size: 9pt)
#set par(justify: true)
#set document(
  title: [The Engagement Index, Explained from Scratch],
)
#show raw: set text(size: 8pt)
#set math.equation(numbering: "(1)")

#let med = math.op("median")
#let MAD = math.op("MAD")

#align(center)[
  #text(size: 20pt, weight: "bold")[The Engagement Index, Explained from Scratch]
  #v(0.2em)
  #text(size: 11pt)[A plain-language walkthrough of the engagement-z-scoring pipeline, for readers with no signal-processing background]
  #v(0.2em)
  #text(size: 9pt)[NOVA Buildathon 2026 --- COG-BCI / PVT]
]
#v(0.6em)

#columns(2)[

*How to use the equations.* Every numbered equation below is the *exact* formula from
`documents/engagement_zscoring.pdf`, and its number matches the PDF (e.g. @eq-ell in
this document is Eq. (7) there). Whenever a formula in the PDF looks intimidating,
find its number here: the surrounding prose is the plain-English version of that same
equation.

= The big picture: what are we trying to do?

We are studying *attention* and *attention lapses* (moments when the mind wanders
and a person reacts too slowly). To study this, people wear an electrode cap on
their scalp, which records the tiny electrical signals their brain produces. That
recording is called an *EEG*.

The problem is that a raw EEG is a long, messy, constantly-changing stream of
numbers --- one number per electrode, sampled $128$ times every second. Reading a
classroom's attention state directly from those raw numbers is hopeless.

So we build a single number, the *engagement index*, that is supposed to summarise
"how engaged or attentive is this brain right now?" from a short snapshot of EEG.
Then we compare that number across people and across moments.

The purpose of this document is to walk through, step by step, *why* each
transformation exists and *what each resulting number means*. Every step below has
a reason; nothing is done "because it looks technical."

= The raw material: what is an EEG signal?

Think of an electrode as a microphone, but for electricity instead of sound. It
records the voltage between that spot on the scalp and a reference point. This
voltage changes continuously over time because millions of brain cells are firing.

This gives us, for each electrode, a *wave* --- a value that rises and falls over
time. Two of the most important facts about this wave:

1. *It is a mix of many frequencies.* The brain produces several rhythms at once,
   layered on top of each other. A bit of slow rolling (theta), a bit of medium
   rolling (alpha), a bit of faster buzzing (beta).
2. *The wave is huge and meaningless on its own.* The absolute voltage at any
   single moment depends on random stuff --- how thick the skull is, whether the
   electrode is pressed firmly, how much scalp oil is present. The same brain state
   can produce wildly different raw voltages on two different days.

These two facts drive everything that follows.

#quote(block: true)[
  *A concrete example of "what we're holding."* Picture one electrode over $2$
  seconds of recording. It is a list of $256$ numbers (one per sample at $128$
  Hz), each a voltage in microvolts, bouncing up and down irregularly --- say
  "$12.3, 11.8, -2.1, 8.9, 9.4, ...$". If we could peel that single wavy line
  apart, we would find three distinct rhythms added together: a slow theta wave
  turning over about $5$ times per second, an alpha wave turning over about $9$
  times per second, and a fast beta wave turning over about $15$ times per
  second. The single number we see at each instant is just the sum of all three
  (plus noise). "Peeling them apart" is precisely what the Fourier step later in
  this document does --- the reason we have to do it is that we cannot measure
  "how strong is beta" by staring at the raw voltage line; the rhythms are
  tangled together in it.
]

= Step 0: cleaning the signal before we measure anything

Before we can measure "how much beta is there," we must make sure the signal only
contains brain activity we care about, not electrical junk.

- *Power-line noise.* Mains electricity in the wall hums at $60$ Hz (in North
  America). The body acts like an antenna and picks up this hum, drowning out the
  small brain signal. So we remove a narrow band of frequencies around $60$ Hz.
  This is the "notch" filter, $cal(N)$.
  #quote(block: true)[
    *How you'd implement it.* First transform the $2$-second window into its
    frequency parts (the same Fourier idea from later in this document). The hum
    appears as a sharp spike sitting exactly at $60$ Hz and nowhere else. You
    simply set to zero every frequency slice inside a narrow band around that
    spike --- say $55$ to $65$ Hz --- then transform back to a voltage line. The
    brain rhythms at $4$--$20$ Hz were far away from $60$ Hz, so they are left
    almost untouched, but the steady $60$ Hz wobble is gone.
  ]
- *Only keep the interesting range.* Brain rhythms of interest live between roughly
  $4$ and $20$ Hz. Everything outside --- slow drift, fast muscle noise --- is not
  what we want. So we apply a "band-pass" filter $cal(B)$: keep only $4$--$20$ Hz,
  discard the rest.
  #quote(block: true)[
    *How you'd implement it.* The same transform trick, but this time you keep a
    *band* instead of removing one. Every frequency slice below $4$ Hz and above
    $20$ Hz is set to zero, leaving only the $4$--$20$ Hz band. In practice this
    is done with a "Butterworth" filter, which is a recipe that adjusts each
    sample so the result passes frequencies inside the band and suppresses those
    outside. Notice the filter is applied *twice* in the pipeline (before and
    after resampling) --- each application is the same operation.
  ]
- *Speed up the data a bit.* The raw recorder saved $500$ samples per second, which
  is more than we need for rhythms under $20$ Hz. We re-sample $cal(R)$ down to
  $128$ samples per second. This makes the data smaller and easier to work with.
  #quote(block: true)[
    *How you'd implement it.* You go from $500$ samples per second to $128$, so
    you keep roughly one in four samples. You can't just throw away the others
    (that would distort the wave), so instead you smoothly "re-place" the
    samples onto the new, coarser grid --- for each new sample position, you
    blend the surrounding old samples together. After resampling, $2$ seconds of
    data holds $256$ samples instead of $1000$. This is done *after* the first
    band-pass so that the removed fast content cannot fold back into the slower
    band.
  ]
- *Make the numbers comparable between people.* Different recordings have wildly
  different average sizes and ranges. We "centre" each channel (subtract its own
  average) and scale it, and clamp extreme spikes. This is the operation $cal(K)$.
  It is like turning the volume knob on each person's recording so that everyone is
  roughly at the same volume before we compare them.
  #quote(block: true)[
    *How you'd implement it.* Take one electrode's $256$ voltages and compute
    their average, say $3.2$ microvolts. Subtract that average from every sample
    (so the new average is $0$), then divide every sample by $8$ to shrink the
    range, then push any value outside $[-4, 4]$ to exactly $4$ or $-4$. This is
    done separately for each electrode, so each electrode ends up centred on $0$
    with a similar range. The clip is the one piece of cleaning that does *not*
    cancel out later --- a deliberate, acknowledged small bias.
  ]

Put together, the whole cleaning stage is one operation applied identically to every
recording, and the design document writes it as one expression:

$ cal(P) = cal(K) ∘ cal(B) ∘ cal(R) ∘ cal(B) ∘ cal(N) $ <eq-p>

(Read it right-to-left: first notch $cal(N)$, then band-pass $cal(B)$, then resample
$cal(R)$, then band-pass again $cal(B)$, then normalise-and-clip $cal(K)$.)

A nice side effect: the notch, band-pass and resample steps are *linear*, meaning
they change the size of the waves by a fixed factor. As we'll see, the way we
compute the index uses a ratio, and fixed factors cancel out of a ratio. So most of
the cleaning is *invisible* in the final number --- which is exactly what we want.

= Step 1: chop the recording into short windows

The EEG is continuous. But we don't want to measure the brain over the whole day ---
we want to measure it at specific moments, in short pieces.

Why short pieces? Because "attention" changes second by second. A $2$-second
snapshot is a good size: long enough to get a reasonable measurement, short enough
to represent "how the brain is *right now*" rather than "how it was all day."

So we cut the recording into *windows*, each exactly $2$ seconds long. The design
document records, for each trial $i$, which exact samples go into its window. If
$t_i$ is when the stimulus appears and $delta$ is a $100$ ms guard, the window is
the $2$ seconds of samples ending just before the stimulus:

$ cal(T)_i = {t_i - delta - T + 1, dots, t_i - delta} $ <eq-trial>

$ X_i = [cal(P)x]_(dot, cal(T)_i) in RR^(C times T) $ <eq-x>

*Plainly:* @eq-trial says *which* $2$ seconds we grab (the chunk ending $100$ ms
before the stimulus), and @eq-x says the result is a little table with one row per
electrode $C$ and one column per sample $T$. This is the *trial window* --- the
moment we want to score, measuring the brain *before* it responds.

We also need *baseline windows* from the quiet rest period. We chop the rest period
the same way, using a sliding position $a_w$ for the $w$-th window:

$ a_w = (w-1) H + tau f_s, quad cal(B)_w = {a_w + 1, dots, a_w + T} $ <eq-a>

$ R_(s,e,w) = [cal(P)r]_(dot, cal(B)_w) $ <eq-r>

*Plainly:* @eq-a says where each rest window starts (spaced $H$ apart, after
trimming $tau$ seconds off each end of the recording), and @eq-r says we extract
those windows from the rest recording $r$. The subscript $(s,e,w)$ labels it:
subject $s$, session $e$, window $w$.

A crucial detail: the baseline windows and the trial windows must be *exactly the
same length* $T$. If we measured rest over $60$ seconds and trials over $2$ seconds,
the numbers would sit on different scales and be impossible to compare. By chopping
rest into identical $2$-second pieces, we make rest and trials directly comparable.
(The overlapping $50%$ hop $H = T\/2$ is just a way to get more windows out of the
same stretch of data --- more windows means a more reliable "baseline" description.)

#quote(block: true)[
  *A concrete example of the implementation.* Suppose the stimulus appears at
  $10.5$ s into a session. With a $100$ ms guard and $2$ s windows, we grab the
  $256$ samples spanning $8.3$--$10.4$ s (the last sample sits $100$ ms before
  the stimulus). This becomes one trial window, laid out as a $62 times 256$
  table (one row per electrode, one column per sample). Separately, the quiet
  rest run might be $60$ s of data: we drop $1$ s off each end (filter
  transient), leaving $58$ s, then slide a $2$-second window across it in
  half-second steps. Because the window is $2$ s and the step is $1$ s, the
  slides overlap, so we pull out roughly $56$ baseline windows from that one run
  --- each still $62 times 256$, exactly the same shape as a trial window.
]

= Step 2: from "voltage over time" to "power at each frequency"

Here is where signal processing enters, but the *idea* is simple.

A $2$-second EEG window is a wavy line. That wavy line is actually a mixture of
several different waves --- one slow, one medium, one fast --- all added together.

The Fourier transform (implemented here by a tool called Welch's method) is a
mathematical trick that *separates the mixture back apart*. It asks: "If I unmix
this wave, how much energy (power) is there in each frequency band?"

The output is a *power spectrum*: a table saying "the slow part has this much power,
the medium part has this much power, the fast part has this much power."

Why Welch's method specifically? Because a single, raw measurement of the power at
each frequency is very noisy and jumps around randomly. Welch's method slices the
$2$-second window into smaller overlapping pieces, measures the power in each piece,
and *averages them*. Averaging smooths out the random noise, giving a steadier, more
trustworthy spectrum. (Measured effect: the window-to-window wobble in our final
index drops noticeably once we average --- the estimate becomes more reliable.)

So after this step, each $2$-second window is no longer $256$ numbers of voltage;
it's a short list of "power at frequency $4$ Hz, power at $4.5$ Hz, ... power at
$20$ Hz." The time axis is gone. We collapsed $2$ seconds of activity into a
frequency fingerprint.

#quote(block: true)[
  *A concrete example of the implementation.* Take one $2$-second trial window
  from one electrode. First slice it into $3$ overlapping $1$-second pieces.
  Transform each $1$-second piece separately, so each gives you a rough
  "how much power at each frequency" curve. The single curve jumps around, but
  the three curves agree on the general shape, so you average them at every
  frequency. The result is a smooth curve: at $9$ Hz it might read "power
  $0.30$", at $15$ Hz "power $0.67$", and so on --- one number per frequency from
  $4$ Hz to $20$ Hz (about $33$ frequencies, spaced $0.5$ Hz apart). That is the
  *power spectrum* of the window. If the window had contained a strong alpha
  burst, you would see a clear bump in the spectrum around $8$--$12$ Hz.
]

= Step 3: compute three band powers

The brain rhythms we care about each occupy their own slice of the frequency range:

- *Theta* --- the slowest band, roughly $4$--$7$ Hz.
- *Alpha* --- the medium band, roughly $7$--$11$ Hz.
- *Beta* --- the faster band, roughly $11$--$20$ Hz.

For each window, we add up all the power inside each band. The design document
writes this "add up the power over a band" as an integral --- just a fancy way of
saying *sum the spectrum's height over that frequency range*:

$ P_b = integral_([f_1, f_2)) hat(P)(f) dif f $ <eq-pb>

*Plainly:* @eq-pb is "the power in band $b$ is the area under the spectrum curve
between frequencies $f_1$ and $f_2$." We do this three times --- once per band ---
giving three numbers per window:

- $P_"theta"$ = total power in the theta band,
- $P_"alpha"$ = total power in the alpha band,
- $P_"beta"$ = total power in the beta band.

These three numbers answer "how strong is each rhythm in this moment?"

A subtle but important detail: the band edges ($7$ Hz and $11$ Hz) are shared between
neighbouring bands. The bracket notation $[f_1, f_2)$ in @eq-pb means we count the
*left* edge but *not* the right edge, so each frequency is counted in only *one*
band. If we counted the boundary frequency in both, we'd double-count it and inflate
the numbers. It's a small fix, but it keeps the measurements honest.

#quote(block: true)[
  *A concrete example of the implementation.* We have the spectrum as about $33$
  numbers, one per frequency from $4$ to $20$ Hz. To get theta power we add up
  the spectrum values at the frequencies $4.0, 4.5, ..., 6.5$ Hz (everything
  from $4$ Hz up to but not including $7$ Hz) --- say that sums to $0.22$. For
  alpha we add the values at $7.0$ through $10.5$ Hz, giving $0.30$; for beta,
  $11.0$ through $19.5$ Hz, giving $0.67$. The half-open rule means $7$ Hz is
  counted in alpha only (not theta) and $11$ Hz in beta only (not alpha); if we
  double-counted those boundary bins the theta and alpha totals would each be
  slightly too big.
]

= Step 4: combine the three powers into one engagement index

Now we have three numbers, but we want one. The classic way to combine them is a
*ratio* --- and specifically, a ratio that is high when a person is engaged and low
when they are not. The design document writes it in log form:

$ ell = ln P_beta - ln(P_alpha + P_theta) $ <eq-ell>

*Plainly,* @eq-ell is the same as $ln(P_beta \/ (P_alpha + P_theta))$ --- the natural
logarithm of the ratio:

```text
Engagement = P_beta / (P_alpha + P_theta)
```

Why this ratio? Research in attention suggests that an engaged, focused brain has
relatively strong beta activity, whereas a relaxed, inattentive brain has stronger
alpha and theta. So:

- When beta is strong relative to alpha+theta $->$ ratio is large $->$ *engaged*.
- When alpha+theta dominate $->$ ratio is small $->$ *not engaged*.

The reason it's a *ratio* (rather than just "beta power") is the scale problem from
earlier. Because electrode quality and skull thickness multiply all the band powers
by the same unknown factor, taking a ratio *cancels that factor*. Two people can have
completely different raw voltages but identical engagement ratios. This is the single
most important trick in the whole method: *the ratio is a "calibration-free" way to
describe the brain, immune to the physical quirks of the recording.* (This is why
$ell$ is called *scale-invariant* in the PDF.)

Why the log? Look at the numbers. Band powers can range from tiny to huge, and a
ratio of two such quantities is heavily skewed --- it has a long tail of very large
rare values. That skew makes statistics on it unreliable. Taking the logarithm
*squeezes the range down* and makes the distribution roughly symmetric and
"bell-shaped," which is the shape most statistical tools expect. Because
$ln(a\/b) = ln(a) - ln(b)$, @eq-ell is still just "beta vs alpha+theta," but on a
nicer scale.

*What the number means now:* a positive $ell$ means beta is relatively strong
(engaged); a negative $ell$ means alpha+theta are relatively strong (less engaged);
zero means they're balanced. And importantly, the value is measured on a stable,
comparable scale regardless of electrode quality.

#quote(block: true)[
  *A concrete example of the implementation.* Carry the band powers from the
  previous example: $P_"theta" = 0.22$, $P_"alpha" = 0.30$, $P_"beta" = 0.67$.
  First add the denominator: $P_"alpha" + P_"theta" = 0.52$. Then the ratio is
  $0.67\/0.52 = 1.29$, and the engagement value is its natural logarithm,
  $ln(1.29) approx 0.26$. Now scale one electrode up by a factor of $10$ (say a
  looser cap): all three powers become $2.2, 3.0, 6.7$. The ratio is still
  $6.7\/5.2 = 1.29$ and $ell$ is still $0.26$ --- the $10 times$ change vanished.
  That is exactly the "scale-invariance" the ratio buys us. A window where beta
  collapses (low engagement) might give $ell approx -1.3$; a very attentive
  window might give $ell approx +1.0$.
]

= Step 5: the problem of "no absolute meaning" --- why we need a baseline

Here is the remaining problem. Even though the ratio cancels electrode quality, the
*typical* engagement level is still different for every electrode and every person
on every day. One person's "neutral" might be another person's "very engaged."

So we can't just look at an engagement number and say "this is high." We need a
*reference point* for each person, each session, each electrode.

That reference comes from the *baseline*: the quiet, eyes-open rest period we chopped
into windows earlier. This rest period represents "this person, this session, this
electrode, just sitting there." It is their personal "what normal looks like."

The question we then ask for any moment is not "what is the engagement value?" but:

#quote(block: true)[
  *How far is this moment's engagement from what this electrode reads, for this
  person, in this session, while resting?*
]

This shift from "absolute value" to "distance from personal baseline" is the whole
point of the pipeline.

#quote(block: true)[
  *A concrete example of the implementation.* Suppose we are working with the
  electrode labelled Pz for subject 01, session S1. From the quiet eyes-open rest
  run we compute the engagement value $ell$ for each of the ~$56$ resting windows,
  producing a small list of numbers like "$0.26, 0.94, 0.50, 0.41, ...$". These
  ~$56$ values are this electrode's own "what resting looks like" for this person
  on this day. Later, a trial from the same person and session gets its own
  $ell$, and we judge it against exactly this list --- never against a list from a
  different electrode, a different session, or a different person.
]

= Step 6: measure the baseline with a "centre" and a "spread"

To measure "how far from rest," we need two things for each (person, session,
electrode):

1. *The centre* --- what is the typical resting engagement for this electrode?
2. *The spread* --- how much does the resting engagement normally wobble around that
   centre?

For each electrode we compute the engagement index $ell$ on every resting window,
giving us a little collection of resting values (about $56$ of them). The design
document calls a resting value $x_w$ (the engagement of resting window $w$), and
defines the two quantities as:

$ mu_(s,e,c) = med_w x_w $ <eq-mu>

$ sigma_(s,e,c) = 1.4826 dot med_w abs(x_w - med_w x_w) $ <eq-sigma>

*Plainly:*

- @eq-mu is the *centre* $mu$ --- the *median* of the resting values (the middle value
  when sorted).
- @eq-sigma is the *spread* $sigma$ --- how far, on average, the resting values stray
  from that middle. Read it inside-out: first find the middle value ($med_w x_w$),
  then measure each window's distance from it (the $abs(...)$), then take the median
  of those distances. The $1.4826$ is a purely mathematical scale adjustment so this
  spread comes out in familiar "standard deviation" units.

Why the median and a "robust" spread instead of the simpler mean and standard
deviation? *Because EEG is full of occasional junk.* A blink, a movement, or a loose
electrode can inject one terrible resting window --- a huge outlier.

- The *mean* gets dragged around by a single bad value.
- The *median* barely moves: it's defined by what's in the middle, so one bad window
  can't change it.

Similarly, the standard deviation is very sensitive to outliers (it squares the
distances, so a big outlier dominates). The robust spread in @eq-sigma, based on the
median of the absolute distances, is far more resistant. In practice, corrupting a
single resting window with an artefact inflates the standard deviation by $168%$ but
leaves @eq-mu and @eq-sigma completely unchanged. Since we later divide every trial
by this spread, a robust measure protects the whole dataset from one bad moment.

There is also a safety guard in the implementation: if any electrode's spread comes
out suspiciously near zero, it is nudged up to a tiny floor. This stops a degenerate
electrode (e.g. one that read almost flat) from producing absurdly huge z-scores
later.

#quote(block: true)[
  *A concrete example of the implementation.* Take the ~$56$ resting values for
  electrode Pz from the previous example. First sort them. The middle value
  (between positions $27$ and $28$ of $56$) is the *centre*; say it comes out to
  $-0.54$. Next measure how far each resting value is from this centre, giving
  $56$ positive distances like "$0.80, 1.48, 1.04, ...$". The middle of *those
  distances* is the MAD, say $0.57$ --- meaning half the resting windows sit
  within $0.57$ of the centre. Multiply by $1.4826$ to get the spread $sigma =
  0.84$. So this electrode's "normal resting engagement" is described as
  "centred at $-0.54$, wobbling by about $0.84$." If one resting window was a
  blink artefact worth $+12$, sorting still puts it at an extreme end, so the
  middle value and the middle distance barely move --- the whole baseline stays
  almost identical.
]

= Step 7: score every moment as a z-score (distance from the baseline)

Now we can score any moment. For each electrode, we take the moment's engagement
value and ask "how many baseline-spreads is this away from the baseline centre?"

$ z_(i,c) = (ell_(i,c) - mu_(s(i),e(i),c)) / sigma_(s(i),e(i),c) $ <eq-z>

*Plainly,* @eq-z says: take trial $i$'s engagement $ell$ at electrode $c$, subtract
that electrode's resting centre $mu$, and divide by that electrode's resting spread
$sigma$. The subscripts $s(i),e(i)$ just remind us that we use the centre and spread
of the *same person and session* the trial came from.

*What the number means* --- this is the key intuition to keep:

- $z = 0$ $->$ this moment is *exactly at* the person's resting-normal level.
- $z = +1$ $->$ this moment is *one spread above* normal (more engaged than rest).
- $z = -1$ $->$ this moment is *one spread below* normal (less engaged than rest).
- $z = -2$ $->$ unusually *inattentive* compared to this person's own rest.

Because it's measured in units of that person's own spread, a z-score is *comparable
across electrodes, sessions, and people*. This is the "standardisation" step. Now "1"
means the same thing everywhere: "one personal spread above that electrode's own
resting normal."

#quote(block: true)[
  *A concrete example of the implementation.* Using the Pz baseline above
  ($mu = -0.54$, $sigma = 0.84$), score a few trial windows for the same
  electrode:

  #table(
    columns: (1fr, 1.6fr, 1fr),
    align: center,
    table.header([*trial $ell$*], [*calculation*], [*z*]),
    [$-0.543$], [($-0.543 + 0.543$)\/$0.842$], [$0.00$],
    [$-1.385$], [($-1.385 + 0.543$)\/$0.842$], [$-1.00$],
    [$-1.600$], [($-1.600 + 0.543$)\/$0.842$], [$-1.26$],
    [$+0.300$], [($+0.300 + 0.543$)\/$0.842$], [$+1.00$],
  )

  The trial with $ell = -1.385$ sits exactly one resting-spread below this
  electrode's normal (z-score $-1$), and the trial with $ell = +0.300$ sits one
  spread above (z-score $+1$). A trial scoring $-2$ would be unusually
  inattentive compared with this electrode's own rest. Crucially, the same z =
  $-1$ means the same thing on a different electrode or a different person's
  session, because each was standardised against its *own* centre and spread.
]

= Step 8: combine the 62 electrodes into one overall score

We have $62$ electrodes, so each moment has $62$ z-scores. We want one number per
moment.

The obvious thing would be to average the $62$ z-scores. But there's a subtlety:

- We must *standardise first, then average* --- never average the raw engagement
  values and then standardise. Because each electrode has its own baseline centre,
  averaging the raw values first would mix incompatible reference points.

The design document first defines the *channel-averaged resting z-score* $g$ for each
resting window $w$ (this is used only to calibrate the final scale):

$ g_(s,e,w) = 1/C sum_c (ell^"rest"_(s,e,w,c) - mu_(s,e,c)) / sigma_(s,e,c) $ <eq-g>

then measures how much that average wobbles across the resting windows --- i.e. the
*actual spread of the average*:

$ sigma^G_(s,e) = 1.4826 dot MAD_w g_(s,e,w) $ <eq-sg>

and finally produces the one score per trial $i$:

$ overline(Z)_i = 1/sigma^G_(s(i),e(i)) dot 1/C sum_(c=1)^C z_(i,c) $ <eq-zbar>

*Plainly:* @eq-zbar is "average the $62$ per-electrode z-scores (the $1/C sum z$ part),
then divide by $sigma^G$." The division by $sigma^G$ is the correction.

Why is that correction needed? A z-score is defined to have a spread of $1$. If you
averaged $62$ *independent* things each with spread $1$, the average would have a tiny
spread (about $1\/sqrt(62)$). But our $62$ electrodes are *not* independent ---
neighbouring electrodes on the scalp pick up very similar brain signals, so they're
strongly correlated. In effect, the $62$ electrodes behave more like *1 or 2
independent measurements*, not $62$.

So we don't assume how much the average should wobble --- we *measure* it in @eq-g and
@eq-sg using the resting windows, then divide by it in @eq-zbar. This rescales the
combined score back into meaningful spread units, so our combined number still reads
like "this many baseline spreads above normal," just like a single electrode's
z-score.

*Final meaning:* after this step, each moment gets *one number* --- a composite
engagement z-score. Roughly:

- a clear negative value $->$ that moment looks inattentive compared to rest;
- near zero $->$ normal;
- a clear positive value $->$ more engaged than rest.

#quote(block: true)[
  *A concrete example of the implementation.* Suppose a trial has z-scores across
  its $62$ electrodes such that the plain average is $-0.40$. If the $62$
  electrodes were independent, that average would wobble very little (about
  $0.13$), so dividing by $0.13$ would give an enormous, meaningless number. But
  the electrodes are correlated, so the measured resting wobble $sigma^G$ of the
  average is much larger --- say $0.85$. We divide the trial's average by that
  measured value: $-0.40\/0.85 approx -0.47$. So this trial's *composite* score is
  about $-0.5$ --- roughly half a resting-spread below normal for the whole head.
  This one number is what a downstream step (predict.py) turns into a
  lapse / not-a-lapse decision.
]

= Step 9: check the whole thing actually measures something

Before trusting these numbers, the method runs three sanity checks. These are "does
this make sense?" tests rather than part of the pipeline itself:

1. *Eyes-closed check.* When people close their eyes, their brain produces a big burst
   of alpha activity. Alpha is in the denominator of our ratio @eq-ell, so closing the
   eyes should push the engagement score *clearly negative*. If we score an
   eyes-closed rest period against the eyes-open baseline and it comes out strongly
   negative, the index is behaving exactly as it should --- it is really responding to
   brain state, not just noise. If it *doesn't* go negative, something is wrong with
   our bands or our measurement.
2. *Do we actually need per-person baselines?* We measure how much of the variation in
   engagement is due to *which person* it is, versus *what they're doing*. If person
   identity dominates, that confirms we *must* use personal baselines and cannot pool
   everyone together.
3. *Does it predict lapses?* The task records reaction times. We label the slowest
   $10%$ of reactions as "lapses." We then check whether the engagement score is lower
   on lapse moments than on normal moments. If it is, the index has real predictive
   value --- it measures the thing we care about.

#quote(block: true)[
  *How each check is implemented.*
  (1) *Eyes-closed.* We have the same person's eyes-closed rest run. We score it
  against the eyes-open baseline (each electrode's $mu$ and $sigma$). Because
  closing the eyes floods the alpha band, which sits in the denominator of
  @eq-ell, the composite should come out clearly negative --- e.g. $-0.8$. If it
  instead came out near $0$, the bands or the spectrum would be wrong.
  (2) *Per-person necessity.* We take the raw $ell$ values and measure how much
  of their variance is "between subjects" versus "within a subject." A high
  between-subject share (a high ICC) confirms that pooling everyone into one
  baseline would have let a classifier learn *who* a person is rather than *how
  attentive* they are.
  (3) *Lapse contrast.* Within each session we compare the mean composite score
  of lapse trials against the mean of non-lapse trials, then average that
  difference across sessions. A negative difference (lapses score lower) means
  the index carries signal. We can also run a statistical test on those
  per-session differences to ask whether the gap is larger than chance.
]

= From score to prediction: how predict.py works

So far we have one number per moment: the composite engagement z-score
$overline(Z)_i$ from @eq-zbar. But a raw number is not yet a "lapse" or "not a
lapse" decision. That final step is `predict.py`.

== What predict.py does

`predict.py` turns the composite score into a binary verdict (lapse / not a
lapse) and then *evaluates how often that verdict is right*. It consumes three
inputs, all already produced by the earlier stages:

- the per-trial composite score $overline(Z)_i$ (one per trial),
- the *labels* (which trials are actually lapses), and
- the *metadata* (which subject and session each trial belongs to).

It implements two *decision rules*, both deliberately parameter-free:

1. *Rank rule (default).* Within each (subject, session), it flags the lowest
   $10%$ of composite scores as lapses. This directly mirrors how the labels
   were built (the slowest reaction-time decile is the "lapse" class), and
   ranking within a session makes the rule immune to session-level offsets.
2. *Threshold rule.* It flags every trial whose composite score falls below a
   fixed cutoff (e.g. $0$), meaning "below the person's resting-normal level."
   This is an *absolute* rule: any trial sitting that far below rest is called a
   lapse, regardless of how it ranks against its own session.

Because both rules have no learned parameters, evaluation is straightforward:
we score each subject's trials directly (leave-one-subject-out), then ask how
well the predicted lapses match the true lapses.

#quote(block: true)[
  *A concrete example of each rule.* Say a session has $90$ trials.
  *Rank rule:* we sort the $90$ composite scores lowest to highest and flag the
  bottom $9$ ($10%$) as lapses, the rest as non-lapses. This exactly matches the
  way labels were built (the slowest reaction-time decile), and because we rank
  within the session, a whole-session drift in scores does not change which
  trials get flagged.
  *Threshold rule:* we choose a cutoff, e.g. composite $< 0$. We flag every trial
  whose score is below $0$ as a lapse and every trial at or above $0$ as not. This
  is absolute --- a session that sits unusually low will have many lapses, a
  session that sits high will have few, regardless of internal ranking.
]

== How well is it doing? Starting with accuracy --- and why it's not enough

The most instinctive way to measure a classifier is *accuracy*: the fraction of
trials the model got right.

$ "Accuracy" = ("correct predictions")/("all trials") $

But here accuracy is *actively misleading*, for one reason: the classes are
extremely unbalanced. Only about $10%$ of trials are lapses; the other $90%$ are
non-lapses. Consider the trivial rule "never predict a lapse, always say
'attentive'." That rule would be *wrong on every lapse* but still correct on
$90%$ of all trials --- so it would report $90%$ accuracy while being useless at
the one thing we care about (catching lapses). A model that merely reproduces
this imbalance can look "good" by accuracy even though it never detects a single
lapse. So accuracy is a poor yardstick here; we need metrics that care about the
rare, important class.

#quote(block: true)[
  *A concrete example of why accuracy lies.* A session has $90$ trials, $9$ of
  which are true lapses. An always-"attentive" rule calls all $90$ non-lapses:
  it is right on the $81$ real non-lapses and wrong on the $9$ real lapses, so
  accuracy is $81\/90 = 90%$. A rule that actually finds, say, $5$ of the $9$
  lapses while making $4$ false alarms would have lower accuracy ($82\/90 =
  91%$? --- it gets $5 + 77 = 82$ right) yet is obviously more useful, because it
  catches lapses at all. So the $90%$ accuracy figure tells you almost nothing
  about whether lapses are being detected; the only way to tell is to look at the
  lapse class specifically.
]

== Precision, recall, and the F1 score

To fix this, we stop counting "right or wrong" and instead count *which kinds of
mistakes* we make. For the lapse class:

- *Precision* = of the trials we called lapses, how many really were lapses?
  (How many "alarms" were correct?)
- *Recall* = of the true lapses, how many did we catch? (How many real lapses
  did we find at all?)

The *F1 score* is the harmonic mean of the two:

$ "F1" = 2 dot ("precision" dot "recall")/("precision" + "recall") $

F1 is a single number between $0$ and $1$ that only reaches a high value when
*both* precision and recall are high. This is exactly what we want: a model that
cries "lapse" constantly would have high recall but terrible precision (low F1),
and a model that never says "lapse" would have high precision but zero recall
(low F1). Because F1 ignores the $90%$ majority and focuses on the rare lapse
class, it is the headline metric for this problem.

The evaluation also reports two useful reference points to interpret any F1:

- a *chance F1*: what an uninformative rule with the same prediction rate would
  score; and
- a *null distribution*: the F1 you'd get if the labels were shuffled within
  sessions while the predictions were held fixed. If the observed F1 sits far
  above this null, the score is carrying real signal rather than luck.

#quote(block: true)[
  *A concrete example of precision, recall and F1.* Suppose our rule flags $10$
  trials as lapses. Of those $10$, $6$ are true lapses ($4$ are false alarms), so
  precision $= 6\/10 = 0.60$. The session had $9$ true lapses and we caught $6$,
  so recall $= 6\/9 = 0.67$. Then F1 $= 2 dot (0.60 dot 0.67)\/(0.60 + 0.67) =
  0.63$. Now compare a rule that flags everything: recall becomes $1.0$ (we catch
  all $9$) but precision drops to $9\/90 = 0.10$, and F1 $= 2 dot (0.10 dot
  1.0)\/(0.10 + 1.0) = 0.18$ --- much worse, because it is crying "lapse" almost
  all the time. And an always-"attentive" rule has recall $0$, so F1 $= 0$. Only a
  rule that balances precision and recall well reaches a high F1.
]

== Going further: the AUROC (rank-based evaluation)

F1 is great, but it has a limitation: it depends on the exact cutoff of the
decision rule (where we draw the "lapse" line). Change the cutoff and F1
changes. It would be nicer to have a single number that measures the model's
*discrimination ability* --- how well the score separates lapses from
non-lapses --- without committing to one cutoff.

That is what the *AUROC* (Area Under the Receiver Operating Characteristic)
provides. The idea: imagine sweeping the threshold from "everything is a lapse"
to "nothing is a lapse," and at every step plotting the true-positive rate
against the false-positive rate. The area under that curve is a number between
$0$ and $1$:

- $0.5$ means the score is no better than guessing (random ordering of trials);
- $1$ means the score perfectly separates lapses from non-lapses (every lapse
  scores below every non-lapse).

Crucially, the AUROC *does not depend on any chosen cutoff* --- it only asks
"do real lapses tend to have lower composite scores than real non-lapses?" This
makes it a cleaner test of whether the engagement index carries information,
independent of how we later threshold it. One caveat: unlike accuracy or F1, the
AUROC measures *ordering* rather than a concrete decision, so it is best
reported alongside F1 rather than instead of it.

#quote(block: true)[
  *A concrete example of the AUROC idea.* Ignore any cutoff and just ask: if we
  pick one true lapse and one true non-lapse at random, how often does the lapse
  have the *lower* composite score (i.e. is correctly judged more inattentive)?
  If the index carried no signal, this would happen half the time ($0.5$). If it
  always happened, the index would perfectly separate the two classes ($1.0$).
  The AUROC is essentially this "probability of correct ordering" averaged over
  all pairs, so a value of $0.7$ means a random lapse is judged lower than a
  random non-lapse $70%$ of the time. Unlike F1, it needs no threshold --- but it
  also does not tell you how to actually make the binary call; that is why we
  report both.
]

= The thread through all of it

The thread through all of it is the same two ideas repeated at every level:

1. *Use a ratio* (and its log) so electrode quality and recording quirks cancel out.
2. *Measure everything relative to a personal baseline* so that "how attentive is this
   brain" means the same thing no matter whose brain it is, which electrode, or which
   day.

That's the entire method: take a noisy, physically-incomparable EEG, and reduce it to
a single, comparable, interpretable number --- *how many personal baseline spreads
above or below a person's own resting-normal engagement are they, right now.*

]
