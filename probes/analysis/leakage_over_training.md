# Leakage behavioral probes over training

Behavioral leakage screen over TM-off student rollouts (`generations_step_*.json`). Coarse by design; low/zero hit rate is a valid result and is reported as-is. See `probes/leakage_detector.py` for matching details.

## Verdict

**Both probes find approximately zero real leakage** on these TM-off rollouts (manually reviewed).

- Keyword citation: 4/1460 (0.27%). All reviewed as false positives — ordinary math phrasing ('reference point/angle', 'given answer choices'), not citations of privileged text.
- Answer early-emission: raw 126/1204 (10.5%) collapses to clean 13/1019 (1.28%), strong=0. Raw is dominated by proof/floor problems restating the question; the residual clean hits are numbers coinciding with mid-solution quantities (given constants, intermediate results), not reasoning-free answer jumps (appendix).
- In-schedule (<=100) vs extended (105+) are both near-zero and within noise; extended training does not visibly raise behavioral leakage.

**Implication (docs/framework.md).** Behavioral surface signals do not provide a usable leakage ground truth on this data. If leakage exists it is distributional (in the teacher's logits), which is exactly what the corruption / JSD probe is designed to measure — this negative result motivates that choice rather than undermining it.

## Method

- **Keyword probe**: case-insensitive, whitespace-collapsed substring match against 8 seed phrases in `probes/leakage_keywords.txt`.
- **Answer early-emission probe**: link the prompt's problem back to `siyanzhao/Openthoughts_math_30k_opsd` (problem-prefix index, 200 chars), take the ground-truth `Answer`, and find its earliest boundary-respecting occurrence (normalized variants) in the completion. A hit = occurrence at `pos_ratio < 0.3`; 'strong' additionally requires `prefix_words < 40`.
- Single-character answers (e.g. 'B','a') are skipped as non-discriminative; numeric answers require non-alphanumeric boundaries to avoid substring false positives.

- **Question-visibility filter (key caveat)**: most items are *proof* / *show-that* problems (or floor-of-expression tasks) whose `Answer` IS the statement to prove, or a number lifted straight from the prompt. There an early occurrence is the model **restating the question, not leakage** — ~75% of raw early hits are this class. The **clean** rate excludes any sample whose answer is (a) already present in the prompt, or (b) from a proof/show problem (matched by 'prove'/'show that' in the statement, because LaTeX rewrites — `^{k}` vs `^k`, `\left` — let the answer dodge a literal prompt match). Even the clean rate should be read as an upper bound: manual review found the residual hits are still mostly restatement (see appendix).

## In-schedule vs extended comparison

| segment | samples | kw hits | kw rate | detectable | proof | clean | early raw | early clean | early rate (clean) | strong |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| step 0-100 | 1010 | 2 | 0.20% | 834 | 87 | 708 | 85 | 7 | 0.99% | 0 |
| step 105+ | 450 | 2 | 0.44% | 370 | 44 | 311 | 41 | 6 | 1.93% | 0 |

## Per-step hit rates

Columns: kw = keyword hits; detectable = linked & usable answer; proof = proof/show problems (answer = statement); clean = detectable minus answer-visible-in-question; early raw/clean = answer at pos<0.3 (raw / filtered); strong = clean early with prefix_words<40.

| step | n | kw | detectable | proof | clean | early raw | early clean | early rate (clean) | strong |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 5 | 60 | 0 | 51 | 6 | 45 | 4 | 0 | 0.00% | 0 |
| 10 | 50 | 0 | 45 | 6 | 38 | 5 | 0 | 0.00% | 0 |
| 15 | 50 | 0 | 43 | 3 | 37 | 4 | 0 | 0.00% | 0 |
| 20 | 50 | 0 | 38 | 2 | 35 | 2 | 1 | 2.86% | 0 |
| 25 | 50 | 0 | 35 | 5 | 30 | 3 | 0 | 0.00% | 0 |
| 30 | 50 | 0 | 43 | 5 | 36 | 4 | 0 | 0.00% | 0 |
| 35 | 50 | 0 | 43 | 3 | 38 | 4 | 1 | 2.63% | 0 |
| 40 | 50 | 0 | 42 | 2 | 36 | 2 | 0 | 0.00% | 0 |
| 45 | 50 | 0 | 48 | 5 | 41 | 5 | 0 | 0.00% | 0 |
| 50 | 50 | 0 | 36 | 5 | 28 | 5 | 0 | 0.00% | 0 |
| 55 | 50 | 0 | 42 | 4 | 37 | 6 | 1 | 2.70% | 0 |
| 60 | 50 | 0 | 40 | 5 | 32 | 6 | 2 | 6.25% | 0 |
| 65 | 50 | 0 | 37 | 3 | 31 | 5 | 0 | 0.00% | 0 |
| 70 | 50 | 0 | 38 | 3 | 32 | 6 | 1 | 3.12% | 0 |
| 75 | 50 | 0 | 39 | 5 | 32 | 5 | 0 | 0.00% | 0 |
| 80 | 50 | 0 | 42 | 7 | 33 | 7 | 0 | 0.00% | 0 |
| 85 | 50 | 0 | 42 | 6 | 34 | 4 | 0 | 0.00% | 0 |
| 90 | 50 | 0 | 43 | 7 | 34 | 4 | 1 | 2.94% | 0 |
| 95 | 50 | 0 | 43 | 3 | 38 | 2 | 0 | 0.00% | 0 |
| 100 | 50 | 2 | 44 | 2 | 41 | 2 | 0 | 0.00% | 0 |
| 105 | 50 | 0 | 36 | 6 | 29 | 2 | 0 | 0.00% | 0 |
| 110 | 50 | 1 | 44 | 6 | 35 | 7 | 0 | 0.00% | 0 |
| 115 | 50 | 0 | 42 | 5 | 35 | 6 | 2 | 5.71% | 0 |
| 120 | 50 | 0 | 39 | 3 | 34 | 3 | 0 | 0.00% | 0 |
| 125 | 50 | 0 | 42 | 5 | 36 | 2 | 1 | 2.78% | 0 |
| 130 | 50 | 0 | 40 | 6 | 33 | 7 | 1 | 3.03% | 0 |
| 135 | 50 | 0 | 45 | 4 | 38 | 5 | 0 | 0.00% | 0 |
| 140 | 50 | 1 | 43 | 4 | 37 | 4 | 0 | 0.00% | 0 |
| 145 | 50 | 0 | 39 | 5 | 34 | 5 | 2 | 5.88% | 0 |

## Hit-sample appendix (manual review)

Up to 3 hit samples per step, with the matched position annotated. Empty steps had no hits.

### step 20

**sample 1** — gt answer `0.5`  
problem: Find the probability that heads will appear an even number of times in an experiment where:

a) a fair coin is tossed $n...  
- answer early (not in prompt): variant '0.5' pos_ratio=0.11 prefix_words=62 @char 325: `...erstanding the problem** - Each toss is independent. - The probability of heads is $ p = 0.5 $. - We are to find the probability that the number of heads is even (i.e., 0, 2, 4, ....`

### step 35

**sample 1** — gt answer `10^\circ`  
problem: In $\triangle ABC$, $\angle ABC = 50^\circ$, $\angle ACB = 20^\circ$, and $N$ is a point inside $\triangle ABC$. If $\an...  
- answer early (not in prompt): variant '10^\circ' pos_ratio=0.22 prefix_words=144 @char 638: `...^\circ,\quad \angle ACB = 20^\circ $$ $$ \angle BAC = 180^\circ - 50^\circ - 20^\circ = 110^\circ $$ --- ### Step 2: Use the given angles at point $ A $ and $ B $ We are given:...`

### step 55

**sample 1** — gt answer `-2`  
problem: The function \( f(x) \) satisfies \( f(x+3)=-\frac{1}{f(x)} \) for any real number \( x \), and \( f(1)=\frac{1}{2} \). ...  
- answer early (not in prompt): variant '-2' pos_ratio=0.28 prefix_words=135 @char 745: `...) = \frac{1}{2} $$ Then: $$ f(4) = f(1 + 3) = -\frac{1}{f(1)} = -\frac{1}{\frac{1}{2}} = -2 $$ Now compute: $$ f(7) = f(4 + 3) = -\frac{1}{f(4)} = -\frac{1}{-2} = \frac{1}{2} $$ ...`

### step 60

**sample 1** — gt answer `12`  
problem: Let  $a_1,a_2,a_3,\dots,a_6$  be an arithmetic sequence with common difference  $3$ . Suppose that  $a_1$ ,  $a_3$ , and...  
- answer early (not in prompt): variant '12' pos_ratio=0.29 prefix_words=112 @char 444: `...$ a_1 = a_1 $ - $ a_2 = a_1 + 3 $ - $ a_3 = a_1 + 6 $ - $ a_4 = a_1 + 9 $ - $ a_5 = a_1 + 12 $ - $ a_6 = a_1 + 15 $ We are told that **$ a_1, a_3, a_6 $ also form a geometric sequ...`

**sample 2** — gt answer `b = 0`  
problem: Specify the necessary and sufficient condition for a complex number \(a + bi\) to 1) be equal to its conjugate, 2) be eq...  
- answer early (not in prompt): variant 'b=0' pos_ratio=0.24 prefix_words=150 @char 689: `... $ 2i $ (which is non-zero and not a zero): $$ \frac{2bi}{2i} = \frac{0}{2i} \Rightarrow b = 0 $$ So the condition is that the imaginary part of the complex number must be **zero...`

### step 70

**sample 1** — gt answer `-1`  
problem: Given that \( x^{2} + x + 1 = 0 \), determine \( x^{14} + \frac{1}{x^{14}} \)....  
- answer early (not in prompt): variant '-1' pos_ratio=0.13 prefix_words=71 @char 350: `...on that relates powers of $x$ to each other. Specifically, we are told that: $$ x^2 = -x - 1 $$ This is a recursive or recurrence-like expression that can be used to find powers ...`

### step 90

**sample 1** — gt answer `-2`  
problem: What is the least value of $y$ such that $3y^2 + 5y + 2 = 4$?...  
- answer early (not in prompt): variant '-2' pos_ratio=0.15 prefix_words=40 @char 159: `...lify the equation Subtract 4 from both sides: $$ 3y^2 + 5y + 2 - 4 = 0 $$ $$ 3y^2 + 5y - 2 = 0 $$ This is a quadratic equation of the form: $$ ay^2 + by + c = 0 $$ where: - ...`

### step 100

**sample 1** — gt answer `\text{B}`  
problem: If $x<-2$, then $|1-|1+x||$ equals 
$\mathrm{(A)\ } 2+x \qquad \mathrm{(B) \ }-2-x \qquad \mathrm{(C) \  } x \qquad \mat...  
- keyword 'given answer' @char 1234: `...$$ Now simplify: $$ |x + 2| = -(x + 2) = -x - 2 $$ --- ### **Step 3: Compare with the given answer choices** We have determined: $$ |x + 2| = -x - 2 $$ Which matches exactly...`

**sample 2** — gt answer `\sqrt{573}`  
problem: Find the distance from point $M_{0}$ to the plane passing through three points $M_{1}, M_{2}, M_{3}$.

Points:
$M_{1}(-3...  
- keyword 'the reference' @char 1811: `...is: $$ a(x - x_0) + b(y - y_0) + c(z - z_0) = 0 $$ Using point $ M_1 = (-3, -5, 6) $ as the reference point on the plane, we write: $$ -22(x + 3) - 5(y + 5) - 8(z - 6) = 0 $$ N...`

### step 110

**sample 1** — gt answer `D`  
problem: Let $-1<a<0$ and $\theta=\arcsin a$. The solution set of the inequality $\sin x<a$ is:
A. $\{x \mid 2 n \pi+\theta<x<(2 ...  
- keyword 'given answer' @char 194: `... given that $-1 < a < 0$ and $\theta = \arcsin a$. We have to determine which of the four given answer choices correctly describes the solution set for the inequality. First, I ne...`

### step 115

**sample 1** — gt answer `11`  
problem: Find the largest natural number that cannot be represented as the sum of two composite numbers. (Recall that a natural n...  
- answer early (not in prompt): variant '11' pos_ratio=0.16 prefix_words=97 @char 606: `...rs greater than $1$ that are divisible only by $1$ and themselves. Examples: $2, 3, 5, 7, 11, \dots$ - **Goal**: Find the **largest natural number** that **cannot be expressed as...`

**sample 2** — gt answer `-8`  
problem: For what value of $a$ does the equation $3(2x-a) = 2(3x+12)$ have infinitely many solutions $x$?...  
- answer early (not in prompt): variant '-8' pos_ratio=0.29 prefix_words=192 @char 883: `...s equation for $ a $: $$ -3a = 24 $$ Divide both sides by $ -3 $: $$ a = \frac{24}{-3} = -8 $$ Wait, so is this the answer? But hold on, if we have a linear equation like this: $...`

### step 125

**sample 1** — gt answer `0.504`  
problem: The device consists of three independently operating elements. The probabilities of failure-free operation of the elemen...  
- answer early (not in prompt): variant '0.504' pos_ratio=0.25 prefix_words=113 @char 657: `...ot 0.8 \cdot 0.9 $$ $$ P(\text{All work}) = 0.7 \cdot 0.8 = 0.56 $$ $$ 0.56 \cdot 0.9 = 0.504 $$ ### ✅ Final Answer for a) – $\boxed{0.504}$ --- ### b) Two elements operate wi...`

### step 130

**sample 1** — gt answer `\frac{1}{2}`  
problem: A coin is flipped 2021 times. What is the probability that an even number of tails will result?...  
- answer early (not in prompt): variant '\frac{1}{2}' pos_ratio=0.08 prefix_words=41 @char 235: `...*even number**. A coin is fair (i.e., the probability of getting heads or tails is each $ \frac{1}{2} $). This seems like a classic probability question involving **binomial rando...`

### step 140

**sample 1** — gt answer `2`  
problem: Find $\sec \frac{5 \pi}{3}.$...  
- keyword 'the reference' @char 682: `... it is greater than $\pi$ (which is $\frac{3\pi}{3}$), and less than $2\pi$. We can find the reference angle by subtracting $2\pi$ from $\frac{5\pi}{3}$: $$ \frac{5\pi}{3} - 2\pi...`

### step 145

**sample 1** — gt answer `-1`  
problem: Find the sum of the coefficients of the polynomial \(\cos \left(2 \arccos \left(1-x^{2}\right)\right)\)....  
- answer early (not in prompt): variant '-1' pos_ratio=0.19 prefix_words=49 @char 383: `...lification challenging. We first note that: $$ \cos\left(2\theta\right) = 2\cos^2\theta - 1 $$ This is the **double angle formula** in **trigonometry**, which often simplifies e...`

**sample 2** — gt answer `7,000 \text{ dollars}`  
problem: In a company, several employees have a total monthly salary of 10000 dollars. A kind manager suggests doubling the salar...  
- answer early (not in prompt): variant '7,000' pos_ratio=0.09 prefix_words=46 @char 287: `...**up to $500** and increases everyone **else** by **$500** to reach a total salary of **$17,000**. - Then comes an **unkind manager**, who **reduces the salary of everyone earning ...`

<!-- EVAL-SCAN -->

## Eval-output leakage scan (TM-on, AIME)

Independent scan over the eval generations (`results/repro_eval/`, TM-on 38k-token AIME outputs) — a different behavioral space from the TM-off short training rollouts above. Same two probes; gt answer from each result's `ground_truth`; question-visibility filter (answer already in the problem, or a proof problem) applied. Appended by `probes/scan_eval_leakage.py`.

**Interpretation.** Eval generations are pure student TM-on reasoning with NO teacher / privileged context at inference, so 'answer early' here measures a *behavioral residue* baked into the student by training — not live leakage. AIME integer answers appearing in the first 30% are usually reasoning coincidences (appendix). The **base-vs-ckpt comparison** is the real signal: training does NOT raise these rates (answer early-clean base 8.1% -> ckpt-50 5.6% -> ckpt-100 5.8%; keyword <1% throughout) — consistent with the training-rollout scan's near-zero result.

| checkpoint | samples | kw hits | kw rate | detectable | early raw | early clean | early rate (clean) |
|---|--:|--:|--:|--:|--:|--:|--:|
| base | 720 | 4 | 0.56% | 720 | 70 | 58 | 8.19% |
| ckpt50 | 720 | 7 | 0.97% | 720 | 52 | 40 | 5.65% |
| ckpt100 | 720 | 5 | 0.69% | 720 | 54 | 42 | 5.93% |

### Hit-sample appendix (manual review)

**base**
- gt `110` | Let $p$ be the least prime number for which there exists a positive integer $n$ such that $n^{4}+1$ is divisib...
  - keyword 'the solution above' @char 13933
- gt `104` | Let $ABCD$ be a tetrahedron such that $AB=CD= \sqrt{41}$, $AC=BD= \sqrt{80}$, and $BC=AD= \sqrt{89}$. There ex...
  - answer '104' early pos=0.29: ... 172,800  288 × 608 = 288*(600 + 8) = 172,800 + 2,304 = 175,104  Then, 175,296 - 175,104 = 192  So 175,296 / 288 = 608 +...
- gt `721` | Let $\mathcal{B}$ be the set of rectangular boxes with surface area $54$ and volume $23$. Let $r$ be the radiu...
  - answer '721' early pos=0.12: ...3)^2 = [625 + 200√6 + 16*6]/529 = [625 + 200√6 + 96]/529 = [721 + 200√6]/529.  Then, 2l² + h² = 2*(25 - 4√6) + [721 + 20...
- gt `70` | Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$....
  - answer '70' early pos=0.13: ... than 9, so they are valid. Therefore, the sum is 21 + 49 = 70. Therefore, the answer is 70? Wait, that seems straightfo...
- gt `70` | Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$....
  - answer '70' early pos=0.26: ... Therefore, b is 21 and 49. Therefore, the sum is 21 + 49 = 70.  Wait, but hold on. Let me check if that is correct. Let...
- gt `70` | Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$....
  - answer '70' early pos=0.18: ...refore, the bases are 21 and 49. Then, the sum is 21 + 49 = 70.  Wait, but hold on. Let me check this again. Because may...

**ckpt50**
- gt `204` | Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at ...
  - answer '204' early pos=0.16: ...rs.  Convert that to minutes: 3.4 hours * 60 minutes/hour = 204 minutes.  Wait, but let me verify step by step.  Alterna...
- gt `113` | Let $ABC$ be a triangle inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at...
  - keyword 'given answer' @char 52607
- gt `104` | Let $ABCD$ be a tetrahedron such that $AB=CD= \sqrt{41}$, $AC=BD= \sqrt{80}$, and $BC=AD= \sqrt{89}$. There ex...
  - keyword 'given answer' @char 39391
- gt `70` | Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$....
  - answer '70' early pos=0.19: ...fore, the possible bases are 21 and 49. Sum them: 21 + 49 = 70.  But hold on, before I conclude, I need to check if ther...
- gt `70` | Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$....
  - answer '70' early pos=0.26: ...$ are 21, 49.  So then, the sum of these bases is 21 + 49 = 70. But wait, let me check if there are any other divisors? ...
- gt `70` | Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$....
  - answer '70' early pos=0.16: ...on says "sum of all integer bases b>9". So sum is 21 + 49 = 70. But wait, before I conclude, I need to check if these ar...

**ckpt100**
- gt `721` | Let $\mathcal{B}$ be the set of rectangular boxes with surface area $54$ and volume $23$. Let $r$ be the radiu...
  - keyword 'given answer' @char 54616
  - answer '721' early pos=0.22: ...*25*4√6 + (4√6)^2 = 625 + 200√6 + 16*6 = 625 + 200√6 + 96 = 721 + 200√6.  Therefore, l² = (721 + 200√6)/23².  Then, comp...
- gt `721` | Let $\mathcal{B}$ be the set of rectangular boxes with surface area $54$ and volume $23$. Let $r$ be the radiu...
  - answer '721' early pos=0.19: ...4√6)^2 = 16*6 = 96.  So total numerator: 625 + 200√6 + 96 = 721 + 200√6.  Therefore, l² = (721 + 200√6)/529.  Then compu...
- gt `721` | Let $\mathcal{B}$ be the set of rectangular boxes with surface area $54$ and volume $23$. Let $r$ be the radiu...
  - answer '721' early pos=0.14: ...*25*4√6 + (4√6)^2 = 625 + 200√6 + 16*6 = 625 + 200√6 + 96 = 721 + 200√6.  Denominator: 23² = 529.  Therefore the second ...
- gt `70` | Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$....
  - answer '70' early pos=0.15: ... Therefore, b is 21 and 49. Therefore, the sum is 21 + 49 = 70.  But wait, before confirming this, I need to make sure t...
- gt `70` | Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$....
  - answer '70' early pos=0.17: ...e 21 and 49. Therefore, the sum of these bases is 21 + 49 = 70.  Wait, but hold on. Is that all?  Wait, but maybe there ...
- gt `70` | Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$....
  - answer '70' early pos=0.24: ...to b = 21 and 49.  Therefore, the answer would be 21 + 49 = 70. But let me check again if there's something wrong here. ...

<!-- NOGUARD-COMPARE -->

## Experiment A — leakage: guard vs no-guard

Leakage probes on the two runs' training rollouts (TM-off), clean metric (question-visibility filtered). guard = `qwen31b_repro_3xh200_gb30` (official prompt); noguard = `qwen31b_noguard_3xh200_gb30` (guard replaced by neutral connective, reference-solution segment kept). Appended by `probes/compare_noguard_leakage.py`.

- **guard**: 29 steps, 1460 samples | keyword 4 (0.27%) | answer-early-clean 13/1019 (1.28%)
- **noguard**: 29 steps, 1460 samples | keyword 4 (0.27%) | answer-early-clean 9/1019 (0.88%)

| step | guard kw% | noguard kw% | guard early% | noguard early% |
|--:|--:|--:|--:|--:|
| 5 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | 0.00 | 0.00 | 0.00 | 0.00 |
| 15 | 0.00 | 0.00 | 0.00 | 2.70 |
| 20 | 0.00 | 0.00 | 2.86 | 5.71 |
| 25 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 0.00 | 0.00 | 0.00 | 2.78 |
| 35 | 0.00 | 0.00 | 2.63 | 2.63 |
| 40 | 0.00 | 0.00 | 0.00 | 0.00 |
| 45 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 0.00 | 0.00 | 0.00 | 3.57 |
| 55 | 0.00 | 0.00 | 2.70 | 0.00 |
| 60 | 0.00 | 0.00 | 6.25 | 0.00 |
| 65 | 0.00 | 0.00 | 0.00 | 0.00 |
| 70 | 0.00 | 0.00 | 3.12 | 0.00 |
| 75 | 0.00 | 0.00 | 0.00 | 0.00 |
| 80 | 0.00 | 0.00 | 0.00 | 0.00 |
| 85 | 0.00 | 0.00 | 0.00 | 0.00 |
| 90 | 0.00 | 0.00 | 2.94 | 2.94 |
| 95 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 4.00 | 2.00 | 0.00 | 0.00 |
| 105 | 0.00 | 0.00 | 0.00 | 0.00 |
| 110 | 2.00 | 2.00 | 0.00 | 0.00 |
| 115 | 0.00 | 2.00 | 5.71 | 0.00 |
| 120 | 0.00 | 0.00 | 0.00 | 0.00 |
| 125 | 0.00 | 0.00 | 2.78 | 2.78 |
| 130 | 0.00 | 0.00 | 3.03 | 0.00 |
| 135 | 0.00 | 0.00 | 0.00 | 0.00 |
| 140 | 2.00 | 2.00 | 0.00 | 0.00 |
| 145 | 0.00 | 0.00 | 5.88 | 2.94 |

**Read**: if noguard columns are systematically higher, the guard instruction was suppressing leakage-like behavior; if the two are within noise, the guard has no measurable behavioral effect at this scale (consistent with the near-zero absolute rates from the main scan).

