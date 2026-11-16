# Article outline: a logistic regression meets a Deadlock update

The skeleton for the blog post. **You write the prose.** This file has, per section, the chart,
the numbers, the insight and the wording traps. Numbers come from `reports/tables/` (extract run
2026-11-08 16:30 UTC). Rerun the commands before publishing if the data is re-extracted.

**Focus (user's decision, 2026-11-08):**
- The 24 h before vs the 24 h after the "Halloween 2026 + Gameplay Update": how hero win rates
  differ.
- A **logistic regression** that predicts match winners from the 12 heroes, and what the update
  does to it.
- The code was trimmed to match (2026-11-08): picks/bans, Ranked vs Unranked, synergy and the
  stability analysis are gone; they're in git history if ever needed.

9 charts, in two notebooks that follow the article: `01_winrates` (Parts 1–2) and `02_model`
(Parts 3–6).

---

## Title ideas

- "What a logistic regression learns about Deadlock heroes, and what a patch does to it"
- "Predicting Deadlock matches from heroes alone, before and after the Halloween update"
- "60% from heroes alone: a logistic regression vs a Deadlock patch"

## Hook (1–2 sentences)

If you only know which 12 heroes are in a match, how well can you predict the winner? And what
happens to that prediction when the developers patch the game overnight?

---

## Part 1: The data and the moment of the update

**Chart:** `quality_patch_dip.png`

- **Source:** the public Deadlock API data dumps, 1,016,323 matches started 10-28 → 10-31 UTC.
- **Population:**
  - Normal mode, Unranked + Ranked, complete 12-player matches;
  - 799,794 matches in all;
  - what was dropped and why is in the README, under "What was discarded".
- **Insight:** the data doesn't record a game version. The update shows as a **dip in match
  starts**: ~2,000 per 10 min, down to 219 at 20:50 UTC.
- **Design:**
  - 24 h before vs 24 h after;
  - starts in the 20:30–21:30 buffer are excluded;
  - the windows were fixed before looking at any post-update result.
- **Model windows:** the model trains on the earlier matches (10-28 00:00 → 10-29 20:30), then is
  tested on the 24 h before and the 24 h after.

## Part 2: Win rates before and after (Wilson CI)

**Charts:** `winrate_ci_pre.png` (main chart), `winrate_ci_post.png`, `patch_before_after.png`

- **Wilson CI in one paragraph:** the range of plausible true win rates given the games seen.
  With ~60k games per hero, the CI is about **±0.4 pp**, so Wilson and the simple interval are
  nearly identical here. Wilson matters for small samples (e.g. at 1,000 games the CI is ±3 pp).
- **Before:** 44.2% (Pocket) → 57.7% (Graves). 42 of 44 heroes differ from 50% after Holm.
- **The differences, before → after:**
  - 20 of 44 heroes changed after Holm;
  - Violet 52.7% → 47.1% (−5.6 pp), Mo & Krill −5.5, Nurse Harrow −3.2;
  - Warden +3.4, McGinnis +3.3, Deadman Danny +2.3.
- **Proof it isn't the usual daily wiggle (two sentences):**
  - the same comparison between two ordinary days, with no update, finds 0 of 44 heroes
    changed (`winlock patch` prints it);
  - the same comparison across the Friday → Saturday one week earlier also finds 0 of 44, and
    its largest change is 0.8 pp (`winlock robustness`). So "it was just the weekend" doesn't
    explain a −5.6 pp move.
- **The card for social posts:** `patch_movers.png`, the 10 largest changes.
- **Newest heroes (ringed in the scatter):** the update also made the 6 newest heroes
  bannable (0% of matches banned them before, 1–9% after). Their changes mix the balance
  changes with who still gets to play them.
- **Holm in one sentence:** 44 tests at once give about 2 false alarms by luck, so the bar is made
  stricter.
- **Wording trap:** "games with Violet were won 47.1% of the time after the update", not "Violet
  is weak". A win rate measures the hero **and** the players who pick it.

## Part 3: The model

**No chart:** a small diagram or formula is enough.

- **What it predicts:** **who wins a match**, not a hero's win rate. The win rates show up
  indirectly, as each hero's weight.
- **Features:** one row per match, 44 columns: +1 if the hero is on Team0, −1 if on Team1, 0 if
  absent. Target: Team0 won (1) or lost (0).
- **Model:** P(Team0 wins) = σ(b + w·x). Each weight w is a hero's "strength", in log-odds.
- **Training:**
  - 361,004 matches (10-28 00:00 → 10-29 20:30);
  - L2 penalty: log loss + (1/C)·Σw²;
  - C chosen on the last 20% of train **in time**, never on test; C = 0.1 won, although every C
    gave nearly the same loss.
- **Why split by time:** a random split would let the model peek at the future.
- **The model sees no players, on purpose.**

## Part 4: Is it any good?

**Charts:** `model_accuracy.png` (the "before" rows), `model_calibration.png`

| Model (test: 24 h before the update) | Accuracy (95% Wilson CI) | Log loss |
|---|---|---|
| Logistic regression | **59.9%** (59.7–60.2%) | 0.663 |
| Summed win rates, no fitting | 59.7% | 0.664 |
| Side only (a coin flip) | 50.1% | 0.693 |

- **Wilson again:** accuracy = correct / matches is a proportion, so it gets a Wilson CI. The CI
  is far above 50%, so this isn't luck.
- **Log loss vs accuracy:** log loss also rewards being *honestly* unsure, and punishes confident
  mistakes.
- **Calibration:** when the model says 70%, Team0 wins about 70% of the time.
- **Key paragraph (keep, adapt the numbers).** "Inefficient" is the wrong word: the model is
  **deliberately limited**. It never sees who is playing, and that was the experiment's
  design. Win rates alone are enough to beat chance consistently, but not enough to call a
  single match confidently: 4 in 10 predictions are still wrong. The missing part most likely
  depends on the players, which I chose not to use. That makes the 60% a measure of how far
  hero information alone can go.

## Part 5: What the model learned

**Chart:** `model_coefficients.png`

- The weights correlate **0.95** with the heroes' win rates, so the model mostly re-learns the
  win rates.
- **The exception is McGinnis:** raw win rate 48.3%, but a **positive** weight: +0.108, with a
  95% bootstrap interval of +0.089 to +0.129. Once the model accounts for who she plays with
  and against, she looks better than her win rate says.
- **What "positive" means:** the weights sum to zero, so a weight compares a hero with the
  *average hero* (win rate 49.3%), not with 50%. McGinnis is below average by win rate and
  above average by weight.
- **Bootstrap in one sentence:** refit the model 500 times on matches drawn with replacement
  and see how much each weight moves; the middle 95% of the refits is the interval.
- **Reading a weight:** σ(w) − 0.5 is roughly the change in win chance for one hero against an
  average opponent. Use this to turn one hero's weight into pp (exercise below).
- **The summed-win-rate model** (59.7%) is the same formula with weights = logit(win rate) and no
  fitting. Fitting adds only +0.2 points.

## Part 6: The update breaks the model, and retraining fixes it

**Charts:** `model_accuracy.png` (before vs after rows), `model_retrain.png`,
`model_calibration.png` (the "after" line)

- **The same model, 24 h after the update: 58.8%** (58.6–59.1%), down about 1 point from 59.9%.
- **Why:** **distribution shift**. The weights describe the old game, and the patch changed what
  each hero is worth (Part 2).
- **Calibration after:** slightly overconfident at the extremes (it says 71%, the team wins 69%).
- **Retraining:**
  - refit the same model on the first 70% of post-update matches;
  - score both models on **the same** last 30% (from 10-31 15:40 UTC);
  - old model 58.8%, retrained **59.9%**;
  - the log-loss difference is −0.0058 (95% CI −0.0066 to −0.0049).
- **Why the same matches:** the comparison is paired, match by match, so the luck of which
  matches landed in each test set cancels out.
- **Insight:** a few hours of new matches are enough to bring the model back.

## Part 7: Limitations

- **One update, one weekend of control.** Post = Friday evening → Saturday evening. An
  ordinary Friday → Saturday a week earlier moved nothing, but that is a single weekend.
  There's also a Halloween event on the post side.
- **Bans:** the newest heroes became bannable with the update, so who plays them changed too.
- **New heroes** were still being learned.
- **Late arrivals:** the data wasn't complete when it was extracted. Extracted again 8 days
  later it has 9–10% more matches, and no hero's win rate moves by more than 0.15 pp.
- **No players in the model:** win rates and weights mix the hero with the players who pick
  them.

## Closing

Suggestion: "Heroes alone get a simple model to 60%. A patch knocks it down a point, and a few
hours of new data bring it back. The other 40% is the players."

---

## Social drafts

### LinkedIn

Image: `patch_movers.png` (1080×1350, the 4:5 size feeds show largest). Lead with the idea a
reader who never played Deadlock can use: a model describes the world it was trained on.

> A model trained on yesterday's world gets worse when the world changes. I measured by how
> much, on ~800k matches of the game Deadlock.
>
> The game got a big update on October 30. I compared the 24 h before with the 24 h after.
>
> • The update moved hero win rates by up to 5.6 points. Between two days with no update,
>   including the same Friday → Saturday a week earlier, nothing moved (0 of 44 heroes).
> • A logistic regression that sees only the 12 heroes of a match calls 59.9% of winners
>   (95% CI 59.7–60.2%). A coin flip gets 50%.
> • The same model after the update: 58.8%. Refit on a few hours of new matches: back to
>   59.9%.
>
> The honest part: the model never sees the players, on purpose. 60% is how far heroes alone
> go.
>
> What I practised: confidence intervals, multiple-comparison correction, placebo tests, and
> splitting data by time so the model never sees the future.
>
> Code and charts: [GitHub link]
> Data: Deadlock API (deadlock-api.com). Not affiliated with Valve.
>
> #DataScience #MachineLearning #Python #Statistics

### Twitter / X thread

1. Can you predict a Deadlock match from the 12 heroes alone? A logistic regression, ~360k
   matches, and a patch in the middle. 🧵
2. Before the Halloween update: 59.9% correct (coin flip: 50%). [model_accuracy.png]
3. The update moved win rates (Violet −5.6 pp, Warden +3.4), and the model dropped to 58.8%.
4. Retrained on a few hours of new matches: back to 59.9%. [model_retrain.png]
5. Heroes get you to 60%. The rest is the players. Write-up + code: [link]

---

## Pen-and-paper exercises for this article

1. **Wilson:** 11 wins in 20 games. Compute the simple and the Wilson 95% interval.
2. **CI width:** a hero with ~60,000 games at ~50%. Check that the 95% half-width is about
   0.4 pp.
3. **Sigmoid:** hero A (w = +0.2) on Team0 against hero B (w = −0.1) on Team1, b = 0. Compute
   P(Team0 wins).
4. **Accuracy CI:** the model got 121,726 of 203,085 matches right. Compute the accuracy and its
   95% half-width (the simple formula is fine at this n).
5. **Holm:** p-values 0.004, 0.009, 0.011, 0.03, 0.30, α = 0.05. How many survive?
6. **Paired comparison:** why compare the old and retrained models on the *same* matches?
7. **Bootstrap:** five matches with outcomes 1, 1, 0, 1, 0 (win share 60%). Draw 5 with
   replacement three times by hand (roll a die, reroll a 6) and compute each share. Why do the
   shares differ, and what would the middle 95% of 500 such shares tell you?
8. **Sum to zero:** three heroes have weights +0.2, −0.1 and w. If the weights must sum to
   zero, what is w? What does a weight of 0 then mean?
