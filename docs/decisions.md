# Decision log

One entry per decision. Format: context → decision → consequence. Newest first.

---

## 2026-11-08: The article focuses on the logistic regression

**Context.** The rebuild produced many analyses: win rates, patch, placebo, picks/bans, Ranked vs
Unranked, model, synergy. That's too much for one undergraduate article.

**Decision.** (user's choice)
- The article covers win rates before vs after the update (Wilson CI, Holm) and the **logistic
  regression**: how well it predicts winners, what it learned, how the update breaks it, and
  retraining. The pairs model is a short extension.
- Picks/bans and Ranked vs Unranked are left out of the article and notebooks. The code and
  commands stay.
- The placebo stays as one sentence: it shows the before/after changes aren't the usual daily
  variation.
- No new models, and no uncertainty bands on the weights.
- **The code is trimmed to match** (user's choice, so the repo stays small enough to own):
  - removed: `picks.py`, `synergy.py`, the `evaluate`/`picks`/`synergy` commands, the
    coverage simulation and split-half control, 9 chart functions, their tests and tables;
  - kept: 8 charts (the update dip, win rates before/after, the before/after scatter, model
    accuracy, coefficients, retraining, calibration) in two notebooks, `01_winrates` and
    `02_model`;
  - the placebo is still computed and printed by `winlock patch`, but has no chart.
  - Everything removed is in git history.

**Consequence.** `docs/article-outline.md` is restructured around the regression (7 parts, with
the teammate-pairs extension dropped). The numbers didn't change after the trim.

---

## 2026-11-08: One story about the update; the 2026-10-20 analysis is removed

**Context.** The 2026-10-20 file turned out to be a residual fragment: ~42k Normal matches,
where a full day has ~200k. Its patch was unknown. The patch window (10-28 → 10-31) is
complete enough, has a known intervention and supports every earlier analysis with 5–9× more
data. Two parallel analyses would make the article harder to follow.

**Decision.** (user's choices)
- **Removed:** the 10-20 tables, figures, config (`start_day`, `[split]`, the data path),
  README results and article numbers. The earlier entries below stay as history. The raw
  10-20 parquet is left on disk but nothing reads it.
- **Named windows** in `[windows]` replace `start_day` and the 70/30 split. The dates are the
  ones fixed earlier today, before any post-patch rate was computed:
  - `train`: 10-28 00:00 → 10-29 20:30;
  - `placebo_pre`: 10-28 20:30 → 10-29 20:30;
  - `pre`: 10-29 20:30 → 10-30 20:30;
  - `post`: 10-30 21:30 → 10-31 21:30.

  `[comparisons]` names the windows each comparison uses. Windows overlap, so analyses select
  them by `start_time` instead of a per-row label.
- **ML:** fit on `train` (C chosen on its last 20% in time), then test on `pre` (same game)
  and on `post` (new game).
  - **Retrain check:** refit on the first 70% of `post` and score the old and the refit model
    on the **same** last 30%, compared match by match with the paired log-loss difference.
- **Evaluation and synergy** use train → test = `train` → `pre`, entirely in the old game. The
  split-half control runs inside `pre`. The synergy volcano chart is dropped.
- **Picks and bans:**
  - pick rate = matches the hero was picked in / matches;
  - ban rate = matches the hero was banned in / matches, where a match's bans are the union
    over its rows;
  - both get Wilson CIs, and the before/after change gets z with Holm/BH, like the win rates.
- **Ranked vs Unranked:** the patch and placebo comparisons are repeated inside each mode.

**Consequence.** Each window holds ~200–360k matches, so most per-hero CIs are about ±0.4 pp
wide. Almost every hero now "differs from 50%", which makes effect sizes and the placebo the
real evidence, not p-values.
- The `pre` test window is used by the evaluation, the models and synergy, so treat it as a
  reused test set.
- `post` is used once by the old model and split 70/30 for the retrain check.

---

## 2026-11-08: Patch comparison windows, buffer and placebo (fixed before any post-patch rate)

**Context.** `winlock extract` (run 2026-11-08 16:30 UTC) gave 1,016,323 matches started
10-28..10-31 UTC. Normal match starts drop from ~2,000 to 218 per 10 min around the update
(10-30 20:51:49 UTC) and recover by 21:30. `game_mode_version` is 2 on both sides, so it is not
a patch marker; the cut has to be by `start_time`. Duplicates in the dumps are identical copies
(no row was re-sent with a different `created_at`), so `first_created_at` = arrival time.

**Decision.** (in `[comparisons]` of `config/analysis.toml`, chosen by the user)
- **Buffer:** matches started 10-30 20:30–21:30 UTC belong to neither side.
- **Patch:** before = 10-29 20:30 → 10-30 20:30, after = 10-30 21:30 → 10-31 21:30 (24 h each,
  so both sides cover one full day cycle).
- **Placebo:** the same comparison at a fake cut 24 h earlier (10-28 20:30 → 10-29 20:30 vs
  10-29 20:30 → 10-30 20:30), entirely before the update: what the method finds when nothing
  changed.
- Same population rules as the 10-20 analysis (Normal, Unranked + Ranked, TeamWin, complete
  12-player/6-win matches). Every hero is tested, Holm and BH across heroes.
- Matches flagged `new_player_pool`, `low_pri_pool` or `not_scored` (3.9%) are **kept**
  (`exclude_flags = []`), so the population matches Part 1 (user's choice).
- The extracted run is used as is; a later re-extraction checks whether late arrivals change
  the result.

**Consequence.** Complete matches: patch before 203,085, after 229,335; placebo before 199,607,
after 203,085 (the placebo's "after" is the patch's "before"). Limitations: the post side is
Friday evening → Saturday evening, the pre side Thursday → Friday, so a weekday/weekend player
mix difference is not separated from the patch (the placebo is Wednesday → Thursday vs
Thursday → Friday, so it does not cover a weekend either); the Halloween event runs on the post
side only; ~22–29% of each day's matches arrived more than 3 days late, and rows were still
arriving at extraction time.

---

## 2026-10-31: Deadlock-style chart theme with hero portraits

**Context.** The charts used a plain light reference style. For the post we wanted them to look
like the game.

**Decision.**
- Colors and fonts live in `theme.py`, not in `figures.py`; `[figures] theme` in
  `config/analysis.toml` picks `"deadlock"` (default) or `"clean"` (the old look).
- Deadlock theme: dark smoke background, the game's own UI colors from the deadlock-api
  `/v1/assets/colors` list (Amber `team1_color` #d4860b, Sapphire `team2_color` #4d75c3, cream
  `base_text`, brass `golden_*`) and an art-deco double frame.
- On the win-rate chart, Amber = above 50% and Sapphire = below 50% (Holm). This is a color
  convention only: it has nothing to do with which team the hero was on.
- Fonts: free OFL look-alikes bundled in `src/winlock/assets/fonts/` (Cinzel Decorative,
  Josefin Sans, with their licenses). The game's own fonts (Radiance, Reaver) are commercial and
  are not redistributed.
- Hero portraits: `winlock portraits` downloads them once from `api.deadlock-api.com` into
  `data/reference/portraits/` (Valve art, git-ignored). Charts work without them.

**Consequence.** The charts no longer depend on the theme in their logic, and both looks are
tested. The portraits need one network step on a new machine.

---

## 2026-10-31: Reframe around the Halloween 2026 update; extract the patch window

**Context.** The 2026-10-20 analysis answered "is a single day's win rate signal or noise?". On
2026-10-30 20:51 UTC Valve published "Halloween 2026 + Gameplay Update": 35 heroes changed,
~50 items, mid-boss and comeback mechanics, the 6 newest heroes became bannable, plus a
Halloween event. A known intervention gives a stronger test: do win rates move when the game
changes, more than they move between two ordinary days?

**Decision.**
- Analyse **every hero** (no directional predictions per hero), with Holm/BH, before vs after.
- Windows (to be fixed in the population config before looking at post-patch rates): placebo
  10-28 20:51 → 10-29 20:51, pre 10-29 20:51 → 10-30 20:51, post 10-30 20:51 → 10-31 20:51.
- Data from the public dumps with `winlock extract`: whole UTC days 10-28..10-31 (a superset),
  only 18 columns, deduplicated per (match_id, player_slot) keeping the newest created_at
  (anonymous players share account_id 0, so account_id can't be the key). First run on
  2026-10-31 ~21:00 UTC; a second run ~2026-11-02 for late arrivals, compared via meta.json.
- 2026-10-20 stays as the development dataset where the methods were built.

**Consequence.** The update bundles many changes and an event on a Friday→Saturday boundary:
the analysis can say "the update window changed X", not "nerf Y caused X".

---

## 2026-10-31: Holm on the headline, split-half control, code structure

**Context.** The baseline chart colored heroes by "own 95% CI excludes 50%": 43 uncorrected
tests read at once, the same multiple-comparison problem handled everywhere else. The temporal
split also mixes "time passed" with "different players online".

**Decision.**
- Per hero, an exact binomial test against 50% with Holm (main) and BH; the chart colors heroes by
  the Holm result. Headline: 32 of 43 heroes differ from 50% (36 uncorrected).
- Split-half control: the same train/test comparison between even and odd `match_id`s, which
  cover the same hours, so any difference is chance alone. It uses the whole population
  (including test), for checking noise only, not for evaluating a model.
- Code: analysis steps moved from `cli.py` to `evaluation.py` and `synergy.py`; `cli.py` only
  loads, calls, prints and saves. One log-loss function (`stats.log_loss`). `quality.py` checks
  are now a command (`winlock quality`). Refactor checked: every number in `reports/tables/` is
  unchanged.

**Consequence.** Split-half SD of z = 0.98: the binomial SE is honest on this day. The train →
test SD of z (0.76) is lower than chance, and that is not explained.

---

## 2026-10-31: Teammate synergy, with the test set reused

**Context.** Hero win rates carry signal (60% test accuracy). The next question: do some
teammate pairs win more than the two heroes' individual strength predicts? Only one file is
available, so no fresh data exists for a new evaluation.

**Decision.**
- Teammate pairs only (946 possible), **no minimum** of matches per pair
  (`synergy.min_pair_matches = 1`): every pair seen in train is tested, and Holm/BH correct for
  all of them.
- Synergy = observed wins together − the wins the additive win-rate model expected, per pair. The
  z-score uses the Poisson-binomial variance Σp(1−p). Candidates are found on train, then checked
  for **replication** on test (correlation of the pair z-scores between the two halves).
- Model: logistic regression with hero + pair features (sparse), with C chosen on the validation
  slice, compared with the heroes-only model on test through the paired per-match log-loss
  difference.
- **The test set is reused**, which was already used on 2026-10-31 for the first models. Nothing was
  tuned on it, but the analysis was designed after seeing those results, so the test is no longer
  fully "unseen". This is recorded as a limitation for the article.

**Consequence.** Results are exploratory: the replication correlation and the model comparison
are the evidence, and individual pairs are only candidates. A second day of data would be needed
to confirm any specific pair.

---

## 2026-10-31: Evaluation on the test set and a first match-outcome model

**Context.** The train baseline (Wilson CI per hero) existed. The question is whether those rates
hold on matches the analysis never saw. The test set (the last 30% of matches in time) is used
here for the first time.

**Decision.**
- *Do the rates hold?* For each hero:
  - z = (test − train) / sqrt(SE_train² + SE_test²), corrected for 43 tests with **Holm** (main)
    and BH (reported alongside);
  - **coverage** = share of heroes whose test rate falls inside their train 95% CI.
  The benchmark for coverage is not 95%: even with no change, test has its own noise. So it is
  compared with the formula 2Φ(z·SE_a/sqrt(SE_a² + SE_b²)) − 1 and with a binomial simulation
  (`evaluation.n_sims`, `stats.seed`).
- *Do the rates carry signal?* Three models predict "Team0 wins" for test matches from the
  heroes alone:
  - a constant (side only);
  - the win-rate model (sum of train logits, no fitting);
  - L2 logistic regression with C chosen on the last 20% of train (in time), never on test.
  Accuracy gets a Wilson CI (correct calls out of n matches); log loss and Brier are reported
  alongside.
- Heroes absent from train (Nurse Harrow, released 2026-10-20, only played after 13:35) are not
  evaluated and count as neutral (rate 0.5, coefficient ≈ 0) in the models.

**Consequence.** The test set has now been used. Any change to the population, the models or
their settings needs fresh data (e.g. another day) to be evaluated honestly.

---

## 2026-10-31: Python 3.15

**Context.** An OS upgrade removed the system Python 3.14 the venv was built on.

**Decision.** `.python-version` = 3.15 (the new system Python); all locked dependencies install
on it. Added `scikit-learn` for the model.

**Consequence.** None for the results; `uv sync` recreates the venv.

---

## 2026-10-30: Complete matches only, temporal 70/30 split, test set held out

**Context.** Matches that started on 2026-10-20 averaged ~11.3 rows, not 12. This came mostly
from StreetBrawl (~8 rows per match). In Normal mode, `quality.match_completeness` shows 98.9% of
matches with exactly 12 rows and 6 winners. The rest have 2–11 rows (players likely disconnected
or the match never really started). The goal is a baseline win rate per hero that later models
and tests check on unseen data.

**Decision.**
- Drop incomplete matches **whole** (not just their rows): keep only matches with
  `players_per_match = 12` rows and `wins_per_match = 6` winners. Every kept match then adds
  exactly 6 wins and 6 losses, so the overall win rate is exactly 50%.
- **Temporal** 70/30 split by match (`[split]` in the config): matches are ranked by
  `start_time` (ties broken by `match_id`), and the first 70% are train, the rest test. All
  players of a match stay on the same side. A random split would mix morning and evening, so it
  could not tell whether a rate *holds later*.
- The test side is **not looked at** (no win rates, no charts) until the evaluation step. Only its
  size is printed.

**Consequence.** 41,685 matches (500,220 rows): 29,179 train (00:00–13:35 UTC) and 12,506 test
(13:35–21:56 UTC). Train and test cover different hours of the day. If the player mix changes
between the morning and evening, rates can shift for that reason too; this is a limitation to keep
in mind when reading the evaluation.

---

## 2026-10-30: Analysis day by `start_time`

**Context.** The file is cut by `created_at`, but `quality.start_day_profile` showed that only
55.5% of rows (50,712 matches) started on 2026-10-20 UTC. The rest are late-ingested matches from
464 different days, back to 2024-06-24, i.e. from other patches.

**Decision.** The Week 1 population keeps only matches with `start_time` on 2026-10-20 UTC
(`population.start_day` in `config/analysis.toml`). Days are always computed in UTC
(`db.connect` sets the time zone).

**Consequence.** One clean day, but a smaller sample than the whole file, so CIs are wider. The
last hour of the day is incomplete (matches only appear once they end and the file stops at
22:13 UTC); this does not bias win rates by itself, but is noted for the day-to-day extension,
which will need equal hour windows.

---

## 2026-10-27: Toolchain and population definition

**Context.** The Week 1 plan suggested `requirements.txt` + venv. The data file is ~1.6 GB, too
large to load comfortably into pandas.

**Decision.**
- `uv` + `pyproject.toml` instead of `requirements.txt` (reproducible lockfile, one command).
- Python 3.14 (the system one; uv here does not download interpreters automatically).
- DuckDB to read the parquet with filters and column selection; only aggregates go to pandas.
- Population fixed before looking at any rate: `game_mode = 'Normal'`,
  `match_mode IN ('Unranked', 'Ranked')`, `match_outcome = 'TeamWin'` (in
  `config/analysis.toml`). On 2026-10-15 this kept 87.4% of rows.

**Consequence.** Any change to the population requires a new entry here, with the reason.
