# Data dictionary: `match_player` (patch-window extract)

One row per player per match. `winlock extract` keeps 18 of the table's 157 columns, plus
`first_created_at`, which it adds itself. File:
`data/raw/patch_window/20261108T1630Z/match_player.parquet`, holding matches that started
10-28 → 10-31 UTC, deduplicated: 11,420,407 rows, 1,016,323 matches.

Mark **verified** only after checking it in the data. Aggregates only: never print `account_id`.

| Column | Type | Meaning | Verified? | How verified / notes |
|---|---|---|---|---|
| `match_id` | uint64 | Match ID; partitioned by `intDiv(match_id, 1000000)` and grows with time | yes (2026-11-08) | `quality.match_completeness`: 99.04% of Normal matches have 12 rows and 6 winners |
| `player_slot` | uint8 | Seat in the match (1–12) | yes (2026-11-08) | Used only to deduplicate (`(match_id, player_slot)` is unique after `extract.deduplicate`) |
| `account_id` | uint32 | Identifies the player. **Personal data: never publish** | yes (2026-11-08) | 620 rows (0.005%) anonymous (null or 0). Never enters the `population` table |
| `hero_id` | uint8 | Hero ID (names in `data/reference/heroes.csv`) | yes (2026-11-08) | 44 ids (1–88), all in `heroes.csv`; no hero twice in a population match (69 other matches have duplicates: 54 StreetBrawl, 15 Normal, all dropped by the filters) |
| `team` | varchar | Player's team, `Team0` / `Team1` | yes (2026-11-08) | 5,710,047 vs 5,710,360 rows; every population match is 6 vs 6 |
| `won` | boolean | Player's team won the match | yes (2026-11-08) | Population matches have exactly 6 `true`, all on one team |
| `start_time` | timestamptz | Match start; defines every window (UTC) | yes (2026-11-08) | 2026-10-28 00:00:14 → 10-31 23:59:59 (the extract filters on it) |
| `created_at` | timestamptz | When the row reached the database | yes (2026-11-08) | Duplicate copies (1,587,924 rows) all share their `created_at`: they're copies, not updates |
| `first_created_at` | timestamptz | Earliest `created_at` over a row's copies (added by `extract.deduplicate`) | yes (2026-11-08) | Equal to `created_at` in this run. ~55% of matches arrive within 1 day of starting, 71–78% within 3 days |
| `game_mode` | varchar | Filter: `'Normal'` | yes (2026-11-08) | Matches: Normal 835,551, StreetBrawl 180,772 |
| `match_mode` | varchar | Filter: `'Unranked'`, `'Ranked'` | yes (2026-11-08) | Matches: Unranked 898,174, Ranked 110,331, CoopBot 4,479, PrivateLobby 3,338, Tutorial 1. Ranked = 13.3% of population rows |
| `match_outcome` | varchar | Filter: `'TeamWin'` | yes (2026-11-08) | Only `TeamWin`: the filter removes nothing here |
| `game_mode_version` | uint32 | **Not a patch marker** | yes (2026-11-08) | 2 for every Normal match and 1 for every StreetBrawl match, on both sides of the update: it follows the mode, not the patch |
| `average_badge` | uint32 | Skill tier | yes (2026-11-08) | Filled for 100% of Ranked rows, 0% of every other mode. Not analysed yet |
| `duration_s` | uint32 | Match duration in seconds | yes (2026-11-08) | 0–6,324 s, median 2,096 s (~35 min); 4,641 rows under 5 min. Not used (it's an outcome of the match) |
| `banned_hero_ids` | uint8[] | Heroes banned in the match | yes (2026-11-08) | 1.5% of Normal matches have different lists on different rows: some rows carry an empty list while their teammates carry the bans. A match's bans = the union over its rows (`population.banned`). ~1.5 bans per match on both sides of the update |
| `not_scored` | boolean | Match did not count (meaning per the manifest; not checked further) | yes (2026-11-08) | 638 matches flagged. Kept (`population.exclude_flags = []`) |
| `low_pri_pool` | boolean | Low-priority matchmaking pool | yes (2026-11-08) | 5,297 matches flagged. Kept |
| `new_player_pool` | boolean | New-player matchmaking pool | yes (2026-11-08) | 26,915 matches flagged; Normal matches with any of the three flags: 32,758 (3.9%). Kept |

Not extracted: `party`, which was 100% null in the earlier 2026-10-20 file, so solo vs group
can't be analysed.
