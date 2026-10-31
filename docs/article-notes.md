# Article notes

Ideas and wording to reuse in the blog post. Numbers from `reports/tables/` (2026-10-20 UTC).

## Key paragraph: the model is limited by design (Part 2, Discussion)

> "Inefficient" is the wrong word: the model is **deliberately limited**. It never sees who is
> playing, and that was the experiment's design. Win rates alone are enough to beat chance
> consistently, but not enough to call a single match confidently: 4 in 10 predictions are
> still wrong. The missing part most likely depends on the players, which I chose not to use.
> That makes the 60% a measure of how far hero information alone can go, which is more
> interesting than calling the model inefficient.

## Two questions, two tools

| Question | Tool | Result |
|---|---|---|
| Does a hero's win rate differ from 50%, and does it hold over the day? | Wilson CI per hero, Holm, train vs test coverage | 32 of 43 differ after Holm; 0 changed between morning and evening |
| Do win rates help predict matches the model has never seen? | Held-out test (train in the morning, score in the evening) + Wilson CI on accuracy | 60.1% correct (95% CI 59.3–61.0%), so not luck |

Wording: "supported on this day", not "proved".

## Social posts (drafts; publish only after the blog post is live)

Update the numbers if the player-clustering or shrinkage analyses change them.

### LinkedIn

> Are hero win rates in Deadlock real, or just noise? I spent my first data project finding out.
>
> I took one day of public match data (41,685 matches, 43 heroes) and asked a simple question:
> how much can you trust a win rate?
>
> What I found:
> • Win rates ranged from 43% to 57%. After correcting for testing 43 heroes at once (Holm),
>   32 of them still differ from a fair 50%.
> • They held up over the day: rates measured in the morning matched the evening (r = 0.98).
> • Knowing only the 12 heroes in a match, a simple model called 60% of evening matches
>   correctly (95% Wilson CI: 59.3–61.0%). Better than a coin flip, far from certain.
> • Hero "synergy": 903 teammate pairs tested, none survived correction.
>
> The honest part: a win rate measures the hero AND the people who pick it, and this is one day
> of data. The model never sees who is playing, on purpose: the 60% shows how far hero
> information alone can go.
>
> Biggest lesson: the interesting work wasn't the model, it was knowing what I'm allowed to claim.
>
> Full write-up, charts and code: [blog link] · [GitHub link]
> Data: Deadlock API (deadlock-api.com)
>
> #DataScience #Statistics #Python #Deadlock

### Twitter / X thread

1. Are Deadlock hero win rates real or noise? I analysed one day of public data: 41,685 matches,
   43 heroes. 🧵
2. Win rates ranged from 43% to 57%. With 95% Wilson CIs and a correction for testing 43 heroes
   at once, 32 still differ from 50%. [chart: train_winrate_ci.png]
3. They held over the day: morning vs evening r = 0.98, no hero changed significantly.
4. Heroes alone predict 60% of matches (CI 59.3–61.0%). Better than a coin flip, but 4 in 10
   calls are still wrong: the rest is the players.
5. Caveats: one day, unknown patch, a win rate includes who picks the hero. Write-up + code:
   [link]
