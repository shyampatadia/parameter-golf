# Presentation script — Parameter Golf
**Total target: ~8 minutes.** Speak at a measured pace; the timings below assume ~145 words/min.

---

## Slide 1 — Title  *(~15 s)*

Hi everyone, I'm Shyam. Today I'll walk you through a controlled study I ran for the OpenAI Parameter Golf challenge — basically, how do you compress a working language model down to 16 megabytes, and which compression levers actually matter?

---

## Slide 2 — "Can a language model fit in 16 MB?"  *(~40 s)*

To set the stakes: 16 megabytes is smaller than a single MP3 song. It's smaller than most PowerPoint decks. And it's about a thousand times smaller than GPT-2 — a model that's already considered ancient by today's standards.

OpenAI set this exact constraint as their *Parameter Golf* challenge: train a model entirely from scratch, ship the weights and the training code together in under sixteen million bytes, and then we score how well it predicts unseen English text. The challenge gives us a reference baseline to beat — 1.2244 bpb. I'll explain what that number means in just a second.

---

## Slide 3 — "What does *score* mean? bits per byte"  *(~50 s)*

So bpb stands for bits-per-byte, and the intuition is really simple: it's how many bits the model needs to *guess each byte* of unseen text. It's literally the same idea as a compression ratio — a good model compresses text by spending fewer bits on each correct prediction; a bad model wastes bits on wrong guesses.

The scale on the right makes this concrete. If you guess randomly over all 256 byte values, that's 8 bpb — you're wasting all 8 bits. If you only know letter frequencies — that "e" is more common than "z" — you get down to about 4.5. The challenge baseline is at 1.22. Frontier models like GPT-4 are around 1.0. And a perfect oracle — a model that always knows what comes next — is at zero.

So **lower is better**, and every hundredth of a bpb is a real, measurable quality difference.

---

## Slide 4 — "How does a model become a 16 MB artifact?"  *(~40 s)*

Here's the pipeline. We train in bfloat16 for 20,000 steps. Then we *quantize* — that just means storing each weight in fewer bits — INT8 or INT6 or INT4. Then we run zlib lossless compression on the quantized weights. That gives us our artifact. We measure that against 16 megabytes. Then we round-trip it: decompress, dequantize, rehydrate the model, and score it.

Inside this pipeline there are three big levers we can push: the **architecture** itself — depth, width, attention design; the **quantizer** — the precision we squeeze the weights into; and the **recipe** — the optimizer and the schedule. Now, lots of efficient-LM papers change all three at once, which makes the attribution unclear. So my method here is to vary one lever at a time — 47 of 48 knobs held fixed — and run 46 controlled experiments at 20,000 steps each.

---

## Slide 5 — "Lever #1: precision"  *(~55 s)*

The first lever — and intuitively the highest-leverage one — is precision.

The reason is simple math. At 16 megabytes, INT8 buys you about 8 million parameters. INT6 buys you 11 million. So lower precision *should* be a free win — you just get more model capacity. Right?

I tested INT8 versus INT6 head-to-head, with QAT — quantization-aware training, which means we *simulate* the rounding to 6 bits during training, so the model has a chance to adapt.

And here's the surprise. Look at the curves on the left — during training, INT6 and INT8 reach exactly the same loss in memory. The QAT does its job. But after we round-trip — actually quantize, compress, decompress — INT8 holds at 1.216 bpb. INT6 collapses to 1.349.

The lesson is sharp: more bits to *spend* doesn't mean more capacity if you can't actually *read them back*. The 6-bit bins just can't represent the trained weight distribution at this scale.

---

## Slide 6 — "Levers 2, 3, and 4"  *(~50 s)*

Three more levers, briefly.

**Position encoding** — how the model knows token order. RoPE rotates each query and key vector by an angle that depends on its position. *Partial* RoPE applies that rotation to only part of each head's dimensions. It wins at 1.215 bpb. The interesting finding here: the *form* of the position prior is second-order — what really matters is *having* one at all. No-PE is the only condition that does badly.

**Optimizer** — Muon orthogonalizes each weight matrix's gradient before stepping, using a Newton–Schulz iteration. It beats AdamW by more than 0.2 bpb at this scale. So Muon isn't just a tunable here — it's load-bearing.

**Weight factorizations** — low-rank, block-diagonal, fixed random projection. The idea is to force structure on the weights to save parameters. All three land between 1.30 and 1.40 — clearly worse than just using dense matrices. So this whole region of the design space is *falsified* as a capacity lever at our scale.

---

## Slide 7 — "Now stack the winners"  *(~40 s)*

The natural next step: combine everything that works. Partial RoPE *plus* INT6 to fund extra capacity *plus* 11 layers *plus* a wider MLP.

And it produces the most informative run of the whole study — for completely the wrong reason. Training-time loss drops to 1.16 — the best we ever saw. But the *final* round-trip loss goes *up* to 1.28 — worse than the simple baseline.

So the capacity expansion is real — minus 0.06 in training. But the INT6 quantizer is taxing us 0.12 on the round-trip. The two effects almost exactly cancel.

Which leaves the question: which lever is paying the bill? Is it the capacity, or is it the quantizer?

---

## Slide 8 — "The decisive experiment"  *(~55 s)*

To answer that, two follow-ups, each isolating one side of the trade.

**(a) Hold the architecture fixed, swap the quantizer.** I tried TurboQuant-INT4. TurboQuant is a clever idea — rotate each weight row by a random orthogonal matrix, which makes the values fall on a known statistical distribution, then quantize with a codebook tuned to that distribution. Important context: TurboQuant was originally designed for the *KV-cache* — that's an inference-time activation cache, not the weights. We're applying the same machinery to weights at 4 bits, which is outside the regime it was validated on. And in fact it fails catastrophically — 3.6 bpb final. So that lever is a dead end.

**(b) Hold the quantizer fixed at known-safe INT8, ease the architecture.** Just go from 9 layers to 11, MLP at 2 instead of 3. That lands at 1.194 bpb — 0.030 below the reference, at 19.3 megabytes.

So the verdict is clean: **the quantizer was paying the bill**. The architecture lever genuinely works — as long as you don't crush precision to fund it.

---

## Slide 9 — "Three more levers"  *(~50 s)*

Three more axes I added beyond the proposal.

**Multi-Query Attention** shares a single K and V projection across all attention heads instead of one set per head. The freed parameters fund a 12-layer model. That hits 1.198 bpb.

**QK-gain initialization** — a learnable scalar on the query projection. I tried three init values; the spread was 0.002 bpb. Null effect — the parameter just learns its way to the same place regardless of where you start it.

**Learning-rate and warmup sweep**, three by three. Spread 0.0019. Also a null. The default recipe was already at the optimum.

**Eval protocol** — instead of scoring non-overlapping 1024-token chunks, slide the window 256 tokens at a time, so every token sees more left-context. Same weights, but the score drops to 1.160. That's not leaderboard-comparable — but it's a fair upper bound on what the model actually knows.

I want to note: three of these are nulls, and I'm reporting them as findings — because negative results constrain the design space and save future work.

---

## Slide 10 — "The headline: a Pareto frontier"  *(~45 s)*

When you put every run on a single size-versus-quality plane, you get this picture. The Pareto frontier is the lower-left envelope — each point on it is the best bpb you can achieve at that size.

And the result has a really clean shape: there's a sharp elbow right around 15.9 megabytes. Below that, every megabyte you save costs you about 0.02 bpb — steep tradeoff. Above it, the curve flattens out — you have to spend more and more megabytes to buy each additional bit of quality.

So the framing I want to push back against is the binary one. The 16-megabyte mark isn't a wall. It's a soft elbow on a smooth curve. The in-budget winner is at 1.215 bpb. Spend an extra 3.3 megabytes and you get to 1.194. That's a real, measurable trade — not a constraint violation.

---

## Slide 11 — "Are these numbers real, or seed luck?"  *(~25 s)*

Quick sanity check. I re-ran the three surviving in-budget axes at three different random seeds. The standard deviations are around one or two thousandths of a bpb. The per-axis effects we care about are 0.05 to 0.20 bpb — ten to a hundred times the seed noise. So the single-seed claims hold up.

---

## Slide 12 — "What we learned"  *(~50 s)*

Five takeaways.

One — sub-INT8 weight quantization is fragile at this scale. Neither QAT nor random rotation saves it. The quantizer, not the recipe, is the binding constraint.

Two — capacity expansion is real, but only INT8 carries it. The exact same architectural change forfeits its gain under INT6 and lands 0.030 below reference under INT8.

Three — inductive biases matter at the margin, not the headline. RoPE, MQA, depth — each is worth a hundredth or two of a bpb. None of them rivals the precision lever in effect size.

Four — 16 megabytes is a soft elbow on a smooth Pareto curve. It's not a hard cutoff to architect around.

Five — three nulls reported as findings. Weight factorizations, QK-gain init, LR-warmup. Each one rules out a region of the design space.

---

## Slide 13 — Thanks  *(~15 s)*

Thank you. The full report and code are on the GitHub link. Happy to take questions.

---

### Pacing notes

- **Total spoken: ~1150 words ≈ 7 min 55 s** at 145 wpm.
- The two longest sections (slides 5 and 8) are where the *story turns* — slow down on those, especially "the surprise" on slide 5 and "the verdict" on slide 8.
- Slide 11 (multi-seed) is a deliberate quick beat — ~25 seconds — to give you breathing room before the takeaways.
- If you're running long, the cuttable pieces are: the parenthetical on weight factorizations (slide 6), and the QK-gain detail (slide 9). Each saves ~10 s.
