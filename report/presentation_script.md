# Presentation script — Parameter Golf
**Total target: ~8 minutes** at a measured pace (~135 words/min).

The structure is built around three questions that the project answers, plus a clearly stated goal up front.

---

## Slide 1 — Title  *(~15 s)*

Hi everyone, I'm Shyam. The project I'll walk you through today asks a simple question: how small can a language model get and still be useful?

---

## Slide 2 — The goal  *(~50 s)*

The goal of this project is straightforward: train a useful language model that fits in 16 megabytes. That's the constraint OpenAI sets in their *Parameter Golf* challenge.

To put 16 megabytes in perspective — that's smaller than a single MP3 song, smaller than most PowerPoint decks, and about a thousand times smaller than GPT-2.

The reason this is an interesting problem: when capacity is this scarce, every megabyte you save on one component has to be earned somewhere else, so the constraint tests which design decisions actually matter. And on the practical side, a working model at this size is small enough to ship inside an app or run on-device.

The number we're trying to beat — the published baseline for this challenge — is 1.2244 bpb. I'll explain what bpb means in just a second.

---

## Slide 3 — bpb intuition  *(~50 s)*

So bpb is bits-per-byte. It's how many bits the model spends to *guess each byte* of unseen text. It's the same idea as a compression ratio — a good model spends fewer bits on correct guesses; a bad model wastes bits on wrong ones.

The scale on the right anchors this. If you guess randomly over all 256 possible byte values, that's 8 bpb — you're using all 8 bits and getting no benefit from a model. If you only know which letters are common in English, you get to about 4.5. The challenge baseline is at 1.22. Frontier models like GPT-4 are around 1.0. A perfect oracle is at zero.

So lower is better, and every hundredth of a bpb is a measurable difference in quality.

---

## Slide 4 — What is this model, and what does it do?  *(~45 s)*

Before we get into compression, let me say what this model actually *is*. It's a decoder-only Transformer — the same architectural family as GPT. It has one capability: read a sequence of tokens, and output a probability distribution over the next token.

That single capability is enough. Generation is just sampling from that distribution, appending the new token to the input, and repeating — that's the loop every modern language model uses to produce text.

The diagram on the right makes it concrete: feed it the prefix "The quick brown", and the model returns a probability for every possible next token — fox at 62 percent, dog at 8, and so on.

We train this model from scratch on FineWeb — a 10-billion-token English shard that HuggingFace released — using a 1,024-piece SentencePiece tokenizer. No pretrained weights, no transfer learning. Every result is from a model trained entirely from random initialization.

---

## Slide 5 — How a model becomes a 16 MB artifact  *(~50 s)*

Now, how do we get from a trained model to a 16 megabyte artifact?

This is the pipeline. We train in bfloat16 for 20,000 steps. We *quantize* the weights — store each one in fewer bits. We compress the result with zlib. That's our artifact, and we measure it against the 16 megabyte limit.

I want to pause on the last box — "round-trip" — because the word can be confusing. Round-trip here has nothing to do with the dataset. It means: compress the artifact, decompress it, dequantize the weights, then re-evaluate. We always score the model the *user* would receive — not the bf16 model sitting in GPU memory at the end of training. That distinction matters.

So the question this project asks is: out of all the things we can change in this pipeline — the architecture, the quantizer, the training recipe — which ones actually move the needle? I tested each decision one at a time. Forty-six runs at twenty thousand steps each.

---

## Slide 6 — Question 1: how aggressively can we shrink the weights?  *(~55 s)*

The first question, and the one with the highest leverage, is precision.

Here's the math. At 16 megabytes, INT8 — eight-bit weights — buys you about 8 million parameters. INT6 buys you about 11 million. So lower precision *should* be a free win — more capacity for the same budget.

I tested INT8 and INT6 head-to-head, with quantization-aware training. That means we *simulate* the rounding to 6 bits during training, so the model has a chance to adapt.

Look at the curves on the left. During training, the two converge to the *same* loss. The QAT does its job — in memory, INT6 is just as good as INT8.

But after we round-trip through zlib, INT8 holds at 1.216 bpb, and INT6 falls to 1.349. The 6-bit bins can't represent the trained weight distribution at this scale, and QAT doesn't recover the gap.

So the takeaway is: more bits to *spend* doesn't mean more capacity if you can't read them back.

---

## Slide 7 — Question 2: does the rest of the design matter?  *(~55 s)*

Three more decisions, one at a time.

First, position encoding — how the model knows the order of tokens. I tested four schemes. Partial RoPE wins at 1.215 bpb, but the gap between any two *position-aware* schemes is small. The only thing that hurts is removing position information entirely. So having a position prior is first-order; the form of it is second-order.

Second, the optimizer. Muon, which orthogonalizes each weight matrix's gradient before stepping, beats AdamW by more than 0.2 bpb at this scale. The optimizer is load-bearing here, not a free knob.

Third, weight factorizations — low-rank, block-diagonal, fixed random projection. These force structure on the weights to save parameters. All three land between 1.30 and 1.40 — worse than just using dense matrices. So at our scale, structured weights don't compete with dense ones.

---

## Slide 8 — Question 3: can we just stack the winners?  *(~45 s)*

The natural next move: combine everything that worked. Take the best position encoding, switch to INT6 to fund extra capacity, push to 11 layers and a wider MLP.

What we get is informative — but for the wrong reason.

Training-time loss drops to 1.16 — the lowest training loss in the study. But the *final* round-trip loss goes *up* to 1.28 — worse than the simple baseline.

So the capacity expansion is real — minus 0.06 in training. But the INT6 quantizer is taxing us 0.12 on the round-trip. The two effects almost exactly cancel.

This raises the question: which lever is responsible — the capacity, or the quantizer?

---

## Slide 9 — The answer  *(~55 s)*

To answer that, two follow-ups, each isolating one side of the trade.

First — hold the architecture fixed, and swap the quantizer. I tried TurboQuant at 4 bits. TurboQuant is a method that rotates each weight row by a random orthogonal matrix, which makes the values fall on a known statistical distribution, then quantizes with a codebook tuned to that distribution. One important caveat: TurboQuant was originally designed for the *KV-cache* — that's an inference-time activation cache, *not* the weights. We're applying the same machinery to weights at 4 bits, which is outside what it was validated on. The result is 3.6 bpb final — the quantizer fails far below 8 bits, so this lever is a dead end.

Second — hold the quantizer fixed at INT8, and ease the architecture. Go from 9 layers to 11, MLP at 2 instead of 3. That lands at 1.194 bpb — 0.030 below the reference, at 19.3 megabytes.

So the result is clear: the quantizer was the binding constraint. The architecture lever works — as long as you don't crush precision to fund it.

---

## Slide 10 — What else we tried  *(~45 s)*

Four more axes, more briefly.

Multi-Query Attention shares a single K/V projection across all attention heads instead of one per head. The freed parameters fund a 12-layer model. That reaches 1.198 bpb.

QK-gain initialization — a learnable scalar on the query projection. I tried three init values; the spread was 0.002 — null effect.

Learning-rate and warmup sweep, three by three. Spread 0.0019 — also null. The default recipe is already at the optimum.

Eval protocol — instead of scoring non-overlapping context windows, slide the window 256 tokens at a time so every token sees more left-context. Same weights score 1.160 bpb. But this is not leaderboard-comparable — the official protocol uses non-overlapping windows — so I'm reporting it as an upper bound, not as our score.

Three of those are null effects, and I'm reporting them as findings. Negative results constrain the design space.

---

## Slide 11 — The full size-quality picture  *(~40 s)*

When you put every run on a single size-versus-quality plane, you get this Pareto frontier — the lower-left envelope of what's achievable at each size.

There's an elbow right around 15.9 megabytes. Below that elbow, every megabyte you save costs you about 0.02 bpb. Above it, the curve flattens out.

The takeaway: 16 megabytes isn't a hard wall — it's an elbow on a continuous curve. The in-budget winner is at 1.215 bpb. Spend an extra 3.3 megabytes and you reach 1.194. It's a continuous trade, not a binary.

---

## Slide 12 — Are these numbers real?  *(~25 s)*

A sanity check. I re-ran the three surviving in-budget axes at three different random seeds. The standard deviations come in around one or two thousandths of a bpb. The per-axis effects we care about are 0.05 to 0.20 — ten to a hundred times the seed noise. The findings hold up.

---

## Slide 13 — What we learned  *(~50 s)*

Five takeaways.

One — the quantizer is the binding constraint. Below 8 bits, weight quantization breaks at this scale, and the fixes I tried don't recover it.

Two — architecture matters, but only if the quantizer can carry it. The same 11-layer model fails under INT6 and wins under INT8.

Three — inductive biases like RoPE, MQA, and depth matter at the margin. Each is worth a hundredth or two of a bpb — real but small.

Four — 16 megabytes is an elbow on the curve, not a hard cutoff. Spending 3.3 megabytes more buys 0.020 bpb — a continuous trade.

Five — three nulls are findings too. Weight factorizations, QK-gain init, learning-rate sweeps. Saying *no* rules out a region of the design space, and that's as useful as a positive result.

---

## Slide 14 — Where we started, where we ended  *(~40 s)*

To close, the bigger picture — the journey from start to finish.

We started with one number: the published 1.2244 baseline at 15.85 megabytes. One configuration, no insight into which of its choices were earning the score.

We ended with two competitive models: an in-budget winner at 1.215 bpb at 15.87 megabytes, and an over-budget winner at 1.194 bpb at 19.30 megabytes — 0.030 bpb below the published reference.

We also mapped the entire size-quality curve from 7 megabytes up to 19, and we know which compression decisions earn the gains and which ones don't.

And the model is runnable code. The smoke test in `experiments/demo/` loads a trained checkpoint, runs a forward pass on a prompt, and prints the top-k next-token predictions — confirming the artifact works as a language model.

---

## Slide 15 — Thanks  *(~10 s)*

Thank you. The full report and code are on the GitHub link. Happy to take questions.

---

### Pacing notes

- **Total: ~8 min** at 135 wpm.
- **Structure:**
  1. **Set up** — goal (slide 2), metric (3), the model (4), pipeline + the question (5).
  2. **Three questions** — precision (6), other design choices (7), stacking the winners (8).
  3. **Resolution** — the A/B that identifies the bottleneck (9).
  4. **Filling in** — other axes (10), Pareto (11), seed check (12).
  5. **Close** — takeaways (13), start → end + demo (14).
- **Slides to slow down on:** slide 6 (the INT6 round-trip result), slide 8 (the question that drives slide 9), slide 9 (the conclusion that the quantizer is the binding constraint).
- **Cuttable if running long:** the QK-gain bullet on slide 10 (~10 s), or the entire seed-check slide 12 (~25 s).
