# 5. Models and how they were trained

The project uses a **ladder** of models, from rules with no learning at all up to deep networks. The point of the ladder is fairness: a fancy model only counts as good if it beats the simple ones on exactly the same rows.

| Tier | Model | Input | Code |
|:---:|:---|:---|:---|
| 0 | Naive momentum, naive flat | nothing / past labels | [baselines.py](../../src/models/baselines.py) |
| 1 | Logistic regression | 27 features | [baselines.py](../../src/models/baselines.py) |
| 2 | Random Forest | 27 features | [tabular_trees.py](../../src/models/tabular_trees.py) |
| 3 | LightGBM | 27 features | [tabular_trees.py](../../src/models/tabular_trees.py) |
| 4 | LSTM | raw book, 100 rows × 40 columns | [recurrent.py](../../src/models/recurrent.py) |
| 5 | DeepLOB, TLOB | raw book, 100 rows × 40 columns | [deeplob.py](../../src/models/deeplob.py), [tlob.py](../../src/models/tlob.py) |

All the settings below are the ones used in Experiment A ([exp_a_predictive_power.py](../../src/experiments/exp_a_predictive_power.py)) unless a section says otherwise.

## One rule shared by every learned model: balanced class weights

The classes aren't equally common. On synthetic data about 70% of rows are FLAT. A model that always says FLAT would be 70% accurate and completely useless. So every learned model weights each class by

```
weight(class) = total rows / (3 × rows in that class)
```

which makes a mistake on a rare class cost more. This pushes models to actually find UP and DOWN moves. It also means **accuracy drops** (they say FLAT less often), which is why the project judges models by macro-F1 instead (guide 6).

Using the same weighting for every model matters: otherwise the models would be optimising different goals and couldn't be compared.

---

## Tier 0: rules with no learning

**Naive flat:** always predicts FLAT. It shows how far "do nothing" gets you.

**Naive momentum:** predicts that the last *known* move repeats. The label at row t is only known k rows later, so at time t the latest known label is y(t−k), and the rule predicts exactly that. (Using y(t−1) instead would leak the future; that was bug 1 of the original audit.)

**Why momentum matters:** it's the bar every real model has to clear. If a model can't beat "the last move continues", it hasn't learned anything useful.

## Tier 1: logistic regression

**How it works.** For each class it computes a weighted sum of the features, then turns the three sums into probabilities (the softmax function). Training finds the weights that make the correct class most probable. It can only draw straight-line boundaries between classes, so it captures "more imbalance → more likely up", but not "imbalance matters only when the spread is 1 tick".

**Training settings.**
- Features **standardised** first: subtract the training mean, divide by the training standard deviation. OFI is in thousands of shares while imbalance is between −1 and 1; without this, the penalty would treat them unfairly.
- scikit-learn `LogisticRegression`, solver `lbfgs`, regularisation C = 1.0, up to 1,000 iterations, balanced class weights.

## Tier 2: Random Forest

**How it works.** A decision tree asks a chain of yes/no questions ("is `imbalance_l1` > 0.3? is `spread_ticks` = 1?") and ends in a leaf that votes for a class. One tree overfits. A Random Forest grows many trees, each on a random sample of rows and with a random subset of features at each split, then lets them vote. The randomness makes the trees' mistakes different, so the vote is more reliable than any one tree.

**Training settings.** 100 trees, max depth 10, at least 10 rows per leaf, class weights balanced within each tree's sample (`balanced_subsample`), all CPU cores.

## Tier 3: LightGBM (gradient-boosted trees)

**How it works.** Also many trees, but built **one after another**: each new tree is trained to fix the errors the trees so far are still making. The final prediction adds up all the trees' outputs. A small learning rate means each tree only nudges the prediction, which generalises better. LightGBM is a fast implementation that grows trees "leaf-wise" (always splitting the leaf that helps most) and bins feature values into histograms.

**Training settings.**
- Objective: multiclass log-loss (3 classes), balanced class weights.
- 63 leaves per tree, max depth 8, learning rate 0.05, at most 200 trees.
- **Early stopping:** after each new tree, check the loss on the validation rows. If it hasn't improved for 15 trees, stop and keep the best number of trees. This stops the model memorising the training data.

**Why it often wins on this data:** tabular data with well-chosen features is exactly what boosted trees are best at. It handles thresholds and interactions automatically and doesn't need huge amounts of data.

On BTC this was the best model (macro-F1 0.467). The C++ version of the BTC model has 109 boosting rounds × 3 classes = 327 trees with 20,601 leaves in total.

---

## The deep models: shared training recipe

All three deep models read a **window of the last 100 rows of the raw 40-column book** (normalised as described in guide 4) and predict the label at the window's last row. The windows end on exactly the same test rows as the tabular models' predictions.

Training recipe ([registry.py](../../src/models/registry.py), `DeepLearningTrainer`):

| Setting | Value | What it means |
|:---|:---|:---|
| Loss | Cross-entropy with balanced class weights | Same goal as the tree models |
| Optimiser | AdamW, weight decay 0.0001 | Adam with a small penalty on large weights |
| Learning rate | 0.0005 (DeepLOB, TLOB), 0.001 (LSTM) | Size of each update step |
| LR schedule | Halve the learning rate if validation loss hasn't improved for 2 epochs | Finer steps once progress stalls |
| Gradient clipping | Max norm 1.0 | Stops rare huge updates from destabilising training |
| Batch size | 128 windows | |
| Max epochs | 10 synthetic, 8 BTC, 15 FI-2010 | One epoch = one pass over the training rows |
| Early stopping | Stop after 3 epochs without validation improvement; restore the best weights | Prevents overfitting |
| Hardware | GPU on Colab (A100/T4) | The laptop can't train these in reasonable time |

**A caveat:** 8–10 epochs is not much training. The deep models may be under-trained, which is one reason not to read too much into their weak BTC scores.

## Tier 4: LSTM

**How it works.** A recurrent network reads the window one row at a time, keeping a "memory" (hidden state) that it updates at each step. An LSTM has gates that decide what to remember and forget, so it can keep information across many steps. The memory after the last row is used to classify.

**Architecture** (62,675 parameters): batch normalisation of the 40 inputs → 2-layer LSTM with 64 hidden units and 20% dropout → take the last step → linear 64 → 32 → ReLU → dropout → linear 32 → 3 classes.

## Tier 5: DeepLOB

**The idea** (Zhang, Zohren and Roberts, 2019): treat the 100 × 40 window like an image and use convolutions to build order-book features automatically, then an LSTM to model how they change over time.

**Architecture** (213,987 parameters), step by step on one window of shape 100 time steps × 40 columns:

1. **Conv block 1:** a 1×2 filter with stride 2 combines each **price with its size** → 20 columns. Then two 4×1 filters mix nearby time steps. (32 channels, LeakyReLU, batch norm.)
2. **Conv block 2:** another 1×2, stride 2, combines **ask with bid** at each level → 10 columns. The paper notes this is where something like a micro-price can form.
3. **Conv block 3:** a 1×10 filter combines **all 10 levels** → 1 column. The 4×1 filters shorten the time axis slightly, to 94 steps.
4. **Two Inception modules:** three parallel branches look at different time scales (3 steps, 5 steps, and a max-pool), 64 channels each, joined into 192. Think of them as learned moving averages of different lengths. (The original paper has one Inception module; ours has two.)
5. **LSTM** with 64 hidden units over the 94 steps; the last step's output goes to
6. **a linear layer** → 3 class scores.

**How well it did.**
- On **FI-2010** it reproduced the paper within 0.7–2.3 accuracy points, so the implementation is sound.
- On **BTC** it scored 0.401 macro-F1, below naive momentum (0.413) and well below LightGBM (0.467); the gap to LightGBM is statistically significant.
- Experiment J's attribution shows it learned to focus on level 1 and on the most recent 10 of its 100 rows.

## Tier 5: TLOB

**The idea** (Berti and Kasneci, 2025): a transformer. Its **self-attention** lets every time step look at every other time step and decide which ones matter, instead of reading the window strictly in order like an LSTM.

**Architecture** (88,795 parameters):
1. **BiN** (bilinear normalisation): normalises the window along both the time axis and the feature axis, learned during training.
2. A linear layer maps the 40 columns at each time step to a 64-number vector.
3. A self-attention layer, then positional encoding (so the model knows the order of the steps).
4. Two transformer blocks: 4-head self-attention over the 100 time steps, then a small feed-forward network (64 → 128 → 64, GELU), with dropout 0.1.
5. Combine the average over all steps with the last step, then a small classifier → 3 classes.

**An honest caveat:** the TLOB paper's "dual attention" attends across **features** as well as across time. In our implementation, both attention layers attend across **time steps**; none attends across the 40 book columns. So this is a simplified TLOB. It scored 0.449 on BTC, better than DeepLOB but below LightGBM.

---

## Models used in specific experiments

| Experiment | Model | Settings that differ |
|:---|:---|:---|
| FI-2010 | Logistic regression, LightGBM, DeepLOB | **No class weights** (to match the paper), LightGBM up to 500 trees with early stopping after 30, DeepLOB 15 epochs, 3 random seeds. Validation = the last 20% of the training days |
| LOBSTER (I) | Logistic regression, LightGBM (300 trees), LightGBM + order flow | Labels without smoothing; the best of the three is reported per stock and horizon |
| Market making (H) | LightGBM | Predicts the *value* of posting a quote (will it fill, and how much it makes or loses 30 s later) |
| Micro-price (K) | Regression versions: Ridge (α = 1, standardised), LightGBM regressor (500 trees, 63 leaves, depth 8), and the **micro-price rule** | The rule is one number: forecast = β × (micro − mid), with β fitted on training rows |
| Optiver RVP | LightGBM regressor | Learning rate 0.05, 63 leaves, up to 3,000 trees, early stopping after 100. Sample weights 1/y², which makes ordinary squared error behave like the competition's percentage error |
| Optiver Close | LightGBM regressor | Mean-absolute-error objective, 255 leaves, depth 10, 70% row subsampling, up to 1,500 trees; early stopping on the last 40 training days, then refit on all training days. One version is retrained every 12 holdout days |

## How long things took

- Tabular models on a BTC segment: seconds to a few minutes on the laptop CPU.
- Deep models: the full BTC run (4 segments, LSTM + DeepLOB + TLOB) took a few hours on a Colab GPU. DeepLOB alone took about 18 minutes per segment in Experiment J.
- FI-2010 with 3 seeds × 4 horizons: about 2 hours on a GPU.

---

## Check yourself

1. Why does every model use balanced class weights, and what does that do to accuracy?
2. What's the difference between a Random Forest and gradient boosting, in one sentence each?
3. What does DeepLOB's first 1×2 convolution with stride 2 combine?
4. What is early stopping protecting against, and what data does it look at?
5. Why didn't the FI-2010 experiment use class weights?

<details><summary>Answers</summary>

1. So rare UP/DOWN classes count as much as FLAT and every model optimises the same goal. Accuracy falls because the models predict FLAT less often.
2. Random Forest: many independent trees on random samples, averaged by voting. Boosting: trees built in sequence, each correcting the previous ones' errors.
3. Each price with its size, at every level and time step.
4. Overfitting (memorising the training rows). It watches the loss on the validation rows, never the test rows.
5. To match how the DeepLOB paper trained and scored, so our numbers can be compared with theirs.

</details>

Next: [6. Metrics](06_metrics.md)
