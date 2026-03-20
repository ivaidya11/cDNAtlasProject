Ready for review
Select text to add comments on the plan
Plan: Physicochemical Class Classifier from Mean Current
Context
The project extracts per-step nanopore current features from simulated peptide traces. features.csv contains 11,971 labelled steps (trace × amino acid) with columns mean_current, std_current, dwell_time, amino_acid, phys_class. The user observed a clean separation between positively and negatively charged residues in mean_current and wants a classifier that:

Separates steps into physicochemical classes using mean current
Makes the natural current ordering of classes explicit (positive < other < negative)
Key Findings from Exploration
Class	Amino acids	Mean current	Separable?
positive	K, R	~73 pA	Yes — cleanly low
nonpolar	G,A,V,L,I,P,F,M,W	~89 pA	No — overlaps polar
polar	S,T,C,N,Q,Y,H	~90 pA	No — overlaps nonpolar
negative	D, E	~101 pA	Yes — cleanly high
std_current and dwell_time carry zero class-discriminative signal (noise + kinetics are class-independent in the simulation)
3-class scheme is correct: positive / negative / other (polar + nonpolar merged)
Physics reason: positive charge (+1) decreases current via -BETA*charge; negative charge (-1) increases it; neutral residues governed only by volume → overlap
Implementation Plan
Step 0 — Install dependencies
uv add scikit-learn scipy
Step 1 — Create classifier.ipynb
Section 1: Setup

Imports: numpy, pandas, matplotlib, seaborn, scipy.stats, sklearn
Define CLASS_COLORS_4, CLASS_COLORS_3, ORDER_4 = ['positive','nonpolar','polar','negative'], ORDER_3 = ['positive','other','negative']
Load features.csv
Create class3 column: replace polar/nonpolar with other
Section 2: Exploratory Visualisation (confirms ordering)

KDE plot: 4-class overlaid (show positive and negative separation clearly)
KDE plot: 3-class overlaid
Violin plot ordered by median (left-to-right = positive → other → negative) with median annotations
Summary statistics table (n, mean, std, p5, p25, median, p75, p95) per class
Scatter of std_current and dwell_time by class — show these add no signal
KDE overlap quantification with np.trapz(np.minimum(kde_a, kde_b), x) for each class pair
Section 3: Threshold Classifier (baseline)

Grid-search t1 (positive/other boundary) and t2 (other/negative boundary) to maximise macro F1
np.linspace(75, 92, 100) × np.linspace(88, 110, 100)
Classification report + confusion matrix
KDE with vertical threshold lines and shaded decision regions
Section 4: Probabilistic and ML Models

X = df[['mean_current']].values, y = df['class3'].values
LabelEncoder with fixed ORDER_3 ordering
Models:
GaussianNB()
Pipeline([StandardScaler, LogisticRegression(multi_class='multinomial', class_weight='balanced')])
RandomForestClassifier(n_estimators=200, class_weight='balanced')
Note: use class_weight='balanced' — "other" has ~8× more samples than positive/negative
5-fold StratifiedKFold cross-validation with macro F1, weighted F1, accuracy
Summary table of CV results
Section 5: Evaluation

80/20 stratified train/test split
Side-by-side confusion matrices for all three models
Per-class F1 bar chart across models
Section 6: Interpretability

Logistic Regression: plot predicted probabilities per class across x = 40–130 pA range
Gaussian NB: plot fitted Gaussian emission per class (μ, σ directly readable as learned physics)
RF: feature importance (trivially 1.0 with one feature — note this confirms single-feature sufficiency)
Section 7: Ordering Summary

Thesis-ready figure: KDE of all 4 classes with median lines, bidirectional arrow showing Δ ≈ 27 pA between positive and negative
Markdown cell explaining physics (ALPHA×volume, BETA×charge terms from logic/database.py)
Recommendation cell: threshold or LR for interpretability; LR as prior for future HMM integration
Critical Files
File	Role
features.csv	Input — all 11,971 labelled steps
logic/database.py	AA_CLASS mapping + simulation constants (ALPHA, BETA, IOS) for physics explanation
stepdetect.ipynb	Reference for plot style / colour conventions
pyproject.toml	Add scikit-learn and scipy dependencies
New file: classifier.ipynb (root of project)

Verification
uv run jupyter notebook classifier.ipynb — run all cells, confirm no import errors
Check class3 value_counts: ~9546 other, ~1224 positive, ~1201 negative
Threshold CV: macro F1 should be ≥ 0.85 (given clean separation)
Confirm GNB fitted means match expected: positive μ ≈ 73 pA, other μ ≈ 89 pA, negative μ ≈ 101 pA
Overlap metric positive/negative should be < 0.01 (nearly no overlap)
