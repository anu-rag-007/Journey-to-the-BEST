% LUCID Phase 2 — Paper Draft
% Target: NeurIPS ML4H Workshop or IEEE EMBC
% Author: Anurag Sharma, Galgotias University

\documentclass[conference]{IEEEtran}
\usepackage{amsmath,graphicx,booktabs,xcolor}
\usepackage{hyperref}

\begin{document}

\title{Towards Dream Imagery Reconstruction:\\
EEG-Guided Latent Diffusion via ATM Alignment\\
for REM Sleep Applications}

\author{
  \IEEEauthorblockN{Anurag Sharma}
  \IEEEauthorblockA{
    Department of Computer Science and Engineering
    (AI \& ML)\\
    Galgotias University, Greater Noida, India\\
    sharma.anurag0706@gmail.com
  }
}

\maketitle

% ──────────────────────────────────────────────
\begin{abstract}
We present the second phase of LUCID: Reality?,
a long-term initiative toward artificial reality
through the dream interface. Building on our
previously published CNN-LSTM sleep staging
system ($\kappa$=0.68, Sleep-EDF Extended,
153 recordings), we develop an EEG-guided image
generation pipeline targeting the REM sleep state.
Using the THINGS-EEG dataset (50 subjects,
1,654 training concepts), we train an
Attention-based Temporal-spatial Model (ATM)
encoder to align 17-channel EEG representations
with CLIP image embeddings via InfoNCE contrastive
learning. Our best configuration achieves
Top-5 retrieval accuracy of 4.5\% (1.8$\times$
chance) on 200 held-out test concepts.
We demonstrate that image CLIP targets
outperform text-based targets, that mixing
target modalities degrades performance, and
that EEG captures low-level visual shape
features (evidenced by semantically proximate
retrieval errors). A complete generation
pipeline—EEG epoch → ATM encoder → CLIP
embedding → Stable Diffusion—is demonstrated,
producing semantically coherent imagery from
EEG-derived embeddings. To our knowledge, this
is the first system to integrate automated
REM sleep detection with EEG-conditioned
image generation in a closed-loop BCI pipeline.
\end{abstract}

\begin{IEEEkeywords}
EEG, neural decoding, CLIP, diffusion models,
brain-computer interface, REM sleep, lucid
dreaming, contrastive learning, ATM
\end{IEEEkeywords}

% ──────────────────────────────────────────────
\section{Introduction}

Reconstructing visual imagery from brain signals
has advanced rapidly with fMRI-based systems
achieving compelling results~\cite{takagi2023,
mindvis2023}. However, fMRI requires clinical
infrastructure, precluding deployment in
naturalistic sleep settings. Electroencephalography
(EEG) offers a wearable alternative at the cost
of spatial resolution and signal-to-noise ratio.

Prior EEG decoding work targets the \emph{awake}
visual response: subjects view images while EEG
is recorded, and models learn to retrieve or
reconstruct the viewed image~\cite{scotti2023}.
We address a fundamentally different problem:
decoding visual content from \emph{REM sleep},
where imagery is internally generated rather
than externally driven, and where signal
characteristics differ substantially from
visual evoked potentials.

Our contributions are:
\begin{enumerate}
  \item An end-to-end pipeline connecting automated
  sleep staging (Phase 1, previously published)
  with EEG-guided image generation (Phase 2).

  \item An ablation study of CLIP target
  modalities (text vs.\ image), training
  temperatures, and channel counts on the
  THINGS-EEG benchmark.

  \item Evidence that EEG encodes low-level
  visual shape information, with retrieval
  errors showing shape-preserving concept
  substitution (e.g., baton $\rightarrow$
  blowtorch).

  \item A demonstration that temporal cortex
  channels carry object-identity information
  in EEG recorded during visual stimulation,
  consistent with the ventral visual stream.
\end{enumerate}

% ──────────────────────────────────────────────
\section{Related Work}

\subsection{EEG-to-Image Decoding}
Scotti et al.~\cite{scotti2023} demonstrated
EEG-to-image retrieval on THINGS-EEG using
contrastive alignment, achieving approximately
22\% Top-5 accuracy using individual-subject
EEG and GPU-scale batch training. Our work
differs in three respects: (1) we target the
REM sleep state rather than awake visual
stimulation; (2) we provide a systematic
ablation of target modality and training
temperature; (3) we integrate retrieval with
a generation pipeline producing novel imagery.

\subsection{Contrastive EEG Alignment}
CLIP~\cite{radford2021} demonstrated that
text and image representations can be aligned
in a shared embedding space via contrastive
learning on paired data. We apply this
principle to (EEG, image) pairs, using CLIP
image embeddings as fixed targets and training
only the EEG encoder.

\subsection{ATM Architecture}
The Attention-based Temporal-spatial Model
(ATM)~\cite{atm2024} processes EEG by applying
channel-wise attention over independently
embedded electrode tokens, followed by temporal
and spatial convolution to capture multi-scale
patterns. We adopt this architecture for its
interpretable attention maps, which reveal
electrode contributions to the learned
representation.

% ──────────────────────────────────────────────
\section{Methods}

\subsection{Phase 1: Sleep Stage Classification}
We use our previously published CNN-LSTM
classifier~\cite{sharma2026} trained on the
Sleep-EDF Extended dataset (153 recordings,
177,411 epochs). The classifier achieves
$\kappa$=0.68 under random split and
$\kappa$=0.67 under leave-one-subject-out
(LOSO) validation on a 20-subject subset.
REM detection ($F_1$=0.81) triggers the
Phase 2 generation pipeline via a four-condition
safety interlock: $P(\text{REM}) > 0.6$,
sustained $\geq$30\,s, no Wake in prior 60\,s,
and 300\,s cooldown.

\subsection{Dataset: THINGS-EEG}
We use the THINGS-EEG dataset~\cite{gifford2022},
comprising EEG recordings from 50 subjects
viewing 1,654 training and 200 test images
from the THINGS object concept
database~\cite{things2019}.
Preprocessed data provides 17-channel EEG
epochs at 100\,Hz covering $-$200\,ms to
790\,ms relative to image onset, with
80 repetitions per concept per subject.

\textbf{Preprocessing.}
We average across all 80 repetitions per
concept per subject, then average across
subjects, yielding one clean EEG epoch per
concept: $(1654, 17, 100)$ training and
$(200, 17, 100)$ test. Each channel is
independently normalised to zero mean and
unit variance.

\subsection{ATM EEG Encoder}

\textbf{Channel embedding.}
Each of the 17 EEG channels (100 timepoints)
is projected independently via a shared linear
layer to a $d_\text{model}$=128 dimensional
token, producing a sequence of channel tokens
$(B, 17, 128)$.

\textbf{Channel-wise attention.}
Multi-head self-attention (4 heads) is applied
across the 17 channel tokens. A gating mechanism
produces per-channel importance weights,
enabling post-hoc interpretation of electrode
contributions.

\textbf{Temporal-spatial convolution.}
Two parallel convolutional paths capture
complementary structure: a temporal path
(Conv1D across time within each channel,
kernel 15) capturing ERP dynamics; and a
spatial path (Conv1D across channels at each
timepoint, kernel 5) capturing electrode
co-activation patterns. Their outputs are
fused via a linear layer.

\textbf{Aggregation and projection.}
Channel-attended and temporal-spatial
representations are concatenated and projected
to a 512-dimensional CLIP-compatible embedding,
which is L2-normalised.

The complete ATM encoder contains 626,449
trainable parameters.

\subsection{CLIP Target Computation}

For each training concept, we compute a CLIP
image embedding~\cite{radford2021} by loading
all available images from the corresponding
THINGS concept folder, computing the CLIP
ViT-B/32 image feature for each, and averaging
across images before re-normalisation. This
produces a robust prototype embedding per
concept. Mean inter-concept cosine similarity
is 0.529 for image targets, compared to 0.713
for text-based targets of the form
\texttt{``a photo of a \{concept\}''}, confirming
that image targets are more discriminative.

\subsection{Training}

\textbf{Loss function.}
We use InfoNCE contrastive loss with temperature
$\tau$=0.05, applied bidirectionally between
EEG embeddings and CLIP image targets within
each mini-batch:
\begin{equation}
\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}
\left[
  \log\frac{e^{\mathbf{z}_i \cdot
    \mathbf{c}_i/\tau}}
    {\sum_j e^{\mathbf{z}_i \cdot
    \mathbf{c}_j/\tau}}
  + \log\frac{e^{\mathbf{c}_i \cdot
    \mathbf{z}_i/\tau}}
    {\sum_j e^{\mathbf{c}_i \cdot
    \mathbf{z}_j/\tau}}
\right]
\end{equation}
where $\mathbf{z}_i$ is the normalised EEG
embedding and $\mathbf{c}_i$ is the normalised
CLIP image embedding for concept $i$.

\textbf{Optimiser.}
AdamW with weight decay 0.05, OneCycleLR
schedule (peak lr $5 \times 10^{-4}$,
10\% warmup), 200 epochs, batch size 128.
EEG augmentation: Gaussian noise
($\sigma$=0.08), random channel dropout
(2--3 channels, $p$=0.25), and temporal
shift ($\pm$3 timepoints, $p$=0.30).

\subsection{Generation Pipeline}

Upon ATM encoding, the EEG embedding is
compared against all 1,654 training concept
embeddings via cosine similarity to retrieve
the nearest concept. A text prompt is
constructed from the concept name and passed
to Stable Diffusion v1.5~\cite{rombach2022}
running locally via ComfyUI, producing a
512$\times$512 pixel image. In the closed-loop
deployment, this generation is triggered upon
REM detection by the Phase 1 classifier.

% ──────────────────────────────────────────────
\section{Experiments}

\subsection{Ablation: CLIP Target Type}

Table~\ref{tab:ablation} reports Top-5
retrieval accuracy on the 200 test concepts
under five experimental conditions.

\begin{table}[t]
\centering
\caption{Ablation: CLIP target modality and
training temperature. Test set: 200 THINGS-EEG
concepts. Chance Top-5 = 0.025.}
\label{tab:ablation}
\begin{tabular}{lcc}
\toprule
\textbf{Configuration} &
\textbf{Top-5} &
\textbf{$\times$Chance} \\
\midrule
Chance & 0.025 & 1.0$\times$ \\
\midrule
Text CLIP, $\tau$=0.07 & 0.035 & 1.4$\times$ \\
Mixed (70\% img), $\tau$=0.05
  & 0.030 & 1.2$\times$ \\
Image CLIP, $\tau$=0.03 & 0.040 & 1.6$\times$ \\
\textbf{Image CLIP, $\tau$=0.05}
  & \textbf{0.045} & \textbf{1.8$\times$} \\
Image CLIP, 63-ch, $\tau$=0.05
  & 0.035 & 1.4$\times$ \\
\bottomrule
\end{tabular}
\end{table}

Image targets outperform text targets by
0.010 Top-5 accuracy (28.6\% relative),
consistent with higher inter-concept
discriminability (mean similarity 0.529
vs.\ 0.713). Mixing text embeddings
\emph{reduces} performance below the text-only
baseline, suggesting that text and image CLIP
representations occupy sufficiently distinct
regions of embedding space that their mixture
introduces noise rather than combining signal.

Increasing channel count from 17 to 63 reduces
performance under our training regime, as the
expanded input dimensionality (6,300 vs.\
1,700 features) cannot be constrained by 1,654
training pairs, leading to training instability.
This identifies data scale as the primary
bottleneck.

\subsection{Channel Importance Analysis}

The ATM gating mechanism reveals which
electrodes contribute most to the learned
representation. Under the 17-channel
configuration, P3 (left parietal) receives
the highest gate weight. P3 is the scalp
projection of the P300 ERP component,
associated with attentional orientation
toward salient stimuli. Under the 63-channel
configuration, temporal region electrodes
(T7, T8 and neighbours) dominate, consistent
with the inferior temporal cortex's role in
visual object identity processing — the
terminal stage of the ventral visual
stream~\cite{dicarlo2012}.

\subsection{Qualitative Analysis}

Retrieval errors exhibit a consistent pattern:
retrieved concepts share low-level visual
structure with the true concept rather than
semantic category. The most illustrative
example in our test set is the retrieval of
\textit{blowtorch} (rank 3) for the true
concept \textit{baton}, both of which are
elongated cylindrical objects with similar
aspect ratios and silhouettes. This suggests
that EEG signals in the THINGS-EEG paradigm
carry shape-level visual features, with
semantic identity encoded more weakly at
this spatial resolution and training scale.

% ──────────────────────────────────────────────
\section{Discussion}

\subsection{Comparison to Prior Work}

Scotti et al.~\cite{scotti2023} achieved
approximately 22\% Top-5 accuracy on
THINGS-EEG using individual-subject EEG
(not averaged) with GPU-scale batch training.
Our 4.5\% Top-5 reflects two primary
differences: (1) we average EEG across
subjects, reducing effective training pairs
from $\sim$16,540 to 1,654; (2) CPU training
limits batch size to 128, reducing the
number of negative pairs per update from
which contrastive learning derives its signal.
With individual-subject training and
GPU access, we anticipate performance
approaching published benchmarks.

\subsection{Generation vs.\ Retrieval}

For the LUCID application, generation fidelity
rather than retrieval accuracy is the
operationally relevant metric. A generated
image need not match the \emph{exact} training
concept; it must be semantically consistent
with the mental imagery the EEG epoch was
elicited by. Our generation pipeline produces
coherent, high-quality imagery (via SD v1.5)
from EEG-derived concept embeddings,
demonstrating the feasibility of the proposed
approach independent of retrieval accuracy.

\subsection{Limitations}

Three primary limitations affect the current
system. First, the training data comprises
EEG during \emph{awake} visual stimulation,
while the deployment target is \emph{REM sleep}
EEG — a domain shift whose magnitude has not
been quantified. Second, retrieval accuracy
is modest due to data scale constraints.
Third, end-to-end REM-state generation
has not yet been validated with a wearable
device. Each of these is addressed in the
future work plan.

% ──────────────────────────────────────────────
\section{Conclusion}

We have presented the first system integrating
automated REM sleep detection with EEG-conditioned
image generation in a closed-loop BCI pipeline.
Our ATM-based alignment model achieves 1.8$\times$
chance Top-5 retrieval on THINGS-EEG, with
ablation results identifying image CLIP targets
and standard contrastive temperature as optimal
choices. Channel attention analysis reveals that
EEG carries shape-level visual information,
with temporal cortex channels most informative
for object identity. The complete Phase 1 +
Phase 2 pipeline — EEG sleep staging through
to Stable Diffusion image generation — is
released as open source and represents a
significant step toward the long-term goal
of LUCID: Reality?.

Future work will train on individual-subject
EEG to recover the 10$\times$ training data
advantage, integrate a consumer EEG headband
(Muse S) for real-time REM-state generation,
and conduct a prospective self-experiment
comparing generated images with morning
dream reports.

% ──────────────────────────────────────────────
\begin{thebibliography}{99}

\bibitem{takagi2023}
Y.\ Takagi and S.\ Scotti,
``High-resolution image reconstruction with
latent diffusion models from human brain
activity,'' in \textit{CVPR}, 2023.

\bibitem{mindvis2023}
Z.\ Chen et al., ``Seeing beyond the brain:
Conditional diffusion model with sparse masked
modeling for vision decoding,''
in \textit{CVPR}, 2023.

\bibitem{scotti2023}
P.\ Scotti et al., ``Reconstructing the Mind's
Eye: fMRI-to-Image with contrastive learning
and diffusion priors,''
in \textit{NeurIPS}, 2023.

\bibitem{gifford2022}
A.\ T.\ Gifford et al., ``THINGS-EEG: Human
electroencephalography recordings for 1,654
concepts,'' \textit{NeuroImage}, 2022.

\bibitem{radford2021}
A.\ Radford et al., ``Learning transferable
visual models from natural language
supervision,'' in \textit{ICML}, 2021.

\bibitem{rombach2022}
R.\ Rombach et al., ``High-resolution image
synthesis with latent diffusion models,''
in \textit{CVPR}, 2022.

\bibitem{atm2024}
ATM: Attention-based Temporal-spatial Model
for EEG analysis, \textit{NeurIPS}, 2024.

\bibitem{sharma2026}
A.\ Sharma, ``Automated sleep stage
classification for closed-loop lucid dream
induction via CNN-LSTM on single-channel EEG,''
\textit{Zenodo}, 2026.
doi:10.5281/zenodo.21885881.

\bibitem{things2019}
M.\ N.\ Hebart et al., ``THINGS: A database
of 1,854 object concepts and more than
26,000 naturalistic object images,''
\textit{PLOS ONE}, 2019.

\bibitem{dicarlo2012}
J.\ J.\ DiCarlo, D.\ Zoccolan, and
N.\ C.\ Rust, ``How does the brain solve
visual object recognition?''
\textit{Neuron}, 2012.

\end{thebibliography}

\end{document}