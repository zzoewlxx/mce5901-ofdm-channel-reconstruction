"""Build the proposal by replacing the body paragraphs of the teammate's original document.xml.

Styles, numbering, page setup and all other parts are copied unchanged (title metadata updated).
Usage:
    unzip docs/proposal/proposal_v1_original.docx -d <tmp_dir>
    python -I docs/proposal/build/build_proposal.py <tmp_dir> <out.docx> [--no-page-break]
Then render with Word and check that it is still two pages.
"""
import os
import re
import sys
import zipfile
from xml.sax.saxutils import escape

src_dir, out_path = sys.argv[1], sys.argv[2]
page_break = "--no-page-break" not in sys.argv

BULLET_PPR = ('<w:pPr><w:widowControl/><w:ind w:left="300" w:hanging="300"/><w:jc w:val="both"/>'
              '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="51"/></w:numPr></w:pPr>')


def t(text):
    sp = ' xml:space="preserve"' if text != text.strip() or "  " in text else ""
    return f"<w:t{sp}>{escape(text)}</w:t>"


def styled(style, text, extra=""):
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{extra}</w:pPr><w:r>{t(text)}</w:r></w:p>'


def plain(text, east_asia_font=None, size_half_points=None):
    rpr = ""
    if east_asia_font or size_half_points:
        rpr = "<w:rPr>"
        if east_asia_font:
            rpr += f'<w:rFonts w:eastAsia="{east_asia_font}"/>'
        if size_half_points:
            rpr += f'<w:sz w:val="{size_half_points}"/>'
        rpr += "</w:rPr>"
    return f"<w:p><w:r>{rpr}{t(text)}</w:r></w:p>"


MEMBERS = [("冯正阳", "225010324"), ("谢昊", "226010063"), ("散恒跃", "226010043"), ("仲培瑜", "225010390")]
TITLE = "OFDM Channel Reconstruction from Sparse Noisy Pilots Using Measured CSI"
REFERENCES = (
    "References: [1] S. Coleri et al., IEEE Trans. Broadcast. 48(3):223–229, 2002. [2] O. Edfors et al., "
    "IEEE Trans. Commun. 46(7):931–939, 1998. [3] M. Soltani et al. (ChannelNet), IEEE Commun. Lett. "
    "23(4):652–655, 2019. [4] L. Li et al. (ReEsNet), IEEE Wireless Commun. Lett. 9(5):615–618, 2020. "
    "[5] F. Euchner et al. (DICHASUS), WSA 2021.")


def bullet(lead, body):
    return (f"<w:p>{BULLET_PPR}<w:r><w:rPr><w:b/></w:rPr>{t(lead + ' ')}</w:r>"
            f"<w:r>{t(body)}</w:r></w:p>")


paras = [
    styled("Title", "Proposal: OFDM Channel Reconstruction"),
    styled("Heading1", "1. The names and student IDs of the members of the group (Group 12)"),
    *[plain(f"Member {i}: {name}    Student ID: {sid}", east_asia_font="SimSun")
      for i, (name, sid) in enumerate(MEMBERS, 1)],
    styled("Heading1", "2. Title for the project"),
    plain(TITLE),
    styled("Heading1", "3. Description of the dataset"),
    bullet("Dataset Name:",
           "CSI Dataset dichasus-005x: Indoor Non-Line of Sight, Single Room, a public set of real wireless "
           "channel measurements collected by the University of Stuttgart with the DICHASUS channel sounder."),
    bullet("Data Source and Access:",
           "Florian Euchner and Marc Gauger published the dataset on the DaRUS research data repository in 2021 "
           "under the CC BY 4.0 license (DOI: 10.18419/darus-2204, available at https://doi.org/10.18419/darus-2204). "
           "The repository provides five TFRecord files and a machine-readable specification of the measurement "
           "parameters."),
    bullet("Measurement Environment:",
           "A transmitter mounted on a vacuum robot moved through an indoor room in which a metallic barrier blocks "
           "the line of sight, so the data capture reflected paths of a real indoor environment. The system uses a "
           "1.272 GHz carrier frequency, 50 MHz bandwidth and 32 co-located receive antennas; transmitter positions "
           "come from the robot's LiDAR."),
    bullet("Sample Size and Structure:",
           "There are 26,161 snapshots from five complete robot runs, about 6.9 GB in total. Each snapshot contains "
           "a CSI tensor of shape (32, 1024, 2): 32 antennas, 1,024 OFDM subcarriers stored in FFT order, and the "
           "real and imaginary parts of each coefficient."),
    bullet("Key Features:",
           "The csi field holds the frequency-domain channel coefficients, snr an SNR estimate for each antenna, "
           "pos-lidar the two-dimensional transmitter position and time the timestamp. Estimated per-antenna phase "
           "and timing offsets of the sounder are published with the dataset and will be compensated."),
    bullet("Inputs and Reconstruction Target:",
           "We use one block of 64 subcarriers (3.1 MHz) inside the occupied band, away from the DC subcarrier and "
           "the band edges. Pilots occupy 16, 8 or 4 approximately uniformly spaced subcarriers, including both "
           "block ends. To emulate least-squares pilot estimates, complex Gaussian noise is added to the measured "
           "CSI at the pilot subcarriers, with a pilot SNR of 0–30 dB defined relative to the average channel "
           "power of the training data. The remaining subcarriers are the reconstruction targets, and the measured "
           "CSI serves as the reference."),
    bullet("Dataset Partition and Quality:",
           "Runs 0053–0055 (15,940 snapshots) are used for training, run 0056 (5,220) for validation and run "
           "0057 (5,001) for testing, so adjacent records of one run never fall into different sets. Each of the "
           "32 antennas of a snapshot gives a separate example. Vectors whose CSI error, estimated from fourth-order "
           "differences across frequency, exceeds −30 dB relative to the channel power are removed before "
           "fitting (about 0.5% in our check; the snr field alone did not identify them). All preprocessing "
           "parameters and covariance estimates are fitted on training data only."),
    styled("Heading1", "4. What Problems It Solves", "<w:pageBreakBefore/>" if page_break else ""),
    bullet("Core Problem:",
           "An OFDM receiver estimates the channel at a few pilot subcarriers and must infer it on the others; "
           "accuracy is limited by pilot noise and pilot spacing. We study how accurately a block of measured "
           "channel coefficients can be reconstructed from a few noisy pilots, i.e. the interpolation and denoising "
           "stage of pilot-based channel estimation."),
    bullet("Practical Significance:",
           "Pilots occupy resources that could carry data, so knowing the accuracy each pilot density achieves "
           "under realistic noise informs the pilot overhead. Classical receivers interpolate least-squares pilot "
           "estimates [1] or apply LMMSE with channel statistics [2]; neural estimators such as ChannelNet [3] and "
           "ReEsNet [4] were mostly evaluated on simulated channels, so measured channels [5] give a useful test. "
           "Our conclusions concern one measured environment, not a demonstrated pilot reduction in hardware."),
    bullet("Preliminary Check:",
           "The measured channel is very smooth in frequency (correlation about 0.97 across the whole "
           "64-subcarrier block): with noise-free observations, linear interpolation from 8 of 64 subcarriers "
           "already reached about −43 dB NMSE, close to the measurement error, so all methods looked alike. "
           "With noisy pilots they separate: on the full training runs and validation run 0056 (test run 0057 "
           "untouched), LMMSE beat linear interpolation by about 4–10 dB with 8 or 16 pilots over 0–30 dB "
           "pilot SNR (paired 95% CI about ±0.1 dB). We therefore study noisy pilots."),
    bullet("Research Question and Objectives:",
           "How do linear and cubic-spline interpolation, covariance-based LMMSE and a small neural network compare "
           "as the pilot SNR (0–30 dB) and pilot density (16, 8 or 4 of 64) vary? Because LMMSE with accurate "
           "statistics is the best linear estimator, we ask whether and when a nonlinear learned model adds value, "
           "for example at low SNR or when the noise level is not known exactly, and at what computational cost."),
    styled("Heading1", "5. Methods Will be Used"),
    bullet("Data Preprocessing:",
           "We read the TFRecords with a tested NumPy-only parser, apply fftshift so that subcarriers are in "
           "frequency order, compensate the antenna offsets and extract the block, chosen on training data and kept "
           "fixed. All channels are scaled by one constant from the training data; normalizing each snapshot by its "
           "own true power would reveal the channel norm to the estimators and, in our check, gave the network an "
           "apparent 1–2 dB advantage."),
    bullet("Conventional Baselines:",
           "Linear and natural cubic-spline interpolation need no training. LMMSE uses the block mean and covariance "
           "estimated from the training runs, the known noise variance and a small diagonal loading chosen on "
           "validation data; with sample statistics it equals ridge regression from the pilots to the withheld "
           "coefficients, linking it to the regularized regression covered in the course."),
    bullet("Learning Model:",
           "A multilayer perceptron with two hidden layers of 128 ReLU units maps the real and imaginary parts of "
           "the noisy pilots and the pilot SNR to the withheld coefficients (about 33,000 parameters). One model per "
           "pilot density is trained over pilot SNRs drawn uniformly from 0–30 dB with fresh noise each epoch, "
           "using mean-squared error, Adam and best-epoch selection on run 0056. We also test a loss weighted by the "
           "expected error at each SNR and a residual variant that only corrects the LMMSE estimate."),
    bullet("Experimental Evaluation:",
           "The primary measure is NMSE on withheld subcarriers (total squared error over total squared reference "
           "magnitude), plotted against pilot SNR for each pilot density on test run 0057, with identical pilots, "
           "noise realizations and test records for all methods. The final evaluation will use three training seeds, and 95% "
           "confidence intervals come from bootstrapping blocks of consecutive test snapshots. We also report "
           "parameter counts, CPU inference time and robustness when the assumed noise variance is wrong. If time "
           "permits, NMSE will be translated into bit error rate for QPSK and 16-QAM with zero-forcing equalization."),
    bullet("Expected Outcomes:",
           "We expect reproducible code and NMSE-versus-SNR curves for all methods. A full-data validation check "
           "(16 and 8 pilots, seed 0, 15 epochs) suggests LMMSE is a strong baseline: the MLP stayed within 0.2 dB "
           "of it up to 20 dB pilot SNR but was 1.0–1.5 dB worse at 30 dB; a weighted loss halved this gap, a "
           "3×256 network did not help, and the residual variant was level with LMMSE. The block covariance is "
           "concentrated in two modes (99.99% of the energy), which may explain this. Time permitting, we will "
           "test joint antenna–frequency reconstruction, where a nonlinear gain is a hypothesis, not a promise. Limitations: one indoor environment, synthetic "
           "pilot noise, and the reference CSI's own error (median about −43 dB)."),
    plain(REFERENCES, size_half_points=17),
]

doc_path = os.path.join(src_dir, "word", "document.xml")
xml = open(doc_path, encoding="utf-8").read()
head, rest = xml.split("<w:body>", 1)
sect = re.search(r"<w:sectPr\b.*?</w:sectPr>", rest, re.S).group(0)
new_xml = head + "<w:body>" + "".join(paras) + sect + "</w:body></w:document>"

words = sum(len(re.sub(r"<[^>]+>", " ", p).split()) for p in paras)
print("paragraphs:", len(paras), "| approx words:", words)

with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
    names = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            full = os.path.join(root, f)
            names.append(os.path.relpath(full, src_dir).replace(os.sep, "/"))
    names.sort(key=lambda n: (n != "[Content_Types].xml", n))
    for n in names:
        if n == "word/document.xml":
            z.writestr(n, new_xml.encode("utf-8"))
        elif n == "docProps/core.xml":
            core = open(os.path.join(src_dir, n), encoding="utf-8").read()
            core = re.sub(r"<dc:title>.*?</dc:title>", f"<dc:title>{escape(TITLE)}</dc:title>", core)
            z.writestr(n, core.encode("utf-8"))
        else:
            z.write(os.path.join(src_dir, n), n)
print("wrote", out_path)
