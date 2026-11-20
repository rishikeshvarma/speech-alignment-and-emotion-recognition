"""Sanity-check the MFA runs in outputs/ and compare them with each other.

There is no hand-labelled reference alignment for this data, so quality is
judged in three ways:

1. MFA's own diagnostics (alignment_analysis.csv): speech log-likelihood and
   phone-duration deviation (how far phone lengths are from the model's
   typical lengths, in standard deviations).
2. Agreement between runs: the same word should start and end at about the
   same time whichever model aligned it. Differences are measured against
   the arpa_g2p run; 20 ms is the usual tolerance in alignment papers.
3. Red flags in the TextGrids: words with no pronunciation (spn), phones
   squeezed to the 30 ms minimum (a sign the aligner ran out of time), very
   short words and long pauses inside an utterance.

Writes CSV tables and summary.md to --results.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import statistics as st
from pathlib import Path

from textgrid_utils import UNKNOWN, read_textgrid, textgrids, phones_in

REFERENCE = "arpa_g2p"
MIN_PHONE = 0.031        # 3 frames x 10 ms, the shortest a 3-state HMM phone can be
SHORT_WORD = 0.08
LONG_PAUSE = 0.30
FOCUS = ["dukakis", "maffy", "melnicove", "margo", "politicize", "judgeships",
         "white", "bait", "new", "no", "bad", "bed"]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "_none_\n"
    keys = list(rows[0])
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    out += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(out) + "\n"


def mfa_diagnostics(run_dir: Path) -> dict[str, float]:
    f = run_dir / "alignment_analysis.csv"
    if not f.exists():
        return {}
    with open(f, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    def mean(col: str) -> float:
        vals = [float(r[col]) for r in rows if r.get(col) not in (None, "", "nan")]
        return round(st.mean(vals), 2) if vals else float("nan")

    return {"speech_loglik": mean("speech_log_likelihood"),
            "phone_dur_dev": mean("phone_duration_deviation")}


def match_words(a, b):
    """Pairs of word intervals with the same label, in order."""
    sm = difflib.SequenceMatcher(a=[w.label for w in a], b=[w.label for w in b], autojunk=False)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            yield a[blk.a + k], b[blk.b + k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--corpus", type=Path, default=Path("corpus"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    runs = textgrids(args.outputs)
    n_utts = len(list(args.corpus.rglob("*.lab")))
    tgs = {exp: {utt: read_textgrid(p) for utt, p in utts.items()} for exp, utts in runs.items()}
    tgs_sil = {exp: {utt: read_textgrid(p, keep_silence=True) for utt, p in utts.items()}
               for exp, utts in runs.items()}

    summary, anomalies, focus_rows, word_rows = [], [], [], []
    for exp, utts in tgs.items():
        words = [w for tg in utts.values() for w in tg["words"]]
        phones = [p for tg in utts.values() for p in tg["phones"]]
        n_unk_words = 0
        for utt, tg in utts.items():
            for i, w in enumerate(tg["words"]):
                ph = phones_in(w, tg["phones"])
                unk = w.label in UNKNOWN or any(p.label in UNKNOWN for p in ph)
                n_unk_words += unk
                word_rows.append({"experiment": exp, "utterance": utt, "index": i, "word": w.label,
                                  "start": round(w.start, 3), "end": round(w.end, 3),
                                  "phones": " ".join(p.label for p in ph)})
                if unk:
                    anomalies.append({"experiment": exp, "utterance": utt, "type": "no pronunciation (spn)",
                                      "label": w.label, "start": round(w.start, 3),
                                      "dur_ms": round(w.dur * 1000)})
                elif w.dur < SHORT_WORD and len(ph) > 1:
                    anomalies.append({"experiment": exp, "utterance": utt, "type": f"word < {SHORT_WORD*1000:.0f} ms",
                                      "label": f"{w.label} [{' '.join(p.label for p in ph)}]",
                                      "start": round(w.start, 3), "dur_ms": round(w.dur * 1000)})
                if w.label in FOCUS:
                    focus_rows.append({"experiment": exp, "utterance": utt, "word": w.label,
                                       "start": round(w.start, 3), "dur_ms": round(w.dur * 1000),
                                       "phones (ms)": " ".join(f"{p.label}({p.dur*1000:.0f})" for p in ph)})
            for p in tg["phones"]:
                if p.dur <= MIN_PHONE and p.label not in UNKNOWN:
                    anomalies.append({"experiment": exp, "utterance": utt, "type": "phone at 30 ms minimum",
                                      "label": p.label, "start": round(p.start, 3),
                                      "dur_ms": round(p.dur * 1000)})
            # pauses between words (silence intervals that are not at the edges)
            seq = tgs_sil[exp][utt]["words"]
            for k in range(1, len(seq) - 1):
                iv = seq[k]
                if iv.label == "" and iv.dur >= LONG_PAUSE:
                    anomalies.append({"experiment": exp, "utterance": utt, "type": f"pause >= {LONG_PAUSE*1000:.0f} ms",
                                      "label": f"{seq[k-1].label} _ {seq[k+1].label}",
                                      "start": round(iv.start, 3), "dur_ms": round(iv.dur * 1000)})

        pd = [p.dur * 1000 for p in phones if p.label not in UNKNOWN]
        n_short = sum(d <= MIN_PHONE * 1000 for d in pd)
        summary.append({"experiment": exp, "aligned": f"{len(utts)}/{n_utts}", "words": len(words),
                        "words_without_pron": n_unk_words, "phones": len(phones),
                        "mean_phone_ms": round(st.mean(pd), 1), "median_phone_ms": round(st.median(pd), 1),
                        "phones_at_30ms": f"{n_short} ({100*n_short/len(pd):.1f}%)",
                        **mfa_diagnostics(args.outputs / exp)})

    # boundary agreement against the reference run
    agreement = []
    if REFERENCE in tgs:
        for exp in tgs:
            if exp == REFERENCE:
                continue
            for spk_filter, label in ((None, "all"), ("F2B", "f2b"), ("ISLE", "isle")):
                d = []
                for utt, tg in tgs[exp].items():
                    if spk_filter and not utt.upper().startswith(spk_filter):
                        continue
                    ref = tgs[REFERENCE].get(utt)
                    if ref is None:
                        continue
                    for a, b in match_words(ref["words"], tg["words"]):
                        d += [abs(a.start - b.start), abs(a.end - b.end)]
                if not d:
                    continue
                d_ms = [x * 1000 for x in d]
                agreement.append({"compared_to_" + REFERENCE: exp, "speaker": label,
                                  "boundaries": len(d_ms), "mean_ms": round(st.mean(d_ms), 1),
                                  "median_ms": round(st.median(d_ms), 1),
                                  "max_ms": round(max(d_ms)),
                                  "within_20ms": f"{100*sum(x <= 20 for x in d_ms)/len(d_ms):.0f}%",
                                  "within_50ms": f"{100*sum(x <= 50 for x in d_ms)/len(d_ms):.0f}%"})

    write_csv(args.results / "alignment_summary.csv", summary)
    write_csv(args.results / "boundary_agreement.csv", agreement)
    write_csv(args.results / "anomalies.csv", anomalies)
    write_csv(args.results / "focus_words.csv", focus_rows)
    write_csv(args.results / "all_words.csv", word_rows)

    # Pronunciation check (isle_variants run): which listed variant did the aligner keep?
    variant_rows = []
    vdict = Path(__file__).resolve().parents[1] / "dictionaries" / "isle_variants.dict"
    if "isle_variants" in tgs and vdict.exists():
        variants: dict[str, list[str]] = {}
        for line in vdict.read_text(encoding="utf-8").splitlines():
            if line.strip():
                w, pron = line.split("\t", 1)
                variants.setdefault(w, []).append(pron.strip())
        for utt, tg in sorted(tgs["isle_variants"].items()):
            for w in tg["words"]:
                chosen = " ".join(p.label for p in phones_in(w, tg["phones"]))
                canon = variants.get(w.label, ["?"])[0]
                variant_rows.append({"utterance": utt[-12:], "word": w.label, "canonical": canon,
                                     "aligner chose": chosen,
                                     "verdict": "canonical" if chosen == canon else "LEARNER VARIANT",
                                     "other options": " / ".join(v for v in variants.get(w.label, [])[1:])})
        write_csv(args.results / "isle_pronunciation_check.csv", variant_rows)

    counts: dict[tuple, int] = {}
    for a in anomalies:
        counts[(a["experiment"], a["type"])] = counts.get((a["experiment"], a["type"]), 0) + 1
    anomaly_counts = [{"experiment": e, "type": t, "count": c} for (e, t), c in sorted(counts.items())]

    md = ["# Alignment analysis\n",
          "## Runs\n", md_table(summary),
          f"\n## Word-boundary agreement with `{REFERENCE}`\n", md_table(agreement),
          "\n## Red flags (counts)\n", md_table(anomaly_counts),
          "\n## ISLE pronunciation check (variant chosen by the aligner)\n", md_table(variant_rows),
          "\n## Focus words (G2P words and ISLE minimal pairs)\n",
          md_table(sorted(focus_rows, key=lambda r: (r["word"], r["utterance"], r["experiment"])))]
    (args.results / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
