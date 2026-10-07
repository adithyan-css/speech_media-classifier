"""Select the open-source speech / music corpus used by the ai_rf2 proof of concept.

Run once, on a machine that has a partial clone of microsoft/DNS-Challenge
(branch interspeech2021/adddata), to regenerate ``data/sources.csv``.  Most users
never need this: the CSV is committed and ``scripts/download_data.py`` reads it.

Every row carries a ``group`` (speaker, singer, actor, article or artist bucket)
and the split is assigned per group, never per clip, so no speaker / singer /
GTZAN artist appears in more than one split (TRD 7.3 spirit: split by source,
never by window).

Usage:
    git clone --depth 1 --filter=blob:none --no-checkout \
        -b interspeech2021/adddata https://github.com/microsoft/DNS-Challenge.git dns
    (cd dns && git ls-tree -r HEAD --name-only datasets/wideband/clean_wideband \
        | grep '\\.wav$' > ../dns_clean.txt)
    (cd dns && git ls-tree -r HEAD --name-only datasets/wideband/noise_wideband \
        | grep '\\.wav$' > ../dns_noise.txt)
    python scripts/select_sources.py --dns-clean dns_clean.txt --dns-noise dns_noise.txt \
        --gtzan-split-dir <dir with coreyker/dnn-mgr gtzan/*_filtered.txt> --out data/sources.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

DNS_MEDIA = "https://media.githubusercontent.com/media/microsoft/DNS-Challenge/interspeech2021/adddata/"
GTZAN_RAW = "https://raw.githubusercontent.com/tlin41390/MusicClassification/main/Data/genres_original/"
SEED = 20260924
#: jazz.00054 is a known-corrupt GTZAN file (not a readable wav in any mirror).
GTZAN_CORRUPT = {"jazz/jazz.00054.wav"}

FIELDS = ["uid", "url", "label", "subtype", "source", "language", "group", "split",
          "license", "byte_start", "byte_len"]


def bucket(key: str, test: float = 0.2, dev: float = 0.1) -> str:
    """Deterministic group -> split assignment by hash."""
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test:
        return "test"
    if h < test + dev:
        return "dev"
    return "train"


def dns_url(path: str) -> str:
    return DNS_MEDIA + quote(path, safe="/")


def uid_of(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()[:12]


def contiguous(files: list[str], n: int, rng: random.Random) -> list[str]:
    files = sorted(files)
    if len(files) <= n:
        return files
    start = rng.randrange(0, len(files) - n)
    return files[start:start + n]


def row(path_or_url, label, subtype, source, language, group, split, license_, url=None,
        byte_start="", byte_len=""):
    return {
        "uid": uid_of(path_or_url + str(byte_start)),
        "url": url or dns_url(path_or_url),
        "label": label, "subtype": subtype, "source": source, "language": language,
        "group": group, "split": split, "license": license_,
        "byte_start": byte_start, "byte_len": byte_len,
    }


def select(dns_clean: list[str], dns_noise: list[str], gtzan_split: dict[str, str]) -> list[dict]:
    rng = random.Random(SEED)
    rows: list[dict] = []

    # 1. English read speech (LibriVox via DNS), 31 s clips, group = reader id.
    readers = defaultdict(list)
    for p in dns_clean:
        if "/read_speech/" in p:
            m = re.search(r"reader_(\d+)_", p)
            readers[m.group(1)].append(p)
    for reader in rng.sample(sorted(readers), 120):
        for p in contiguous(readers[reader], 3, rng):
            rows.append(row(p, "SPEECH", "read_audiobook", "librivox_dns", "en",
                            f"en_reader_{reader}", bucket(f"en_reader_{reader}"), "CC-BY-4.0 (DNS)"))

    # 2. M-AILABS audiobooks (de/fr/it/es/ru). group = speaker; 'mix' books keep book as group.
    mailabs = defaultdict(list)  # (lang, speaker, book) -> files
    for p in dns_clean:
        if "M-AILABS" not in p and "M-AILabs" not in p:
            continue
        lang = {"german_speech": "de", "french_data": "fr", "italian_speech": "it",
                "spanish_speech": "es", "russian_speech": "ru"}[p.split("/")[3]]
        parts = p.split("/")
        if lang == "es":
            book = re.sub(r"_f\d+_16kHz\.wav$", "", parts[-1].split("wavs_", 1)[-1])
            author = book.rsplit("_", 1)[-1]
            speaker, book_key = f"es_author_{author}", book
        else:
            i = [k for k, s in enumerate(parts) if s.endswith("_16k")][0]
            gender, name, book_key = parts[i + 1], parts[i + 2], parts[i + 3]
            speaker = f"{lang}_{name}" if gender != "mix" else f"{lang}_mix_{name}"
        mailabs[(lang, speaker, book_key)].append(p)

    # fixed, speaker-disjoint test / dev speakers; Russian is held out entirely (unseen language)
    fixed = {
        "fr_bernard": "test", "fr_zeckou": "dev",
        "it_lisa_caputo": "test",
        "de_karlsson": "test", "de_angela_merkel": "test",
        "es_author_perezgaldos": "test", "es_author_dumas": "dev",
    }
    per_lang_books = defaultdict(list)
    for key in sorted(mailabs):
        per_lang_books[key[0]].append(key)
    for lang, keys in per_lang_books.items():
        rng.shuffle(keys)
        n_books = {"de": 14, "fr": 10, "it": 10, "es": 16, "ru": 6}[lang]
        clips_per_book = {"de": 70, "fr": 70, "it": 70, "es": 60, "ru": 70}[lang]
        # make sure the fixed test/dev speakers are represented
        chosen = [k for k in keys if k[1] in fixed][: n_books // 2]
        chosen += [k for k in keys if k not in chosen][: n_books - len(chosen)]
        for (lg, speaker, book) in chosen:
            if lg == "ru":
                split = "test"
            elif speaker in fixed:
                split = fixed[speaker]
            elif "_mix_" in speaker:
                split = "train"
            else:
                split = "train"
            subtype = "political_speech" if "merkel" in speaker else "read_audiobook"
            for p in contiguous(mailabs[(lg, speaker, book)], clips_per_book, rng):
                rows.append(row(p, "SPEECH", subtype, "m-ailabs_dns", lg, speaker, split,
                                "M-AILABS licence (BSD-style)"))

    # 3. Spoken Wikipedia (German), long articles; take a 150 s byte range from each.
    wiki = sorted(p for p in dns_clean if "German_Wikipedia" in p)
    bytes_per_s = 32000  # 16 kHz, 16-bit mono
    for p in rng.sample(wiki, 90):
        art = Path(p).stem
        rows.append(row(p, "SPEECH", "read_encyclopedia", "spoken_wikipedia_de", "de",
                        f"de_wiki_{uid_of(art)}", bucket(f"de_wiki_{art}"), "CC-BY-SA-4.0",
                        byte_start=44 + 30 * bytes_per_s, byte_len=150 * bytes_per_s))

    # 4. TUDA German read sentences, Samson close mic; official train/test partitions.
    tuda = [p for p in dns_clean if "german-speechdata" in p and "Samson" in p]
    tuda_train = [p for p in tuda if p.endswith("_train_16k.wav")]
    tuda_test = [p for p in tuda if p.endswith("_test_16k.wav")]
    for p in rng.sample(sorted(tuda_train), 400):
        rows.append(row(p, "SPEECH", "read_sentence", "tuda_de", "de", "de_tuda_train", "train",
                        "CC-BY-4.0"))
    for p in rng.sample(sorted(tuda_test), 150):
        rows.append(row(p, "SPEECH", "read_sentence", "tuda_de", "de", "de_tuda_test", "test",
                        "CC-BY-4.0"))

    # 5. Forschergeist podcast: conversational German, 2 speakers -> test only (style shift).
    podcast = sorted(p for p in dns_clean if "Forschergeist" in p)
    for p in contiguous(podcast, 200, rng):
        rows.append(row(p, "SPEECH", "podcast_conversation", "forschergeist_de", "de",
                        "de_forschergeist", "test", "CC-BY-SA-4.0"))

    # 6. CREMA-D acted emotional speech (English), group = actor.
    crema = defaultdict(list)
    for p in dns_clean:
        if "/crema_d/" in p:
            crema[Path(p).name[:4]].append(p)
    for actor in sorted(crema):
        for p in rng.sample(sorted(crema[actor]), min(5, len(crema[actor]))):
            rows.append(row(p, "SPEECH", "acted_emotional", "crema_d_dns", "en",
                            f"en_crema_{actor}", bucket(f"crema_{actor}"), "ODbL (CREMA-D)"))

    # 7. VocalSet singing (a cappella) -> MEDIA, group = singer.
    vocal = defaultdict(list)
    for p in dns_clean:
        if "/singing_voice/" in p:
            parts = p.split("/")
            vocal[(parts[6], parts[7])].append(p)
    singer_split = {"female9": "test", "male10": "test", "male11": "test",
                    "female8": "dev", "male9": "dev"}
    quota = {"excerpts": 7, "arpeggios": 10, "scales": 10, "long_tones": 5}
    for (singer, kind), files in sorted(vocal.items()):
        for p in rng.sample(sorted(files), min(quota.get(kind, 0), len(files))):
            # VocalSet 'excerpts/spoken' are the singers *speaking* the lyrics: SPEECH.
            spoken = "/excerpts/spoken/" in p
            rows.append(row(p, "SPEECH" if spoken else "MEDIA",
                            "spoken_lyrics" if spoken else f"singing_{kind}", "vocalset_dns",
                            "none", f"singer_{singer}", singer_split.get(singer, "train"),
                            "CC-BY-4.0"))

    # 8. GTZAN genres, fault-filtered artist-disjoint split (Kereliuk et al. 2015).
    for rel, split in sorted(gtzan_split.items()):
        if rel in GTZAN_CORRUPT:
            continue
        genre = rel.split("/")[0]
        rows.append(row(rel, "MEDIA", f"music_{genre}", "gtzan", "none", f"gtzan_{split}", split,
                        "research use (GTZAN)", url=GTZAN_RAW + rel))

    # 9. Freesound noise from DNS (named clips only; AudioSet ids are unlabelled) -> MEDIA.
    fs = defaultdict(list)
    for p in dns_noise:
        name = Path(p).name
        m = re.match(r"(.+?)_Freesound_validated_(\d+)", name)
        if m:
            fs[(m.group(1), m.group(2))].append(p)
    keys = sorted(fs)
    for key in rng.sample(keys, min(150, len(keys))):
        for p in rng.sample(sorted(fs[key]), min(2, len(fs[key]))):
            rows.append(row(p, "MEDIA", f"noise_{key[0]}", "freesound_dns", "none",
                            f"fs_{key[1]}", bucket(f"fs_{key[1]}"), "CC (Freesound, per clip)"))

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dns-clean", required=True)
    ap.add_argument("--dns-noise", required=True)
    ap.add_argument("--gtzan-split-dir", required=True)
    ap.add_argument("--out", default="data/sources.csv")
    a = ap.parse_args()
    dns_clean = [l for l in Path(a.dns_clean).read_text().splitlines() if l.strip()]
    dns_noise = [l for l in Path(a.dns_noise).read_text().splitlines() if l.strip()]
    gtzan_split = {}
    for name, split in [("train", "train"), ("valid", "dev"), ("test", "test")]:
        for line in (Path(a.gtzan_split_dir) / f"{name}_filtered.txt").read_text().split():
            gtzan_split[line.strip()] = split
    rows = select(dns_clean, dns_noise, gtzan_split)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(len(rows), "rows")
    print(Counter((r["label"], r["split"]) for r in rows))


if __name__ == "__main__":
    main()
