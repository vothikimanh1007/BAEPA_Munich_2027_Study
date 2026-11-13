import os
import glob
import json
import requests
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


def load_dharma_epigraphy(dharma_xml_dir: str) -> pd.DataFrame:
    """Parse local TEI-XML inscriptions from the ERC DHARMA repository."""
    xml_files = sorted(glob.glob(os.path.join(dharma_xml_dir, "**", "*.xml"), recursive=True))
    records = []
    ns = {"tei": "[http://www.tei-c.org/ns/1.0](http://www.tei-c.org/ns/1.0)"}

    for fp in xml_files:
        try:
            tree = ET.parse(fp)
            root = tree.getroot()
            filename = os.path.splitext(os.path.basename(fp))[0]

            body_node = root.find(".//tei:body", ns)
            body_text = " ".join(body_node.itertext()) if body_node is not None else ""
            body_text = " ".join(body_text.split())

            if len(body_text) <= 35:
                all_text = " ".join([elem.text for elem in root.iter() if elem.text])
                body_text = " ".join(all_text.split())

            if len(body_text) > 35:
                title_node = root.find(".//tei:titleStmt/tei:title", ns)
                title = (
                    title_node.text.strip()
                    if title_node is not None and title_node.text
                    else filename
                )
                orig_date = root.find(".//tei:origDate", ns)
                date_str = (
                    orig_date.attrib.get("notBefore", "850")
                    if orig_date is not None
                    else "850"
                )
                try:
                    dating_year = float("".join(ch for ch in date_str if ch.isdigit())[:4] or 850)
                except ValueError:
                    dating_year = 850.0

                places = [p.text.strip() for p in root.findall(".//tei:placeName", ns) if p.text]
                persons = [p.text.strip() for p in root.findall(".//tei:persName", ns) if p.text]

                records.append({
                    "artifact_id": filename,
                    "corpus_partition": "DHARMA",
                    "source": "DHARMA_SEA_Epigraphy",
                    "title": title[:60].replace("\n", " "),
                    "region": "Maritime Southeast Asia",
                    "period": date_str,
                    "dating_year": dating_year,
                    "material": "Stone stela / Copper plate",
                    "text_content": f"Southeast Asian epigraph {title} {' '.join(places + persons)} {body_text[:1500]}",
                })
        except Exception:
            continue

    return pd.DataFrame(records)


def load_va_sculptures(va_json_dir: str, fetch_remote: bool = False) -> pd.DataFrame:
    """Load cached V&A portable bronze records or fetch from V&A REST Open API v2."""
    os.makedirs(va_json_dir, exist_ok=True)
    va_files = sorted(glob.glob(os.path.join(va_json_dir, "*.json")))
    records = []

    if len(va_files) == 0 and fetch_remote:
        headers = {"User-Agent": "BAEPA2027Study/1.0"}
        va_url = "[https://api.vam.ac.uk/v2/objects/search?q=Buddha&material=bronze&page_size=100](https://api.vam.ac.uk/v2/objects/search?q=Buddha&material=bronze&page_size=100)"
        resp = requests.get(va_url, headers=headers, timeout=20)
        if resp.status_code == 200:
            for item in resp.json().get("records", []):
                sys_num = item.get("systemNumber", str(np.random.randint(100000, 999999)))
                with open(os.path.join(va_json_dir, f"va_{sys_num}.json"), "w", encoding="utf-8") as jf:
                    json.dump(item, jf, ensure_ascii=False, indent=2)
        va_files = sorted(glob.glob(os.path.join(va_json_dir, "*.json")))

    for jf in va_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                item = json.load(f)
            sys_id = str(item.get("systemNumber", item.get("objectNumber", os.path.splitext(os.path.basename(jf))[0])))
            title = str(item.get("_primaryTitle", item.get("objectType", "Buddhist portable bronze")))
            place = str(item.get("_primaryPlace", "Himalaya / South Asia"))
            desc = str(item.get("summaryDescription", ""))
            records.append({
                "artifact_id": sys_id,
                "corpus_partition": "VA",
                "source": "VA_Portable_Sculpture",
                "title": title[:60].replace("\n", " "),
                "region": place,
                "period": str(item.get("_primaryDate", "Historic")),
                "dating_year": 900.0,
                "material": "Cast copper alloy / Bronze",
                "text_content": f"V&A portable Buddhist sculpture bronze icon {title} {place} {desc} devotion pilgrimage consecration",
            })
        except Exception:
            continue

    return pd.DataFrame(records)


def load_cma_sculptures(cma_json_dir: str, fetch_remote: bool = False) -> pd.DataFrame:
    """Load cached CMA Asian sculpture records or fetch from CMA Open Access API."""
    os.makedirs(cma_json_dir, exist_ok=True)
    cma_files = sorted(glob.glob(os.path.join(cma_json_dir, "*.json")))
    records = []

    if len(cma_files) == 0 and fetch_remote:
        cma_url = "[https://openaccess-api.clevelandart.org/api/artworks/?q=Buddha&has_image=1&limit=100](https://openaccess-api.clevelandart.org/api/artworks/?q=Buddha&has_image=1&limit=100)"
        resp = requests.get(cma_url, timeout=20)
        if resp.status_code == 200:
            for item in resp.json().get("data", []):
                item_id = item.get("id", np.random.randint(100000, 999999))
                with open(os.path.join(cma_json_dir, f"cma_{item_id}.json"), "w", encoding="utf-8") as jf:
                    json.dump(item, jf, ensure_ascii=False, indent=2)
        cma_files = sorted(glob.glob(os.path.join(cma_json_dir, "*.json")))

    for jf in cma_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                item = json.load(f)
            acc_id = str(item.get("accession_number", item.get("id", os.path.splitext(os.path.basename(jf))[0])))
            title = str(item.get("title", "Buddhist Sculpture"))
            culture = item.get("culture", ["Himalayan / East Asian"])
            culture_str = culture[0] if isinstance(culture, list) and culture else str(culture)
            technique = str(item.get("technique", "Gilt bronze / stone"))
            desc = str(item.get("description", ""))
            records.append({
                "artifact_id": acc_id,
                "corpus_partition": "CMA",
                "source": "CMA_Asian_Sculpture",
                "title": title[:60].replace("\n", " "),
                "region": culture_str,
                "period": str(item.get("creation_date", "Historic")),
                "dating_year": 950.0,
                "material": technique,
                "text_content": f"Cleveland Museum portable Buddhist sculpture {title} {culture_str} {technique} {desc} devotion consecration",
            })
        except Exception:
            continue

    return pd.DataFrame(records)


def build_unified_corpus(base_data_dir: str = "../data", fetch_remote: bool = False) -> pd.DataFrame:
    raw_dir = os.path.join(base_data_dir, "raw_data")
    processed_dir = os.path.join(base_data_dir, "processed_data")
    os.makedirs(processed_dir, exist_ok=True)

    df_dharma = load_dharma_epigraphy(os.path.join(raw_dir, "dharma_xml"))
    df_va = load_va_sculptures(os.path.join(raw_dir, "va_sculptures"), fetch_remote=fetch_remote)
    df_cma = load_cma_sculptures(os.path.join(raw_dir, "cma_sculptures"), fetch_remote=fetch_remote)

    df_unified = pd.concat([df_dharma, df_va, df_cma], ignore_index=True)
    out_csv = os.path.join(processed_dir, "unified_buddhist_corpus_tripartite.csv")
    df_unified.to_csv(out_csv, index=False)

    print(f"[01_corpus_ingestion] Unified corpus saved to {out_csv} (N = {len(df_unified)})")
    print(df_unified["corpus_partition"].value_counts())
    return df_unified


if __name__ == "__main__":
    build_unified_corpus(base_data_dir="../data", fetch_remote=False)
