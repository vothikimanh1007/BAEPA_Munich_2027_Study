"""
01_corpus_ingestion.py
----------------------
Part of: Reconstructing Inter-Regional Transmission and Material Bridges in Asian Buddhist Epigraphy and Portable Icons
Repository: BAEPA_Munich_2027_Study
"""

import os
import re
import glob
import json
import requests
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


def extract_year_from_string(s):
    if not s or str(s).lower() in ["unknown", "none", "nan", "historic", ""]:
        return np.nan
    s_str = str(s)
    m_yr = re.findall(r'\b([5-9]\d{2}|1[0-4]\d{2})\b', s_str)
    if m_yr:
        return float(np.mean([int(x) for x in m_yr]))
    m_cent = re.findall(r'\b([6-9]|1[0-4])(?:th|st|nd|rd)\b', s_str.lower())
    if m_cent:
        cents = [(int(c) - 1) * 100 + 50 for c in m_cent]
        return float(np.mean(cents))
    return np.nan


def classify_iconography(text_blob):
    t = text_blob.lower()
    if any(k in t for k in ["avalokiteshvara", "avalokiteśvara", "padmapani", "padmapāṇi", "lokeshvara", "amoghapasa", "tara", "tārā", "maitreya", "manjushri", "bodhisattva"]):
        return "Bodhisattva_Cult"
    if any(k in t for k in ["medicine", "bhaisajyaguru", "bhaiṣajyaguru", "healing", "dharani", "dhāraṇī", "protective", "vajrapani", "tantric", "esoteric"]):
        return "Protective_Esoteric"
    if any(k in t for k in ["reliquary", "stupa", "stūpa", "votive", "tsatsa", "ye dharma", "ye dharmā", "creed", "offering", "monk"]):
        return "Ritual_Votive_Relic"
    if any(k in t for k in ["sima", "sīmā", "sasana", "śāsana", "endowment", "charter", "land", "grant", "temple", "punya", "puṇya"]):
        return "Monastic_Endowment"
    return "Buddha_Canonical"


def classify_macro_region(region_str, partition):
    r = str(region_str).lower()
    if partition == "DHARMA" or any(k in r for k in ["java", "sumatra", "bali", "vietnam", "champa", "campa", "cambodia", "thailand", "malay", "indonesia", "southeast asia"]):
        return "Maritime_Mainland_SEA"
    if any(k in r for k in ["tibet", "nepal", "himalaya", "kashmir", "swat", "ladakh"]):
        return "Trans_Himalayan"
    if any(k in r for k in ["india", "nalanda", "bihar", "bengal", "pala", "gandhara", "south asia"]):
        return "South_Asia_Core"
    if any(k in r for k in ["china", "korea", "japan", "east asia", "dunhuang"]):
        return "East_Asia"
    return "Unassigned_Contact_Zone"


def classify_material_clean(mat_str, partition):
    m = str(mat_str).lower()
    if partition == "DHARMA":
        if "copper" in m or "bronze" in m or "metal" in m:
            return "Inscribed_Copper_Plate"
        return "Lithic_Stone_Stele"
    if "gilt" in m or "gold" in m:
        return "Gilt_Copper_Alloy"
    if "brass" in m:
        return "Cast_Brass"
    if "stone" in m or "schist" in m or "stucco" in m or "clay" in m:
        return "Portable_Stone_Clay"
    return "Cast_Bronze_Copper"


def load_dharma_epigraphy(dharma_xml_dir: str) -> pd.DataFrame:
    xml_files = sorted(glob.glob(os.path.join(dharma_xml_dir, "**", "*.xml"), recursive=True))
    records = []
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    for fp in xml_files:
        try:
            tree = ET.parse(fp)
            root = tree.getroot()
            filename = os.path.splitext(os.path.basename(fp))[0]

            body_node = root.find(".//tei:body", ns)
            body_text = " ".join(body_node.itertext()) if body_node is not None else " ".join([e.text for e in root.iter() if e.text])
            body_text = " ".join(body_text.split())

            if len(body_text) > 35:
                title_node = root.find(".//tei:titleStmt/tei:title", ns)
                title = title_node.text.strip() if (title_node is not None and title_node.text) else filename
                orig_date = root.find(".//tei:origDate", ns)
                date_str = orig_date.attrib.get("notBefore", "850") if orig_date is not None else "850"
                year_val = extract_year_from_string(date_str)
                if np.isnan(year_val):
                    year_val = 850.0

                places = [p.text.strip() for p in root.findall(".//tei:placeName", ns) if p.text]
                raw_region = places[0] if places else "Maritime Southeast Asia"

                records.append({
                    "artifact_id": filename,
                    "partition": "DHARMA",
                    "source": "DHARMA_SEA_Epigraphy",
                    "title": title[:60].replace("\n", " "),
                    "macro_region": classify_macro_region(raw_region, "DHARMA"),
                    "year_ce": year_val,
                    "is_dated": not np.isnan(year_val),
                    "material_clean": classify_material_clean("stone", "DHARMA"),
                    "iconography_class": classify_iconography(f"{title} {body_text[:500]}"),
                })
        except Exception:
            continue
    return pd.DataFrame(records)


def load_va_sculptures(va_json_dir: str) -> pd.DataFrame:
    os.makedirs(va_json_dir, exist_ok=True)
    va_files = sorted(glob.glob(os.path.join(va_json_dir, "*.json")))
    records = []

    for jf in va_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                item = json.load(f)
            sys_id = str(item.get("systemNumber", item.get("objectNumber", os.path.splitext(os.path.basename(jf))[0])))
            title = str(item.get("_primaryTitle", item.get("objectType", "Buddhist portable bronze")))
            place = str(item.get("_primaryPlace", "Himalaya / South Asia"))
            raw_date = str(item.get("_primaryDate", ""))
            year_val = extract_year_from_string(raw_date)
            mat_raw = str(item.get("materials", "bronze"))

            records.append({
                "artifact_id": sys_id,
                "partition": "VA",
                "source": "VA_Portable_Sculpture",
                "title": title[:60].replace("\n", " "),
                "macro_region": classify_macro_region(place, "VA"),
                "year_ce": year_val,
                "is_dated": not np.isnan(year_val),
                "material_clean": classify_material_clean(mat_raw, "VA"),
                "iconography_class": classify_iconography(f"{title}"),
            })
        except Exception:
            continue
    return pd.DataFrame(records)


def load_cma_sculptures(cma_json_dir: str) -> pd.DataFrame:
    os.makedirs(cma_json_dir, exist_ok=True)
    cma_files = sorted(glob.glob(os.path.join(cma_json_dir, "*.json")))
    records = []

    for jf in cma_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                item = json.load(f)
            acc_id = str(item.get("accession_number", item.get("id", os.path.splitext(os.path.basename(jf))[0])))
            title = str(item.get("title", "Buddhist Sculpture"))
            culture = item.get("culture", ["Himalayan / East Asian"])
            culture_str = culture[0] if isinstance(culture, list) and culture else str(culture)
            raw_date = str(item.get("creation_date", ""))
            year_val = extract_year_from_string(raw_date)
            tech = str(item.get("technique", "gilt bronze"))

            records.append({
                "artifact_id": acc_id,
                "partition": "CMA",
                "source": "CMA_Asian_Sculpture",
                "title": title[:60].replace("\n", " "),
                "macro_region": classify_macro_region(culture_str, "CMA"),
                "year_ce": year_val,
                "is_dated": not np.isnan(year_val),
                "material_clean": classify_material_clean(tech, "CMA"),
                "iconography_class": classify_iconography(f"{title}"),
            })
        except Exception:
            continue
    return pd.DataFrame(records)


def build_unified_corpus(base_data_dir: str = "../data") -> pd.DataFrame:
    raw_dir = os.path.join(base_data_dir, "raw_data")
    processed_dir = os.path.join(base_data_dir, "processed_data")
    os.makedirs(processed_dir, exist_ok=True)

    df_dharma = load_dharma_epigraphy(os.path.join(raw_dir, "dharma_xml"))
    df_va = load_va_sculptures(os.path.join(raw_dir, "va_sculptures"))
    df_cma = load_cma_sculptures(os.path.join(raw_dir, "cma_sculptures"))

    df_unified = pd.concat([df_dharma, df_va, df_cma], ignore_index=True)
    df_unified["year_ce"] = df_unified["year_ce"].fillna(900.0)
    out_csv = os.path.join(processed_dir, "unified_buddhist_corpus_tripartite.csv")
    df_unified.to_csv(out_csv, index=False)

    print(f"[01_corpus_ingestion] Unified structured corpus saved to {out_csv} (N = {len(df_unified)})")
    print(df_unified["partition"].value_counts())
    return df_unified


if __name__ == "__main__":
    build_unified_corpus(base_data_dir="../data")
