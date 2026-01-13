import os, re, requests

def read_study_details_msa(folder):
    """Reads study details for a given study, contains info of the study name and dataset source"""
    details_path = os.path.join(folder, "study_details.txt")
    details = {}
    if os.path.exists(details_path):
        with open(details_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key, value = parts
                        details[key.strip()] = value.strip()
    return details

_CHEBI_PATTERN = re.compile(r"^(CHEBI:\d+|chebi:\d+|\d+)$")

def is_valid_chebi_id(value: str) -> bool:
    return isinstance(value, str) and bool(_CHEBI_PATTERN.match(value))
def normalise_chebi_id(value: str) -> str:
    value = value.strip()
    if value.isdigit():
        return f"CHEBI:{value}"
    return value.upper()

def _safe_chebi_name(value, cursor):
    # Skip non-ChEBI columns
    if not is_valid_chebi_id(value):
        return value

    chebi_id = normalise_chebi_id(value)

    # 1️⃣ SQLite lookup
    cursor.execute(
        "SELECT name FROM chebi_name_map WHERE chebi_id = ?",
        (chebi_id,)
    )
    row = cursor.fetchone()

    if row and row[0] != chebi_id:
        #print('In lookup file')
        return row[0]

    #print('Using API')
    # 2️⃣ API fallback (UNCHANGED LOGIC)
    response = requests.get(
        "https://www.ebi.ac.uk/chebi/backend/api/public/compounds/",
        params={"chebi_ids": chebi_id},
        timeout=10
    )

    try:
        data = response.json()
    except ValueError:
        return value

    # CASE 1: dict keyed by CHEBI ID
    if isinstance(data, dict):
        entry = data.get(chebi_id) or data.get(value)
        if entry and entry.get("exists"):
            name = entry["data"].get("name", value)
            cursor.execute(
                "INSERT OR REPLACE INTO chebi_name_map VALUES (?, ?)",
                (chebi_id, name)
            )
            return name

    # CASE 2: list response
    if isinstance(data, list) and data and isinstance(data[0], dict):
        name = data[0].get("name", value)
        cursor.execute(
            "INSERT OR REPLACE INTO chebi_name_map VALUES (?, ?)",
            (chebi_id, name)
        )
        return name

    return value