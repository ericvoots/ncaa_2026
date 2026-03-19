"""Team name index: canonical IDs + fuzzy matching across data sources."""
import json
import re
from pathlib import Path
from typing import Optional

import polars as pl
from rapidfuzz import fuzz, process

import config

# ── Manual overrides for known mismatches ──────────────────────────────────
# Maps variant name → canonical name (as it appears in NCAA API or our master list)
MANUAL_OVERRIDES = {
    # TeamRankings → canonical
    "UConn": "Connecticut",
    "UCONN": "Connecticut",
    "UCF": "Central Florida",
    "UNC": "North Carolina",
    "UNLV": "Nevada-Las Vegas",
    "Ole Miss": "Mississippi",
    "Pitt": "Pittsburgh",
    "SMU": "Southern Methodist",
    "USC": "Southern California",
    "LSU": "Louisiana State",
    "VCU": "Virginia Commonwealth",
    "BYU": "Brigham Young",
    "TCU": "Texas Christian",
    "ETSU": "East Tennessee State",
    "UTSA": "Texas at San Antonio",
    "UTEP": "Texas at El Paso",
    "SIU Edwardsville": "SIUE",
    "SIU-Edwardsville": "SIUE",
    "SIU Edward": "SIUE",
    "Saint Mary's": "Saint Mary's (CA)",
    "St. Mary's": "Saint Mary's (CA)",
    "Saint Peter's": "Saint Peter's",
    "St. Peter's": "Saint Peter's",
    "Saint John's": "St. John's (NY)",
    "St. John's": "St. John's (NY)",
    "Saint Louis": "Saint Louis",
    "St. Louis": "Saint Louis",
    "St Louis": "Saint Louis",
    "SLU": "Saint Louis",
    "Saint Joseph's": "Saint Joseph's",
    "St. Joseph's": "Saint Joseph's",
    "Miami FL": "Miami (FL)",
    "Miami (FL)": "Miami (FL)",
    "Miami OH": "Miami (OH)",
    "Miami (OH)": "Miami (OH)",
    "N Carolina": "North Carolina",
    "N.C. State": "North Carolina State",
    "NC State": "North Carolina State",
    "N Carolina St": "North Carolina State",
    "N Dakota St": "North Dakota State",
    "N Dakota": "North Dakota",
    "S Dakota St": "South Dakota State",
    "S Dakota": "South Dakota",
    "S Carolina": "South Carolina",
    "W Virginia": "West Virginia",
    "W Kentucky": "Western Kentucky",
    "E Washington": "Eastern Washington",
    "E Kentucky": "Eastern Kentucky",
    "E Michigan": "Eastern Michigan",
    "E Illinois": "Eastern Illinois",
    "SE Missouri St": "Southeast Missouri State",
    "SE Louisiana": "Southeastern Louisiana",
    "SW Missouri St": "Southwest Missouri State",
    "N Illinois": "Northern Illinois",
    "N Iowa": "Northern Iowa",
    "N Kentucky": "Northern Kentucky",
    "NW State": "Northwestern State",
    "S Illinois": "Southern Illinois",
    "S Utah": "Southern Utah",
    "S Miss": "Southern Mississippi",
    "Southern Miss": "Southern Mississippi",
    "Loyola Chicago": "Loyola (IL)",
    "Loyola-Chicago": "Loyola (IL)",
    "Loyola MD": "Loyola (MD)",
    "Loyola Marymount": "Loyola Marymount",
    "Texas A&M-CC": "Texas-Am-Corpus",
    "Texas A&M Corpus Chris": "Texas-Am-Corpus",
    "A&M-Corpus Christi": "Texas-Am-Corpus",
    "Corpus Christi": "Texas-Am-Corpus",
    "FGCU": "Florida Gulf Coast",
    "Fla Gulf Coast": "Florida Gulf Coast",
    "Fla Atlantic": "Florida Atlantic",
    "FAU": "Florida Atlantic",
    "F Dickinson": "Fairleigh Dickinson",
    "Fairleigh Dickinson": "Fairleigh Dickinson",
    "FDU": "Fairleigh Dickinson",
    "Geo Washington": "George Washington",
    "Geo Mason": "George Mason",
    "UMBC": "Maryland-Baltimore-County",
    "Md Baltimore County": "Maryland-Baltimore-County",
    "UMass": "Massachusetts",
    "UMass Lowell": "Massachusetts-Lowell",
    "UL Monroe": "La.-Monroe",
    "NJIT": "New-Jersey-Tech",
    "SC Upstate": "South-Carolina-Upstate",
    "Incarnate Word": "UIW",
    "Appalachian St": "Appalachian State",
    "Appalachian St.": "Appalachian State",
    "Ball St": "Ball State",
    "Ball St.": "Ball State",
    "Boise St": "Boise State",
    "Boise St.": "Boise State",
    "Colorado St": "Colorado State",
    "Colorado St.": "Colorado State",
    "Fresno St": "Fresno State",
    "Fresno St.": "Fresno State",
    "Indiana St": "Indiana State",
    "Indiana St.": "Indiana State",
    "Iowa St": "Iowa State",
    "Iowa St.": "Iowa State",
    "Kansas St": "Kansas State",
    "Kansas St.": "Kansas State",
    "Kent St": "Kent State",
    "Kent St.": "Kent State",
    "Michigan St": "Michigan State",
    "Michigan St.": "Michigan State",
    "Mississippi St": "Mississippi State",
    "Mississippi St.": "Mississippi State",
    "Montana St": "Montana State",
    "Montana St.": "Montana State",
    "Morehead St": "Morehead State",
    "Morehead St.": "Morehead State",
    "Murray St": "Murray State",
    "Murray St.": "Murray State",
    "Ohio St": "Ohio State",
    "Ohio St.": "Ohio State",
    "Oklahoma St": "Oklahoma State",
    "Oklahoma St.": "Oklahoma State",
    "Oregon St": "Oregon State",
    "Oregon St.": "Oregon State",
    "Penn St": "Penn State",
    "Penn St.": "Penn State",
    "Portland St": "Portland State",
    "Portland St.": "Portland State",
    "Sacramento St": "Sacramento State",
    "Sacramento St.": "Sacramento State",
    "San Diego St": "San Diego State",
    "San Diego St.": "San Diego State",
    "San Jose St": "San Jose State",
    "San Jose St.": "San Jose State",
    "Utah St": "Utah State",
    "Utah St.": "Utah State",
    "Weber St": "Weber State",
    "Weber St.": "Weber State",
    "Wichita St": "Wichita State",
    "Wichita St.": "Wichita State",
    "Wright St": "Wright State",
    "Wright St.": "Wright State",
    "Kennesaw St": "Kennesaw State",
    "Kennesaw St.": "Kennesaw State",
    "J Madison": "James Madison",
    "Jax State": "Jacksonville State",
    "Jacksonville St": "Jacksonville State",
    "Sam Houston St": "Sam Houston State",
    "Sam Houston": "Sam Houston State",
    "Grambling St": "Grambling State",
    "Grambling St.": "Grambling State",
    "Alcorn St": "Alcorn State",
    "Alcorn St.": "Alcorn State",
    "Bethune-Cookman": "Bethune-Cookman",
    "N Florida": "North Florida",
    "S Florida": "South Florida",
    "C Florida": "Central Florida",
    "W Michigan": "Western Michigan",
    "W Illinois": "Western Illinois",
    "C Michigan": "Central Michigan",
    "C Connecticut": "Central Connecticut",
    "C Arkansas": "Central Arkansas",
    "LIU": "Long Island University",
    "Long Island": "Long Island University",
    "Loyola-Maryland": "Loyola (MD)",
    "G'town": "Georgetown",
    "IL Chicago": "Illinois-Chicago",
    "UIC": "Illinois-Chicago",
    "UT Martin": "Tenn-Martin",
    "UT Arlington": "Texas-Arlington",
    "UT Rio Grande Valley": "Texas-Rio Grande Valley",
    "UTRGV": "Texas-Rio Grande Valley",
    "Abl Christian": "Abilene Christian",
    "Abil Christian": "Abilene Christian",
    "Cal Poly": "Cal Poly",
    "Cal Baptist": "California Baptist",
    "Cal St Fullerton": "Cal State Fullerton",
    "CSU Fullerton": "Cal State Fullerton",
    "Cal St Northridge": "Cal State Northridge",
    "CSUN": "Cal St. Northridge",
    "Cal St Bakersfield": "Cal State Bakersfield",
    "Cal St Sacramento": "Sacramento State",
    "Ark Pine Bluff": "Arkansas-Pine Bluff",
    "Ark Little Rock": "Little Rock",
    "Little Rock": "Little Rock",
    "UALR": "Little-Rock",
    "Ga Southern": "Georgia Southern",
    "Ga Tech": "Georgia Tech",
    "GA Tech": "Georgia Tech",
    "Mt St Mary's": "Mount St. Mary's",
    "Mount St Mary's": "Mount St. Mary's",
    "Col Charleston": "College of Charleston",
    "Charleston": "College of Charleston",
    "Loyola Chi": "Loyola (IL)",
    "NC Wilmington": "UNC Wilmington",
    "UL Monroe": "La.-Monroe",
    "W Carolina": "Western Carolina",
    "S Carolina St": "South Carolina State",
    "App State": "Appalachian State",
    "Miss Valley St": "Mississippi Valley State",
    "Loyola Mymt": "Loyola Marymount",
    "St Fran (NY)": "St. Francis (NY)",
    "St Fran (PA)": "Saint Francis (PA)",
    "UCSB": "UC Santa Barbara",
    "CS Fullerton": "Cal State Fullerton",
    "NC Central": "North Carolina Central",
    "SFA": "Stephen F. Austin",
    "Southeast Mo. St.": "Southeast Mo. St.",
    "Massachusetts": "Mass-Amherst",
    "Chattanooga": "Chattanooga",
    "UCSD": "UC San Diego",
    "CS Northridge": "Cal State Northridge",
    "CS Bakersfield": "Cal State Bakersfield",
    "Tenn Tech": "Tennessee Tech",
    "Hou Christian": "Houston Christian",
    "NC A&T": "North Carolina A&T",
    "E Carolina": "East Carolina",
    "UT Rio Grande": "Texas-Rio Grande Valley",
    "Tenn St": "Tennessee State",
    "Tenn Martin": "Tennessee-Martin",
    "Ga State": "Georgia State",
    "Fla International": "Florida International",
    "FIU": "Florida International",
    "UNCG": "UNC Greensboro",
    "UNC-Greensboro": "UNC Greensboro",
    "UNCW": "UNC Wilmington",
    "UNC-Wilmington": "UNC Wilmington",
    "UNCA": "UNC Asheville",
    "UNC-Asheville": "UNC Asheville",
    "St Thomas": "St. Thomas (MN)",
    "St. Thomas": "St. Thomas (MN)",
    "Samford": "Samford",
    "High Point": "High Point",
    "N Colorado": "Northern Colorado",
    "UAB": "UAB",
}


def _normalize(name: str) -> str:
    """Lowercase, strip periods, extra spaces, and common suffixes."""
    name = name.strip().lower()
    name = name.replace(".", "").replace("'", "'")
    name = re.sub(r"\s+", " ", name)
    return name


class TeamIndex:
    """Resolve team names from any source to a canonical team_id."""

    def __init__(self, path: Optional[Path] = None):
        self._path = path or (config.RAW_DIR / "team_index.parquet")
        self._df: Optional[pl.DataFrame] = None
        self._name_to_id: dict[str, int] = {}
        self._id_to_name: dict[int, str] = {}
        self._all_names: list[str] = []
        if self._path.exists():
            self.load()

    def load(self):
        self._df = pl.read_parquet(self._path)
        self._name_to_id = {}
        self._id_to_name = {}
        for row in self._df.iter_rows(named=True):
            tid = row["team_id"]
            canonical = row["canonical_name"]
            self._id_to_name[tid] = canonical
            self._name_to_id[_normalize(canonical)] = tid
            # Add alt names
            if row.get("alt_names"):
                for alt in json.loads(row["alt_names"]):
                    self._name_to_id[_normalize(alt)] = tid
        self._all_names = list(self._name_to_id.keys())

    def resolve(self, name: str) -> Optional[int]:
        """Return team_id for any name variant. Uses overrides → exact → fuzzy."""
        if not name or not name.strip():
            return None

        # Check manual overrides first
        if name in MANUAL_OVERRIDES:
            canonical = MANUAL_OVERRIDES[name]
            norm = _normalize(canonical)
            if norm in self._name_to_id:
                return self._name_to_id[norm]

        # Exact normalized match
        norm = _normalize(name)
        if norm in self._name_to_id:
            return self._name_to_id[norm]

        # Fuzzy match
        if self._all_names:
            result = process.extractOne(norm, self._all_names, scorer=fuzz.WRatio, score_cutoff=80)
            if result:
                matched_name, score, _ = result
                return self._name_to_id[matched_name]

        return None

    def resolve_or_warn(self, name: str, source: str = "") -> Optional[int]:
        """Like resolve but prints a warning if not found."""
        tid = self.resolve(name)
        if tid is None:
            print(f"  [WARN] Could not resolve team: '{name}' (source={source})")
        return tid

    def get_name(self, team_id: int) -> Optional[str]:
        return self._id_to_name.get(team_id)

    @property
    def df(self) -> Optional[pl.DataFrame]:
        return self._df
