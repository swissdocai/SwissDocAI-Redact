"""What to black out and what to keep readable — the part of the PII prompt the user edits.

The rest of the prompt (output format, copying values exactly, never following
instructions inside the document) is fixed in pii.py: changing it would break
the pipeline or open it to documents that try to steer the model.

    policy = default_policy("de")
    policy.redact.append("Fahrzeugkennzeichen")
    detect_with_llm(client, pages, policy=policy)
"""

from dataclasses import asdict, dataclass, field


@dataclass
class Policy:
    redact: list[str] = field(default_factory=list)  # black out (one item per line)
    keep: list[str] = field(default_factory=list)  # keep readable
    notes: str = ""  # anything else the model should know

    @classmethod
    def from_dict(cls, data: dict) -> "Policy":
        return cls([str(x).strip() for x in data.get("redact", []) if str(x).strip()],
                   [str(x).strip() for x in data.get("keep", []) if str(x).strip()],
                   str(data.get("notes", "")).strip())

    def to_dict(self) -> dict:
        return asdict(self)


_DEFAULTS = {
    "en": Policy(
        redact=[
            "Names of natural persons: patients, relatives, doctors, staff, signatories — also the surname or first "
            "name alone and initials with surname",
            "Dates of birth",
            "Postal addresses of persons: street and house number, postcode and town, P.O. box",
            "Phone and fax numbers and e-mail addresses of persons",
            "Numbers and codes that identify a person or their case: AHV/AVS number, patient, case, file and "
            "reference numbers, insurance and policy numbers, ID and passport numbers, license plates, and links "
            "containing such numbers",
            "IBAN, bank account and card numbers",
        ],
        keep=[
            "Names, addresses, phone numbers, e-mail addresses and websites of organizations: clinics, practices, "
            "companies, insurers, authorities",
            "Dates of letters, appointments and examinations",
            "Medical and other content: history, findings, diagnoses, treatments",
            "Salutations and titles on their own (Herr, Frau, Dr. med., Prof.), professions and specialties",
        ],
    ),
    "de": Policy(
        redact=[
            "Namen von Personen: Patientinnen und Patienten, Angehörige, Ärztinnen und Ärzte, Mitarbeitende, "
            "Unterzeichnende – auch Vor- oder Nachname allein und Initialen mit Nachname",
            "Geburtsdaten",
            "Adressen von Personen: Strasse und Hausnummer, PLZ und Ort, Postfach",
            "Telefon- und Faxnummern sowie E-Mail-Adressen von Personen",
            "Nummern, die eine Person oder ihren Fall erkennbar machen: AHV-Nummer, Patienten-, Fall-, Dossier- und "
            "Referenznummern, Versicherten- und Policennummern, Ausweis- und Passnummern, Kontrollschilder, "
            "Links mit solchen Nummern",
            "IBAN, Konto- und Kartennummern",
        ],
        keep=[
            "Namen, Adressen, Telefonnummern, E-Mail-Adressen und Websites von Organisationen: Kliniken, Praxen, "
            "Firmen, Versicherungen, Behörden",
            "Datum des Schreibens, Termine und Untersuchungsdaten",
            "Medizinische und andere Inhalte: Vorgeschichte, Befunde, Diagnosen, Behandlungen",
            "Anreden und Titel allein (Herr, Frau, Dr. med., Prof.), Berufe und Fachgebiete",
        ],
    ),
    "fr": Policy(
        redact=[
            "Noms de personnes : patients, proches, médecins, collaborateurs, signataires – aussi le nom ou le "
            "prénom seul et les initiales avec le nom",
            "Dates de naissance",
            "Adresses de personnes : rue et numéro, NPA et localité, case postale",
            "Numéros de téléphone et de fax et adresses e-mail de personnes",
            "Numéros qui permettent d'identifier une personne ou son dossier : numéro AVS, numéros de patient, de "
            "cas, de dossier et de référence, numéros d'assuré et de police, numéros de pièce d'identité et de "
            "passeport, plaques d'immatriculation, liens contenant de tels numéros",
            "IBAN, numéros de compte et de carte",
        ],
        keep=[
            "Noms, adresses, numéros de téléphone, adresses e-mail et sites web d'organisations : cliniques, "
            "cabinets, entreprises, assurances, autorités",
            "Date de la lettre, rendez-vous et dates d'examen",
            "Contenu médical et autre : anamnèse, résultats, diagnostics, traitements",
            "Formules d'appel et titres seuls (Monsieur, Madame, Dr méd., Prof.), professions et spécialités",
        ],
    ),
    "it": Policy(
        redact=[
            "Nomi di persone: pazienti, familiari, medici, collaboratori, firmatari – anche il solo cognome o nome e "
            "le iniziali con il cognome",
            "Date di nascita",
            "Indirizzi di persone: via e numero civico, NPA e località, casella postale",
            "Numeri di telefono e di fax e indirizzi e-mail di persone",
            "Numeri che permettono di identificare una persona o la sua pratica: numero AVS, numeri di paziente, "
            "di caso, di pratica e di riferimento, numeri di assicurato e di polizza, numeri di documento d'identità "
            "e di passaporto, targhe, link che contengono tali numeri",
            "IBAN, numeri di conto e di carta",
        ],
        keep=[
            "Nomi, indirizzi, numeri di telefono, indirizzi e-mail e siti web di organizzazioni: cliniche, studi, "
            "aziende, assicurazioni, autorità",
            "Data della lettera, appuntamenti e date degli esami",
            "Contenuti medici e altri: anamnesi, referti, diagnosi, trattamenti",
            "Appellativi e titoli da soli (Signor, Signora, Dr. med., Prof.), professioni e specialità",
        ],
    ),
}


def default_policy(language: str = "en") -> Policy:
    """A fresh copy of the standard policy in ``language`` (de / fr / it / en)."""
    base = _DEFAULTS.get(language, _DEFAULTS["en"])
    return Policy(list(base.redact), list(base.keep), base.notes)
