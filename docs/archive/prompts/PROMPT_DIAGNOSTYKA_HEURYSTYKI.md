# Prompt do Claude Code — czy anomalia neutral-venue pochodzi z modelu, czy z heurystyki fallback

> To jest diagnostyka, nie zadanie do zaimplementowania — tak jak
> `PROMPT_HOME_ADVANTAGE_CHECK.md` wcześniej. Nie zmieniaj żadnego kodu
> silnika na podstawie tego wyniku, dopóki nie zdecydujemy wspólnie, co
> dalej. Skopiuj wszystko poniżej linii.

---

## Kontekst

Ponowna diagnostyka home-advantage (po retreningu z fixem neutral-venue)
dała niepokojący wynik: różnica na neutralnym gruncie nie jest bliska
zeru, a w jednej z dwóch par jest nawet większa niż przy prawdziwym
gospodarzu. Fix w `features/builder.py` ma za sobą przechodzący test
jednostkowy na syntetycznych danych (`test_neutral_venue_does_not_contaminate_true_home_rolling`),
więc problem raczej nie jest w tym, że fix jest źle napisany — problem
może być w tym, że diagnostyka na żywym serwisie w ogóle nie widzi
wyniku tego modelu.

Czytając `football_ai/predictions/service.py`, `_resolve_outcome_probabilities`
(linie 424-453) robi coś istotnego:

```python
if _probability_spread(model_probabilities) >= 0.05:
    return model_probabilities
heuristic = self._heuristic_prediction(payload)
...
blended = {
    "home_win": 0.65 * heuristic_probabilities["home_win"] + 0.35 * goal_based["home_win"],
    ...
}
```

Jeśli rozkład klas z wytrenowanego modelu jest prawie płaski (rozstęp
<0.05 — a to dokładnie sytuacja wyrównanych drużyn, gdzie neutral-venue
ma największe znaczenie), serwis **po cichu odrzuca wynik modelu** i
miesza w `_heuristic_prediction` (linie 494-511):

```python
elo_diff = (home_elo - away_elo) / 400.0
home_win = _logistic(elo_diff + 0.18)
away_win = _logistic(-elo_diff - 0.18) * 0.78
```

Ta funkcja bierze wyłącznie `home_elo`/`away_elo` z payloadu — **nigdy nie
sprawdza `neutral_venue`**. Ma na sztywno wbity bonus `+0.18`/`-0.18` dla
tego, kto jest "home" na arkuszu, niezależnie od tego, czy to naprawdę
mecz u siebie. Jeśli ta ścieżka odpala się choć czasami w diagnostyce, fix
w `features/builder.py` może być całkowicie poprawny, a mimo to niewidoczny
w odpowiedziach serwisu.

Osobna, słabsza hipoteza: dryf schematu zgłoszony wcześniej (12 kolumn
`squad_*` zawsze NaN w produkcji, bo `predictions/service.py` nigdy nie
przekazuje `squad_summary` do `build_features()`) może zwiększać częstość
płaskich rozkładów modelu, więc częściej odpalać ten fallback. To hipoteza
poboczna, nie do potwierdzenia tym skryptem wprost — nie próbuj tego
udowadniać tutaj, tylko odnotuj jeśli rzuci się w oczy.

## Zadanie: jeden skrypt diagnostyczny

Cel: zobaczyć surowy wynik modelu (`_predict_classification`, przed
`_resolve_outcome_probabilities`) obok finalnego wyniku serwisu
(`predict()`), dla dokładnie tych samych ośmiu wywołań co w poprzedniej
diagnostyce (dwie pary × warianty A1/A2/B1/B2).

```python
from football_ai.predictions.service import PredictionService
from football_ai.predictions.schemas import MatchContextPayload
import pandas as pd

service = PredictionService.create()

registry_entry = service.registry.get_latest("match_outcome")
print("=== match_outcome registry entry ===")
print("version:", registry_entry.version if registry_entry else None)
print("artifact_path:", getattr(registry_entry, "artifact_path", None))
print("(sprawdź czy to na pewno świeży artefakt z tego retreningu, nie coś"
      " sprzed 25 sierpnia -- porównaj z datą modyfikacji pliku artefaktu)")
print()


def raw_vs_resolved(home, away, neutral, country, home_elo, away_elo, label):
    payload = MatchContextPayload(
        match_date=pd.Timestamp("2026-11-20", tz="UTC"),
        home_team_name=home,
        away_team_name=away,
        tournament="FIFA World Cup",
        neutral_venue=neutral,
        country=country,
        home_elo=home_elo,
        away_elo=away_elo,
    )
    match_frame = service._build_feature_frame(payload)
    raw_probabilities = service._predict_classification(
        artifact_path=registry_entry.artifact_path,
        frame=match_frame,
        labels=["away_win", "draw", "home_win"],
    )
    spread = max(raw_probabilities.values()) - min(raw_probabilities.values())
    fallback_would_trigger = spread < 0.05
    resolved = service.predict(payload)
    print(f"--- {label} ---")
    print("surowy model:      ", raw_probabilities)
    print(f"rozstęp surowy:     {spread:.4f}  (fallback odpala się: {fallback_would_trigger})")
    print("finalny (serwis):  ", resolved.outcome_probabilities.model_dump())
    print()
    return raw_probabilities, resolved.outcome_probabilities.model_dump()


POLAND_ELO, GERMANY_ELO = 1700.0, 1750.0
PORTUGAL_ELO, NETHERLANDS_ELO = 1780.0, 1790.0

print("##### Para 1: Polska vs Niemcy #####")
a1_raw, a1_final = raw_vs_resolved("Poland", "Germany", True, "Qatar", POLAND_ELO, GERMANY_ELO, "A1 neutralny, Polska=home")
a2_raw, a2_final = raw_vs_resolved("Germany", "Poland", True, "Qatar", GERMANY_ELO, POLAND_ELO, "A2 neutralny, Niemcy=home")
b1_raw, b1_final = raw_vs_resolved("Poland", "Germany", False, "Poland", POLAND_ELO, GERMANY_ELO, "B1 Polska realny gospodarz")
b2_raw, b2_final = raw_vs_resolved("Germany", "Poland", False, "Germany", GERMANY_ELO, POLAND_ELO, "B2 Niemcy realny gospodarz")

print("##### Para 2: Portugalia vs Holandia #####")
c1_raw, c1_final = raw_vs_resolved("Portugal", "Netherlands", True, "Qatar", PORTUGAL_ELO, NETHERLANDS_ELO, "C1 neutralny, Portugalia=home")
c2_raw, c2_final = raw_vs_resolved("Netherlands", "Portugal", True, "Qatar", NETHERLANDS_ELO, PORTUGAL_ELO, "C2 neutralny, Holandia=home")
d1_raw, d1_final = raw_vs_resolved("Portugal", "Netherlands", False, "Portugal", PORTUGAL_ELO, NETHERLANDS_ELO, "D1 Portugalia realny gospodarz")
d2_raw, d2_final = raw_vs_resolved("Netherlands", "Portugal", False, "Netherlands", NETHERLANDS_ELO, PORTUGAL_ELO, "D2 Holandia realny gospodarz")

def swing(x1, x2, key_home="home_win", key_away="away_win"):
    return abs(x1[key_home] - x2[key_away])

print("##### Swing na SUROWYCH prawdopodobieństwach modelu (bez fallbacku) #####")
print("Para 1 -- neutralny:      ", swing(a1_raw, a2_raw))
print("Para 1 -- realny gospodarz:", swing(b1_raw, b2_raw))
print("Para 2 -- neutralny:      ", swing(c1_raw, c2_raw))
print("Para 2 -- realny gospodarz:", swing(d1_raw, d2_raw))
```

## Co zgłosić z powrotem

1. Wersja/data artefaktu zarejestrowanego pod `match_outcome` — czy to na
   pewno świeży model z retreningu po fixie, nie stary.
2. Dla wszystkich ośmiu wywołań: surowe prawdopodobieństwa, rozstęp, i czy
   `fallback_would_trigger` było `True` czy `False`. To jest najważniejsza
   linia w całym raporcie — mówi wprost, czy diagnostyka w ogóle widziała
   wynik modelu, czy heurystykę.
3. Swing liczony na surowych prawdopodobieństwach (bez fallbacku) dla obu
   par, przy neutralnym gruncie vs przy realnym gospodarzu. Jeśli surowy
   swing na neutralnym gruncie jest już mały (jak oczekiwano) — to znaczy,
   że fix działa, a poprzedni niepokojący wynik pochodził z
   `_resolve_outcome_probabilities`/`_heuristic_prediction`, nie z
   `features/builder.py`. Jeśli surowy swing nadal jest nieprawidłowy —
   problem jest głębiej, w samych cechach, mimo przechodzącego testu
   jednostkowego (możliwe, że test nie pokrywa realnego przypadku).
4. Jeśli zauważysz coś dotyczące hipotezy pobocznej (częstość fallbacku a
   dryf schematu squad_*) — odnotuj, ale nie próbuj tego formalnie dowodzić
   w ramach tego skryptu.

## Czego nie robić

- Nie zmieniaj `_resolve_outcome_probabilities`, `_heuristic_prediction`
  ani `features/builder.py` na podstawie tego wyniku bez rozmowy — to
  diagnostyka, decyzję co dalej podejmujemy razem, tak jak poprzednio.
- Nie naprawiaj dryfu schematu `squad_*` przy okazji, nawet jeśli się
  potwierdzi jako czynnik — to osobny wątek, osobna decyzja.
