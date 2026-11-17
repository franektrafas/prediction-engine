# Prompt do Claude Code — sprawdź, czy model tłumi przewagę gospodarza na neutralnym gruncie

> To jest diagnostyka, nie zadanie do zaimplementowania — jeden skrypt,
> odczytać wynik, zgłosić z powrotem liczby. Nie zmieniaj żadnego kodu
> silnika na podstawie tego, dopóki nie zdecydujemy wspólnie, że trzeba.
> Skopiuj wszystko poniżej linii.

---

## Kontekst

`features/builder.py` ma `neutral_venue` (z surowych danych) i
`home_is_tournament_host`/`away_is_tournament_host` (`_add_tournament_host_flag`)
— cechy, które w teorii pozwalają modelowi odróżnić "ta drużyna gra
naprawdę u siebie" od "ta drużyna jest tylko z etykietą 'gospodarz' na
arkuszu meczowym, bo tak przypisano kolejność w rekordzie, np. na
mistrzostwach świata w kraju trzecim". Cechy są w danych treningowych od
początku. To, czego nie wiemy: czy wytrenowany model faktycznie **nauczył
się** to wykorzystywać, czy mimo obecności tych cech nadal traktuje samo
bycie "home_team_name" w rekordzie jako sygnał przewagi.

## Zadanie: jeden skrypt diagnostyczny

```python
from football_ai.predictions.service import PredictionService
from football_ai.predictions.schemas import MatchContextPayload
import pandas as pd

service = PredictionService.create()

def predict(home, away, neutral, country, home_elo, away_elo):
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
    return service.predict(payload)

# Elo spięte na sztywno w obu wariantach -- jedyne, co się różni, to która
# drużyna jest "home_team_name" na arkuszu, i flagi neutral_venue/country.
POLAND_ELO, GERMANY_ELO = 1700.0, 1750.0

# Wariant A: mecz na neutralnym gruncie (MŚ w Katarze), żadna drużyna nie
# jest gospodarzem turnieju -- oczekujemy MAŁEJ różnicy między A1 i A2.
a1 = predict("Poland", "Germany", True, "Qatar", POLAND_ELO, GERMANY_ELO)
a2 = predict("Germany", "Poland", True, "Qatar", GERMANY_ELO, POLAND_ELO)

# Wariant B: Polska naprawdę gra u siebie -- oczekujemy WYRAŹNEJ różnicy
# względem tego, gdyby to Niemcy grały naprawdę u siebie.
b1 = predict("Poland", "Germany", False, "Poland", POLAND_ELO, GERMANY_ELO)
b2 = predict("Germany", "Poland", False, "Germany", GERMANY_ELO, POLAND_ELO)

print("A1 (Polska 'home' na arkuszu, neutralny grunt):", a1.outcome_probabilities)
print("A2 (Niemcy 'home' na arkuszu, neutralny grunt):", a2.outcome_probabilities)
print("B1 (Polska naprawdę u siebie):", b1.outcome_probabilities)
print("B2 (Niemcy naprawdę u siebie):", b2.outcome_probabilities)

# P(Polska wygrywa) powinno być blisko takie samo w A1 (jako home_win) i A2
# (jako away_win), jeśli etykieta na arkuszu nie ma znaczenia na neutralnym
# gruncie. W B powinna być wyraźna różnica -- to jest realna przewaga
# własnego boiska.
swing_neutral = abs(a1.outcome_probabilities.home_win - a2.outcome_probabilities.away_win)
swing_real_home = abs(b1.outcome_probabilities.home_win - b2.outcome_probabilities.away_win)
print(f"Różnica na neutralnym gruncie: {swing_neutral:.3f} (oczekiwane: blisko 0)")
print(f"Różnica przy prawdziwym gospodarzu: {swing_real_home:.3f} (oczekiwane: wyraźnie > 0)")
```

## Co zgłosić z powrotem

1. Wszystkie cztery pełne `outcome_probabilities` (home_win/draw/away_win),
   nie tylko dwie liczby na końcu.
2. `swing_neutral` i `swing_real_home` — czy `swing_neutral` jest realnie
   mały (rzędu pojedynczych punktów procentowych) i wyraźnie mniejszy niż
   `swing_real_home`? Jeśli `swing_neutral` jest duży, porównywalny do
   `swing_real_home` — to znaczy, że model **nie** tłumi przewagi etykiety
   "gospodarz" na neutralnym gruncie, mimo że ma do tego cechy. To byłaby
   realna, konkretna wada, nie tylko podejrzenie.
3. Powtórz z inną parą drużyn (np. jedną z mniejszą różnicą Elo), żeby
   sprawdzić, czy wynik jest spójny, nie przypadkowy dla jednej pary.

## Czego nie robić

- Nie zmieniaj `features/builder.py` ani `training/` na podstawie tego
  wyniku bez rozmowy — to diagnostyka, decyzję co dalej podejmujemy razem.
- Nie odpalaj tego równolegle z trwającym treningiem statystyk, jeśli to
  obciąży CPU na tyle, że spowolni trening — to może poczekać do przerwy
  między targetami, nie jest pilne w sensie deadline'u.
