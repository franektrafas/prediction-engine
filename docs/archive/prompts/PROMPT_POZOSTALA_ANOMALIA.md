# Prompt do Claude Code — skąd bierze się pozostała anomalia dla Polska/Niemcy (duży gap Elo)

> Diagnostyka, jak poprzednio. Krok 1 poniżej nie wymaga żadnego nowego
> kodu -- artefakty już istnieją z retreningu, który właśnie zrobiłeś. Nie
> zmieniaj obsługi cech kategorycznych na podstawie tego bez rozmowy -- to
> jest 2b, nie 2a: usunięcie/przebudowa `home_team_name`/`away_team_name`
> jako cechy to kompromis (tracimy realny sygnał o stylu/jakości
> konkretnej drużyny), nie powtórzenie już zaakceptowanego wzorca.
> Skopiuj wszystko poniżej linii.

---

## Kontekst

Dobra robota na `PROMPT_NAPRAWA_PRZECIEKU_HOME.md` -- `home_is_home`/
`away_is_home` był realnym przeciekiem, fix + nowy test
(`test_every_numeric_feature_mirrors_across_a_home_away_relabel`) są
poprawne i zweryfikowane metodologicznie (cofnięcie fixu → test faluje
dokładnie tam, gdzie powinien). Podział na tury i pominięcie ESPN-owych
targetów (corners/cards/shots/possession) też był słuszny -- potwierdzam
w kodzie: `data_quality/provenance_checks.py:97-105`,
`find_forbidden_licenses()` -- ESPN pod Disney Terms of Use jest tam
formalnie oznaczone jako `FORBIDDEN`, potwierdzone 2026-08-20, nie
"niepewna licencja". Dobrze, że złapałeś to zamiast wykonać prompt
dosłownie -- ten wątek wróci osobno, gdy `data_quality/report.py`
(task #14, wciąż nieodpalony) pokaże ile realnego, nie-ESPN pokrycia
faktycznie jest dla tych statystyk.

Finalna diagnostyka: Portugalia/Holandia (mały gap Elo) wygląda teraz
poprawnie. Polska/Niemcy (duży gap Elo, 50 punktów) wciąż ma odwrócony
kierunek, tylko znacznie mniej wyraźny niż przed fixem (1.24x zamiast
2.33x). To nie jest szum -- to spójny, kierunkowy sygnał, i test
symetrii przechodzi teraz na **cechach numerycznych**. Podejrzenie: cechy
kategoryczne nie są objęte tym testem.

Sprawdziłem `features/builder.py:729-733`:

```python
categorical_columns = [
    column
    for column in ["tournament", "country", "home_team_name", "away_team_name"]
    if column in feature_frame.columns
]
...
feature_columns = numeric_columns + categorical_columns
```

`home_team_name`/`away_team_name` są podawane modelowi wprost jako cechy
kategoryczne. To zupełnie inny mechanizm niż `home_is_home` -- model może
nauczyć się "Niemcy jako home_team_name" jako odrębnej kategorii od
"Niemcy jako away_team_name", z własnym wyuczonym priorem z całej
historii, niezależnie od `neutral_venue` i niezależnie od jakiejkolwiek
cechy numerycznej. Niemcy jako silna, historyczna drużyna grająca
"u siebie" w zdecydowanej większości swoich meczów w danych treningowych
to dokładnie profil, który by to ujawnił mocniej niż Portugalia/Holandia.

## 1. Sprawdź to, zanim napiszesz cokolwiek nowego -- artefakty już istnieją

Retrening `match_outcome` (Krok 4 poprzedniego promptu) już zapisał
`feature_importance.png` i `shap_summary.png` per model w
`football_ai/evaluation/reports/match_outcome/<model_name>/` (patrz
`_persist_evaluation_artifacts` w `training/orchestrator.py`). Obejrzyj
je dla każdego z czterech modeli bazowych i dla stackingu -- gdzie w
rankingu ważności są `home_team_name`/`away_team_name` względem
`home_elo`/`away_elo`/`elo_diff` i cech rolling? Jeśli SHAP pokazuje
wysoki wkład tych dwóch kolumn konkretnie dla przypadków z dużym
`elo_diff`, to potwierdza hipotezę bez pisania nowego kodu.

Jeśli to nie wystarczy do jednoznacznej odpowiedzi, dopiero wtedy: policz
SHAP wprost dla A1/A2 (Polska/Niemcy) i C1/C2 (Portugalia/Holandia) z
`predictions/service.py`'s feature frame, i porównaj kontrybucję
`home_team_name`/`away_team_name` między parami.

## Co zgłosić z powrotem

1. Ranking ważności/SHAP dla `home_team_name`/`away_team_name` -- wysoki
   czy pomijalny względem reszty cech.
2. Czy kontrybucja tych dwóch kolumn different się zauważalnie między
   Polska/Niemcy a Portugalia/Holandia (potwierdzałoby, że to zależy od
   "historycznej lopsidedness" konkretnej pary, nie jest stałym artefaktem).
3. Twoja ocena: czy to realnie tłumaczy pozostałe 1.24x, czy to ślepy trop
   i trzeba szukać gdzie indziej.

## Czego nie robić

- Nie usuwaj ani nie przebudowuj `home_team_name`/`away_team_name` jako
  cech na podstawie samego podejrzenia -- to prawdziwy sygnał dla wielu
  innych właściwości drużyny, nie tylko nośnik przecieku. Decyzja co z tym
  zrobić (usunąć, symetryzować kodowanie, zostawić i zaakceptować
  rezydualne obciążenie przy ekstremalnych różnicach Elo na neutralnym
  gruncie) zapada wspólnie, nie w tym kroku.
- Nie odpalaj kolejnego retreningu, dopóki nie wiemy, czy to w ogóle jest
  właściwy trop.
