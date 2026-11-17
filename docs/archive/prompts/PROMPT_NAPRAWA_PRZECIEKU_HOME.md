# Prompt do Claude Code — znaleźć i naprawić realny mechanizm przecieku etykiety "home" na neutralnym gruncie

> Masz pełną swobodę zmian w silniku na tej fazie (patrz
> `PROMPT_ZAMKNIECIE_WATKOW.md`) -- z jednym wyjątkiem opisanym w kroku 2
> poniżej. Zrób krok 1 (diagnostyka) PRZED jakąkolwiek zmianą kodu -- nie
> zgaduj, który mechanizm przecieka, zmierz to. Jeśli wynik pasuje do
> wzorca znanego z `_add_rolling_features` (patrz krok 2a), napraw to samo
> jak tam, bez pytania. Jeśli wynik pokaże coś strukturalnie innego (krok
> 2b), zatrzymaj się i zgłoś, zanim cokolwiek zmienisz -- to inna klasa
> decyzji niż powtórzenie już zaakceptowanego wzorca. Skopiuj wszystko
> poniżej linii.

---

## Kontekst

Diagnostyka fallbacku (`PROMPT_DIAGNOSTYKA_HEURYSTYKI.md`) jednoznacznie
wykluczyła `_resolve_outcome_probabilities`/`_heuristic_prediction` --
`fallback_would_trigger` było `False` wszędzie, model LightGBM był
wywoływany bezpośrednio. Mimo to anomalia na neutralnym gruncie zostaje:
sam wytrenowany model, na surowych prawdopodobieństwach, wciąż reaguje na
to, która drużyna jest "home_team_name" na arkuszu, nawet gdy
`neutral_venue=True`.

`test_neutral_venue_does_not_contaminate_true_home_rolling` przechodzi --
fix w `_add_rolling_features` jest poprawnie zaimplementowany zgodnie z
tym testem. Wniosek: test pokrywa jeden konkretny mechanizm (rolling
features), a na pełnej, prawdziwej skali danych przecieka coś innego,
czego ten test nie dotyka. Możliwe kandydaci, bez rozstrzygania na ślepo
który to:

- `_add_recency_features` i `_add_h2h_features` -- `PROMPT_ZAMKNIECIE_WATKOW.md`
  prosił o sprawdzenie, czy mają analogiczny podział po `is_home` bez
  gate'u na `neutral_venue`, jako "jeśli tak" -- możliwe, że fix trafił
  tylko do `_add_rolling_features`, a te dwie funkcje zostały nietknięte.
- Surowa kolumna identyfikująca drużynę (`home_team_name`/`away_team_name`
  albo jakiekolwiek ich kodowanie) użyta bezpośrednio jako cecha
  kategoryczna modelu -- to byłby całkowicie inny mechanizm przecieku niż
  cokolwiek w rolling/recency/h2h, bo dotyczyłby surowej tożsamości
  drużyny, nie żadnej agregacji historycznej.
- Elo z wbudowaną korektą za grę u siebie na poziomie samego źródła
  (`scrapers/sources/elo.py` albo obliczenia `elo_diff`) -- część
  systemów ratingowych Elo w piłce dolicza gospodarzowi stały bonus przy
  aktualizacji/odczycie ratingu, niezależnie od tego, czy to naprawdę mecz
  u siebie.

Nie zgaduj który z nich -- krok 1 to zmierzy wprost.

## 1. Diagnostyka: różnica ramek cech dla identycznego meczu z zamienionymi etykietami

Dla pary A1/A2 z poprzednich diagnostyk (Polska/Niemcy, neutralny grunt,
identyczne Elo w obu wariantach, jedyna różnica to która drużyna jest
"home" na arkuszu) zbuduj obie ramki cech i porównaj kolumna po kolumnie.

```python
from football_ai.predictions.service import PredictionService
from football_ai.predictions.schemas import MatchContextPayload
import pandas as pd

service = PredictionService.create()

def feature_row(home, away, home_elo, away_elo):
    payload = MatchContextPayload(
        match_date=pd.Timestamp("2026-11-20", tz="UTC"),
        home_team_name=home,
        away_team_name=away,
        tournament="FIFA World Cup",
        neutral_venue=True,
        country="Qatar",
        home_elo=home_elo,
        away_elo=away_elo,
    )
    return service._build_feature_frame(payload).iloc[0]

a1 = feature_row("Poland", "Germany", 1700.0, 1750.0)
a2 = feature_row("Germany", "Poland", 1750.0, 1700.0)

# Dla każdej pary kolumn home_X/away_X: na neutralnym gruncie z identycznym
# Elo, wartość home_X w a1 powinna odpowiadać wartości away_X w a2 (bo to
# ta sama drużyna, tylko zamieniona etykieta), i odwrotnie. Znajdź kolumny,
# które łamią tę symetrię.
common_columns = sorted(set(a1.index) & set(a2.index))
mismatches = []
for column in common_columns:
    if not column.startswith("home_") and not column.startswith("away_"):
        continue
    mirror_column = (
        "away_" + column[len("home_"):] if column.startswith("home_")
        else "home_" + column[len("away_"):]
    )
    if mirror_column not in a2.index:
        continue
    v1, v2 = a1[column], a2[mirror_column]
    try:
        equal = abs(float(v1) - float(v2)) < 1e-9
    except (TypeError, ValueError):
        equal = (v1 == v2) or (pd.isna(v1) and pd.isna(v2))
    if not equal:
        mismatches.append((column, v1, mirror_column, v2))

print(f"{len(mismatches)} kolumn łamie symetrię home/away na neutralnym gruncie:")
for column, v1, mirror_column, v2 in sorted(mismatches):
    print(f"  {column}={v1!r}  vs  {mirror_column}={v2!r}")

# Sprawdź też cechy niezwiązane z home_/away_ prefiksem, które mogłyby
# nieść tożsamość drużyny wprost (np. surowe nazwy, kody kategoryczne).
identity_like = [c for c in common_columns if "team_name" in c or "team_id" in c]
print("Kolumny z tożsamością drużyny obecne w ramce cech:", identity_like)
```

To da listę dokładnych kolumn, które przeciekają -- zgłoś ją w całości,
zanim przejdziesz do kroku 2.

## 2a. Jeśli przeciek pasuje do już zaakceptowanego wzorca -- napraw bez pytania

Jeśli lista z kroku 1 wskazuje na kolumny pochodzące z `_add_recency_features`
i/albo `_add_h2h_features` (np. `home_days_since_last_match`,
`home_last_result`, `home_h2h_*` itp.) -- zastosuj dokładnie ten sam wzorzec
gate'u co w `_add_rolling_features` (efektywna flaga
`effective_is_home = is_home & (~neutral_venue)` użyta w grupowaniu zamiast
gołego `is_home`), konsekwentnie w obu funkcjach. To jest już zaakceptowana
decyzja architektoniczna, nie nowa -- nie musisz pytać przed zastosowaniem.

## 2b. Jeśli przeciek pochodzi z czegoś innego -- zatrzymaj się i zgłoś

Jeśli lista z kroku 1 wskazuje na:
- surową tożsamość drużyny (nazwa/kod jako cecha kategoryczna), albo
- Elo z wbudowaną korektą za grę u siebie na poziomie źródła, albo
- cokolwiek niepasujące do wzorca z 2a

**nie naprawiaj tego w tym samym przejściu.** To inna klasa problemu --
np. usunięcie surowej tożsamości drużyny jako cechy może obniżyć jakość
modelu dla wielu innych, uzasadnionych sygnałów (styl gry danej drużyny),
a korekta Elo za grę u siebie może być celowa i pożądana poza kontekstem
neutralnym. Zatrzymaj się na raporcie z kroku 1 i konkretną hipotezą co do
mechanizmu -- zdecydujemy dalej wspólnie, tak jak przy poprzednich dwóch
diagnostykach.

## 3. Wzmocnić testy -- niezależnie od tego, co znalezione

Test jednostkowy na poziomie `_add_rolling_features` przeszedł, a bug
i tak przeciekł do produkcji -- to znaczy, że pokrycie testami jest zbyt
wąskie, niezależnie od tego, co dokładnie się znajdzie w kroku 1. Dodaj
nowy test na poziomie **pełnego `build_features()`**, nie pojedynczej
funkcji wewnętrznej: dla syntetycznego meczu z `neutral_venue=True`,
zamiana która drużyna jest "home"/"away" na arkuszu (przy identycznych
wartościach źródłowych dla obu drużyn) musi dać lustrzane wiersze cech --
`home_X` w wariancie zamienionym musi równać się `away_X` w oryginale, dla
KAŻDEJ kolumny inżynieryjnej, nie tylko rolling. To jest właściwie
diagnostyka z kroku 1, zapisana jako trwały test regresyjny, żeby ta klasa
błędu nie wróciła cicho przy następnej zmianie.

## 4. Dopiero teraz: retrening -- ale węziej niż poprzednio

`run_training()` (`training/orchestrator.py:478`) woła `build_features()`
raz dla całego pliku meczów -- każdy z 18 targetów w `TARGET_TASK_MAPPING`
konsumuje dokładnie tę samą ramkę cech, więc fix z kroku 2a/2b technicznie
dotyczy wszystkich. Nie ma w kodzie skrótu "przelicz tylko wagi na starych
hiperparametrach" -- `train_target()` zawsze woła `tune_model()` od zera
(pełny Optuna search, 4 modele bazowe, per target). Ale nie trzeba płacić
za to tak jak za pierwszy pełny przebieg:

- **Pomiń `btts`, `over_1_5`, `over_2_5`, `over_3_5` w ogóle.** Potwierdzone
  w `predictions/service.py`: serwis nigdy nie woła zarejestrowanego modelu
  dla tych czterech targetów -- zawsze liczy je z `home_goals`/`away_goals`
  przez rozkład Poissona (`_poisson_btts_probability`/`_poisson_over_probability`,
  z komentarzem "CONFIRMED REAL... beat a directly-trained... classifier").
  Wytrenowanie ich to policzalny koszt bez żadnego efektu w produkcji --
  zostaw poza `--targets`.
- **Obniż `--n-trials`** względem pierwszego przebiegu. Projekt ma już
  zapisaną, świadomą decyzję (`PLAN_SILNIKA.md`), że kolejne punkty
  procentowe ze strojenia nie mają znaczenia biznesowego -- płacenie za
  pełną głębię Optuny drugi raz z rzędu nie jest tu uzasadnione tak samo
  jak nie było uzasadnione podnoszenie tego wcześniej.
- **Zrób to w dwóch turach, nie jednej.** Najpierw `match_outcome`,
  `home_goals`, `away_goals` -- to jedyne targety z realnie zdiagnozowanym,
  zmierzonym objawem (krok 5, ponowna diagnostyka home-advantage). Potwierdź
  że fix faktycznie usunął anomalię NA TYCH trzech, dopiero potem odpal
  resztę (corners/cards/shots/possession, 10 targetów) w drugiej turze --
  nie warto płacić za pełen zestaw, jeśli krok 5 pokaże, że trzeba wracać
  do kroku 1.

Użyj tej samej komendy co w `PROMPT_ZAMKNIECIE_WATKOW.md` (`--targets`, nie
`--target-names`), np. `--targets match_outcome,home_goals,away_goals` dla
pierwszej tury.

## 5. Finalna, trzecia diagnostyka home-advantage

Odpal ponownie skrypt z `PROMPT_HOME_ADVANTAGE_CHECK.md` na nowym modelu.
Tym razem oczekujemy realnie małego `swing_neutral` na surowych
prawdopodobieństwach modelu -- jeśli nadal nie jest mały, wracamy do kroku
1, nie zgadujemy dalej.

## Co zgłosić z powrotem

1. Pełna lista kolumn łamiących symetrię z kroku 1, z wartościami po obu
   stronach.
2. Którą ścieżkę wybrałeś (2a czy 2b) i dlaczego -- jeśli 2b, zatrzymaj się
   tam i nie idź dalej do kroków 3-5 bez naszej decyzji.
3. Treść nowego testu regresyjnego z kroku 3 i potwierdzenie, że przechodzi
   (i że faktycznie by wykrył pierwotny bug, gdyby istniał wcześniej).
4. Metryki retreningu z kroku 4.
5. Wynik finalnej diagnostyki z kroku 5 -- surowe prawdopodobieństwa,
   swing przed/po, dla obu par.

## Czego nie robić

- Nie zgaduj mechanizmu przecieku bez uruchomienia kroku 1 -- to była
  dokładnie pułapka poprzednim razem (fix poprawny lokalnie, niewystarczający
  globalnie).
- Nie wracaj do `_resolve_outcome_probabilities`/`_heuristic_prediction` --
  to już wykluczone, potwierdzone poprzednią diagnostyką.
- Nie idź do kroku 4 (retrening) jeśli krok 1 wskazał na ścieżkę 2b i nie
  podjęliśmy jeszcze wspólnej decyzji co robić z tym mechanizmem.
