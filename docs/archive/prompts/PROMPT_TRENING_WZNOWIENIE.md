# Prompt do Claude Code — wznowienie treningu: statystyki meczowe

> Kontekst: PLAN_SILNIKA.md mówi, że tuning silnika jest ogólnie wstrzymany
> na rzecz warstw nad silnikiem — to nie jest wyjątek od tej zasady, tylko
> dokończenie już rozpoczętego, przerwanego treningu tych samych targetów.
> Skopiuj wszystko poniżej linii.

---

Dokończ trening dla targetów statystykowych, które dziś (2026-08-19) miały
zostać wytrenowane, ale run został przerwany. Cel: żeby kartki, strzały,
strzały celne i posiadanie piłki w `/predict` liczyły się realnie per mecz,
zamiast wracać do zahardkodowanej stałej.

## Diagnoza — dlaczego to jest przerwane, nie zepsute

`predictions/service.py::_predict_regression()` zwraca literalną stałą
(`default=...`) dla każdego targetu, dla którego `ModelRegistry` nie ma
zarejestrowanego modelu — bez żadnego związku z drużynami. Sprawdzony stan
`football_ai/models/artifacts/registry.json` po przerwaniu:

- Zarejestrowane w pełni (nie ruszać): `match_outcome`, `home_goals`,
  `away_goals`, `over_1_5/2_5/3_5`, `home_corners`, `away_corners`.
- **Częściowo, do dokończenia**: `home_yellow_cards` ma tylko
  `random_forest` i `lightgbm` — `catboost` był w trakcie tuningu
  (Optuna study istnieje, ale bez wyniku), `xgboost`, `stacking_ensemble`
  i finalny `selected_model` nigdy nie powstały. Nie trzeba nic czyścić
  ręcznie w registry przed ponownym uruchomieniem — `ModelRegistry.register()`
  tylko dopisuje wpisy, a `train_target()` i tak doda brakujące modele i na
  końcu zarejestruje `selected_model`, który `get_latest()` preferuje.
- **W ogóle brak wpisu**: `away_yellow_cards`, `home_shots`, `away_shots`,
  `home_shots_on_target`, `away_shots_on_target`, `home_possession`,
  `away_possession`.
- **Celowo pomijane, nie naprawiać w tej rundzie**: `home_xg`/`away_xg` —
  to nie bug, tylko realny brak pokrycia w danych StatsBomb (rzadkie xG
  dla meczów międzynarodowych). `run_training()` i tak je pominie z
  logiem — to poprawne zachowanie, nie regresja do sprawdzenia.

**Bug nazewniczy dla posiadania piłki już naprawiony, nie dotykaj go
ponownie.** `features/builder.py::build_features()` ma teraz na samej
górze alias (`home_possession_pct` → `home_possession`,
`away_possession_pct` → `away_possession`) z komentarzem zaczynającym się
od "CONFIRMED REAL naming mismatch (found 2026-08-19...". Sprawdź, że ten
blok tam jest, zanim zaczniesz — jeśli go nie ma, coś nadpisało zmiany i
trzeba to zgłosić, a nie trenować dalej na starym kodzie (bez tego aliasu
`home_possession`/`away_possession` i tak zawsze zostaną cicho pominięte,
niezależnie ile razy odpalisz trening).

## A. Przed uruchomieniem

1. Upewnij się, że **żaden proces `training.orchestrator` nie działa w
   tle** (poprzedni run się wywalił/został przerwany, ale sprawdź realnie,
   nie zakładaj). `ModelRegistry.save()` robi pełny odczyt-modyfikacja-zapis
   `registry.json` bez blokady — dwa równoległe procesy piszące w to samo
   nadpiszą sobie wzajemnie wpisy.
2. Ustal, jakiego pliku użyto do dzisiejszego runu (`v_local_stats_2026-08-19`,
   ten sam run co zarejestrował `home_corners`/`away_corners`) — nie mam
   tego zapisanego w żadnym committed skrypcie ani w historii poleceń,
   którą widziałbym z mojej strony. Sprawdź historię terminala/notebooka,
   albo po prostu wczytaj kandydatów przez pandas i zobacz, który faktycznie
   ma kolumny `home_corners`/`home_yellow_cards`/`home_possession_pct`:
   `football_ai/data/processed/international_results_training_history.parquet`
   jest moim najlepszym typem (jedyny lokalny plik z realnym merge'em ESPN),
   ale **potwierdź to zamiast zakładać** — użycie innego pliku danych dla
   tych targetów niż dla już zarejestrowanych corners byłoby niespójne w
   ramach tego samego tagu wersji.
3. Odpal `pytest football_ai/tests` (albo
   `python -m unittest discover football_ai/tests`) raz przed startem — masz
   pełne zależności, więc to powinno przejść czysto (ja to sprawdzałem w
   okrojonym środowisku bez joblib/pydantic — u Ciebie powinno wyjść 0
   failures, nie tylko 0 w tym, co dało się zaimportować).

## B. Trening

Ten sam plik danych i ten sam tag wersji co dzisiejszy przerwany run (żeby
`home_yellow_cards` dokończyło się na identycznych danych, na których
zaczęło):

```
python -m football_ai.training.orchestrator \
  --matches-path <plik ustalony w kroku A.2> \
  --version v_local_stats_2026-08-19 \
  --target-names home_yellow_cards,away_yellow_cards,home_shots,away_shots,home_shots_on_target,away_shots_on_target,home_possession,away_possession
```

## C. Po treningu

1. Sprawdź `football_ai/models/artifacts/registry.json` — każdy z ośmiu
   targetów powyżej powinien mieć wpis `"model_name": "selected_model"`.
2. Realny test, nie tylko obecność w registry: policz `/predict` (albo
   przejdź przez panel diagnostyczny w Streamlicie — „Panel diagnostyczny —
   skąd pochodzi każda liczba") dla **dwóch różnych par drużyn** i potwierdź,
   że kartki/strzały/strzały celne/posiadanie faktycznie się różnią między
   meczami, a panel diagnostyczny pokazuje "Wytrenowany model", nie
   "Wartość domyślna (brak modelu)".
3. Zgłoś mi: jakiego pliku ostatecznie użyłeś w kroku A.2, i końcowy stan
   registry dla tych ośmiu targetów.

## Czego nie robić

- Nie ruszaj wersji/wpisów dla `match_outcome`, `home_goals`, `away_goals`,
  `over_1_5/2_5/3_5`, `home_corners`, `away_corners` — te są już gotowe.
- Nie odpalaj dwóch równoległych procesów `training.orchestrator`.
- Nie próbuj naprawić `home_xg`/`away_xg` w tej rundzie — to osobny,
  większy temat (brak danych, nie kod), nie szybka poprawka.
- Nie zmieniaj ponownie aliasu w `features/builder.py` — jest już
  naprawiony, tylko potwierdź że jest.
- Nie dotykaj `streamlit_app.py` — PDF dla „Perspektywy grupy" to osobny,
  już zrobiony wątek, zajmiemy się jego weryfikacją oddzielnie.
