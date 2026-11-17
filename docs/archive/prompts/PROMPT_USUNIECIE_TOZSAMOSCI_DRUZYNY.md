# Prompt do Claude Code — usunąć home_team_name/away_team_name jako cechy kategoryczne

> Decyzja podjęta: usuwamy, nie symetryzujemy, nie akceptujemy rezydualnego
> obciążenia. Uzasadnienie w `PLAN_SILNIKA.md`/rozmowie: duży rozjazd Elo +
> neutralny grunt to profil meczu grupowego mistrzostw świata, czyli rdzeń
> tego, co ten silnik ma przewidywać, nie margines. Ale zrób najpierw
> tanią ablację (krok 1), zanim usuniesz na stałe -- jeśli koszt w
> metrykach okaże się zaskakująco duży, zatrzymaj się i zgłoś, zamiast
> kontynuować mimo dowodu przeciwnego. Skopiuj wszystko poniżej linii.

---

## Kontekst

Potwierdzone SHAP-em: dla Polska/Niemcy (duży gap Elo) `home_team_name`
jest najważniejszą pojedynczą cechą w predykcji, i to konkretnie dla
Niemiec jako historycznie zdominowanej kategorii -- niosącej wyuczony
prior "Niemcy jako home_team_name" niezależny od `neutral_venue` i od
jakiejkolwiek cechy numerycznej (w tym już naprawionego `home_is_home`).
Portugalia/Holandia, para bardziej "generyczna", nie pokazuje tego samego
efektu w tym stopniu -- potwierdza to, że to zależy od historycznej
lopsidedness konkretnej drużyny, nie jest stałym artefaktem.

`features/builder.py:729-733` definiuje:

```python
categorical_columns = [
    column
    for column in ["tournament", "country", "home_team_name", "away_team_name"]
    if column in feature_frame.columns
]
```

## 1. Ablacja -- zmierz koszt przed usunięciem na stałe

Wytrenuj `match_outcome` dwukrotnie na tych samych danych: raz z obecnym
`categorical_columns` (zawiera `home_team_name`/`away_team_name`), raz z
`categorical_columns` ograniczonym do `["tournament", "country"]`. Użyj
tego samego `--n-trials 10`, tego samego podziału train/val/test. Porównaj
F1/accuracy na walidacji i teście.

Jeśli różnica jest mała (obie wersje w podobnym rzędzie wielkości) --
usunięcie jest tanie, idź do kroku 2. Jeśli różnica jest duża (realny,
niebagatelny spadek jakości bez tych cech) -- **zatrzymaj się, zgłoś to z
liczbami, nie usuwaj mimo to.** To by znaczyło, że te kolumny niosą więcej
realnego sygnału niż zakładaliśmy, i decyzja wymaga ponownego przemyślenia
(np. symetryzacja zamiast usunięcia), nie mechanicznego wykonania tego
promptu.

## 2. Usuń `home_team_name`/`away_team_name` z `categorical_columns`

W `features/builder.py` -- ogranicz `categorical_columns` do
`["tournament", "country"]` (albo cokolwiek innego zostanie po
weryfikacji, że nie ma tam ukrytej analogicznej pułapki -- sprawdź czy
`country`/`tournament` same nie kodują pośrednio tożsamości drużyny w
sposób, który mógłby odtworzyć ten sam problem; jeśli nie widzisz takiego
mechanizmu, nie szukaj go na siłę, to nie jest oczekiwane).

Sprawdź, czy same kolumny `home_team_name`/`away_team_name` są gdzieś
jeszcze potrzebne poza `feature_columns` (np. do merge'y, do
`_add_h2h_features`, do raportowania) -- usuń je tylko z listy cech
podawanych modelowi, nie z ramki danych w ogóle, jeśli są używane gdzie
indziej jako klucz.

## 3. To zmiana we współdzielonej infrastrukturze -- zakres retreningu

`categorical_columns` wpływa na wszystkie targety, nie tylko
`match_outcome`. Dołącz ten fix do tej samej puli co pozostały,
zablokowany retrening (`match_outcome`/`home_goals`/`away_goals` z
`PROMPT_NAPRAWA_PRZECIEKU_HOME.md` już to obejmował -- ten fix wchodzi w
ten sam batch, nie osobno). Nowy tag wersji (Optuna nie powinna dziedziczyć
prób z zestawu cech sprzed usunięcia tożsamości drużyny).

## 4. Finalna diagnostyka -- trzeci raz

Odpal `PROMPT_HOME_ADVANTAGE_CHECK.md` jeszcze raz na obu parach. Tym
razem oczekujemy, że Polska/Niemcy też pokaże właściwy kierunek (swing
neutralny wyraźnie mniejszy niż przy realnym gospodarzu), nie tylko
"bliżej".

## Co zgłosić z powrotem

1. Wyniki ablacji z kroku 1 -- liczby, nie ocena "mały"/"duży" bez liczb.
2. Czy usunięcie faktycznie nastąpiło i gdzie (pliki/linie).
3. Metryki retreningu z kroku 3.
4. Finalna diagnostyka -- czy Polska/Niemcy jest teraz w prawidłowym
   kierunku, nie tylko bliżej niego.

## Czego nie robić

- Nie usuwaj cech bez wyniku ablacji z kroku 1.
- Nie trenuj tego osobno od reszty zablokowanego retreningu -- to ta sama
  pula, jeden batch.
