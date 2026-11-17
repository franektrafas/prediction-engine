# Prompt do Claude Code — finalny retrening trójki + naprawa źródła niestabilności

> To już nie diagnostyka -- realny fix + finalny retrening, żeby zamknąć
> temat home-advantage. Nie szukaj niczego nowego, to jest wykonanie, nie
> dochodzenie. Skopiuj wszystko poniżej linii.

---

## Kontekst

Potwierdzone: dwa realne przecieki (`home_is_home`/`away_is_home`,
tożsamość drużyny jako cecha kategoryczna) zostały poprawnie znalezione i
naprawione. Resztkowa niestabilność między przebiegami (1.85x -> 0.267x na
tej samej komendzie) to szum z nieseedowanego samplera Optuny
(`optuna.create_study()` bez ustawionego samplera) w połączeniu z za
małym `--n-trials 10` na rzadkiej kombinacji danych (duży gap Elo +
neutralny grunt) -- nie strukturalny bug. Dobra decyzja, żeby nie
zostawić w registry modelu wybranego przez szczęśliwe ziarno.

## 1. Napraw źródło niestabilności, nie tylko jej objaw

W `training/tuning.py`, gdziekolwiek `optuna.create_study()` jest wołane
(wewnątrz `tune_model()`) -- dodaj jawny, ustawiony sampler zamiast
polegać na domyślnym, niezaseedowanym zachowaniu:

```python
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(..., sampler=sampler)
```

`42` dla spójności z istniejącym `random_state=42` na modelach bazowych.
To dotyczy wszystkich targetów przez współdzieloną `tune_model()`, nie
tylko tej trójki -- napraw raz, skorzysta cała reszta pipeline'u na
przyszłość. Bez tego każdy kolejny retrening (nie tylko dziś) ma tę samą
ukrytą niereprodukowalność, po prostu nie zawsze się ujawni tak wyraźnie
jak tutaj.

## 2. Finalny retrening trójki z `--n-trials 20`

`match_outcome`, `home_goals`, `away_goals`, nowy tag wersji (nie
`v_stabilnosc_test_2026-08-28` -- to zostaje oznaczone jako testowe/
tymczasowe, nie produkcyjne). Z jawnym seedem z kroku 1 -- powtórzenie tej
samej komendy powinno teraz dawać ten sam wynik, nie loterię.

## 3. Sprzątnij registry

Upewnij się, że `v_stabilnosc_test_2026-08-28` nie zostaje jako
`selected_model`/najnowszy wpis dla `match_outcome` po zarejestrowaniu
nowej, właściwej wersji -- sprawdź `ModelRegistry.get_latest("match_outcome")`
po zakończeniu, powinien wskazywać na nowy tag, nie testowy.

## 4. Finalna diagnostyka -- ostatni raz, żeby potwierdzić, nie zakładać

Odpal `PROMPT_HOME_ADVANTAGE_CHECK.md` jeszcze raz na tym modelu. Oczekiwany,
ale wciąż do zweryfikowania, nie do założenia: obie pary pokazują poprawny
kierunek, i wynik jest stabilny (nie musisz powtarzać dwa razy dla
weryfikacji stabilności -- to już zweryfikowaliśmy jako mechanizm w
poprzednim kroku; teraz chodzi o jeden wiarygodny, oparty na realnym
strojeniu wynik).

## Co zgłosić z powrotem

1. Potwierdzenie, że sampler ma jawny seed, i gdzie dokładnie w kodzie.
2. Metryki finalnego retreningu trójki.
3. Stan registry po sprzątnięciu -- `selected_model` dla `match_outcome`
   wskazuje na właściwy, nowy tag.
4. Finalne liczby diagnostyki dla obu par -- to jest ten wynik, który
   idzie do dokumentacji jako zamknięcie tematu.

## Czego nie robić

- Nie zostawiaj `v_stabilnosc_test_2026-08-28` jako aktywnego modelu w
  registry.
- Nie zaczynaj drugiej/trzeciej tury (corners/cards/shots/possession) --
  to osobno, wciąż zablokowane przez ESPN/raport jakości danych.
