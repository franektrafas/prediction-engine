# Prompt do Claude Code — realne geokodowanie pod Travel Fatigue Index

> Kontekst: budujemy Travel Fatigue Index (dystans, zmiana strefy czasowej,
> dni odpoczynku między meczami) jako jawnie oznaczony kontekst w
> `streamlit_app.py` — nie cechę modelu, tylko coś obok predykcji z jasnym
> podpisem "nie wchodzi do modelu". Sam już przygotowuję resztę (dystans
> Haversine, wiązanie z resztą UI) — brakuje mi tylko wiarygodnych,
> sprawdzalnych współrzędnych/stref czasowych. Nie mam w moim środowisku
> ani bibliotek geokodujących, ani dostępu do sieci, więc to jedyny kawałek,
> którego nie mogę zrobić sam bez zgadywania współrzędnych z pamięci — a przy
> produkcie sprzedającym się na sprawdzalności nie chcę wpisywać
> lat/long na pamięć bez źródła. Skopiuj wszystko poniżej linii.

---

## Zadanie

Zbuduj mały, samodzielny moduł (proponuję
`football_ai/data/geocoding.py`) dający real, źródłowe współrzędne +
strefę czasową dla lokalizacji meczów międzynarodowych — nie coś ręcznie
wpisanego, tylko z prawdziwego zbioru danych.

**Rekomendacja**: `pip install pycountry` (kraj → dane ISO, stabilne, nie
wymaga stałego dostępu do sieci po instalacji) + jeden statyczny,
**źródłowy** CSV/plik z współrzędnymi stolic/dużych miast (np. z
GeoNames albo innego publicznego, cytowalnego źródła — nie z pamięci
modelu). Jeśli wolisz, `timezonefinder` (offline, działa z lat/long, nie
wymaga API) do wyliczenia strefy czasowej z koordynatów zamiast osobnej
tabeli stref.

## Ważna komplikacja, na którą trafiłem — nie zakładaj, że to proste

Dwa różne źródła danych meczowych mają **różne pola lokalizacji**, i to
nie jest to samo pole pod inną nazwą:

- Historyczny `records_frame` (martj42, `international_results.py`) ma
  `city`/`country` — bez nazwy stadionu.
- Nadchodzące mecze Ligi Narodów (`uefa_nations_league.py`,
  `fetch_group_matches`) mają tylko **`stadium`** (nazwa stadionu z
  Wikipedii) — bez osobnego miasta/kraju.

Czyli moduł potrzebuje **dwóch ścieżek**: `city + country -> (lat, lon,
tz)` dla danych historycznych, i coś, co obsłuży samą nazwę stadionu dla
nadchodzących meczów (np. mała, jawnie źródłowa tabela
`nazwa_stadionu -> (miasto, kraj)` dla stadionów, które faktycznie
występują w danych Ligi Narodów — nie każdy stadion świata, tylko te
realnie używane). Jeśli nazwy stadionu nie da się rozpoznać — moduł ma
zwrócić `None`, nie zgadywać, żeby UI mógł to po cichu pominąć (ten sam
wzorzec co reszta appki: brak danych = brak wiersza, nie placeholder).

## Interfejs, którego potrzebuję (żebym mógł to od razu podłączyć)

```python
def coordinates_for_city_country(city: str | None, country: str) -> tuple[float, float, str] | None:
    """(lat, lon, IANA timezone name) albo None, jeśli nie da się ustalić."""

def coordinates_for_stadium(stadium_name: str) -> tuple[float, float, str] | None:
    """To samo, po nazwie stadionu -- dla nadchodzących meczów bez city/country."""
```

Dowolne rozsądne nazwy są ok, ale zachowaj ten kształt (dwie funkcje, ten
sam typ zwracany, `None` na brak danych) — resztę (Haversine, UI, dni
odpoczynku) mam już gotowe i podłączę się pod to.

## Weryfikacja

1. `ruff check .` przechodzi.
2. Test jednostkowy sprawdzający kilka znanych par miast dają sensowny
   dystans (np. Warszawa–Londyn ~1450 km, nie 14500 ani 145) — sanity
   check rzędu wielkości, nie dokładność co do metra.
3. Podaj mi, skąd dokładnie wzięły się współrzędne (nazwa źródła, nie
   "z internetu") — chcę to móc zacytować, jeśli ktoś zapyta.

## Czego nie robić

- Nie zgaduj współrzędnych z pamięci modelu — to dokładnie to, czego
  unikamy.
- Nie buduj pełnego geokodowania każdego miasta świata — tylko kraje UEFA
  (Liga Narodów) plus to, co realnie pojawia się w `records_frame` dla
  meczów, które user faktycznie ogląda.
- Nie zmieniaj `streamlit_app.py` — to robię ja, żeby nie było konfliktu
  edycji na tym samym pliku.
