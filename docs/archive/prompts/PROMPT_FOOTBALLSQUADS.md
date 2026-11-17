# Prompt do Claude Code — footballsquads.co.uk: realne, historyczne składy

> Nowe źródło danych, świadomie odblokowane w `PLAN_SILNIKA.md` (notatka z
> 20 sierpnia 2026) po tym, jak dotychczasowa pionowa ścieżka (Kroki 1-5)
> okazała się już zbudowana i przetestowana. To realny, spory zastrzyk
> danych — planuj to w fazach, nie jako jeden wielki skrypt. Skopiuj
> wszystko poniżej linii.

---

## Kontekst

`features/builder.py` ma już dziś pola `home_lineup_average_age`/
`home_lineup_market_value`/... , ale komentarz przy ich wykluczeniu z
`excluded_columns` mówi wprost: *"Live-only enrichment fields ... always
zero-filled in historical training data (no point-in-time historical
record exists, only current snapshots)"*. To źródło to realnie zmienia —
pierwszy raz mamy prawdziwe, historyczne, per-turniejowe składy, nie tylko
bieżący snapshot.

**footballsquads.co.uk** — działa od 1999, aktywnie utrzymywana (widziałem
wpisy z sierpnia 2026), linkowana wprost z rsssf.org jako afiliowany
serwis. `robots.txt` blokuje tylko `/cgi-bin/` i `/phpBB2/` (forum) —
strony składów są wprost dozwolone do scrapowania, ale scrapuj grzecznie
(cache, opóźnienia między requestami, sensowny User-Agent).

Dwie osobne części serwisu, różnej wielkości:

1. **Składy reprezentacji na turniejach** (`/national.htm`) — MŚ od 1950,
   Euro od 1980, Copa América, Puchar Narodów Afryki, Puchar
   Konfederacji. Mniejszy, dobrze ograniczony zakres — **od tego zacznij**.
2. **Składy klubowe** (`/squads.htm` bieżący sezon, `/archive.htm`
   historyczne) — **NIE ROBIMY TEGO.** Rozważane wcześniej jako "Faza 2"
   pod proxy siły ligi klubu zawodnika, ale po namyśle (20 sierpnia 2026)
   to nieuzasadnione: składy reprezentacji z Fazy 1 już dają `Current
   Club` per zawodnik. Jeśli kiedyś potrzebny będzie sygnał "gra w mocnej
   czy słabej lidze", to mała tabela referencyjna klub→liga→kraj, nie
   pełny scraping rosterów dla dziesiątek lig i sezonów. Nie buduj tego
   bez osobnego, konkretnego uzasadnienia.

Potwierdzony realnie kształt strony składu (Polska, MŚ 2022,
`/national/worldcup/wc2022/poland.htm`) — tabela z kolumnami: `Number`,
`Name`, `Nat`, `Pos`, `Height`, `Weight`, `Date of Birth`, `Birth Place`,
`Current Club`. Struktura URL: `/national/{competition}.htm` (lista lat) →
`/national/{competition}/{rok}.htm` (lista grup/drużyn) →
`/national/{competition}/{rok}/{drużyna}.htm` (tabela zawodników).
Potwierdzone slugi: `worldcup` (wc1950...wc2026), `eurocham`
(euro1980...euro2024), `copaamer`, `africacn`, `confed`.

**AKTUALIZACJA — nie trzeba parsować HTML w ogóle.** Istnieje gotowy,
CC0-licencjonowany cache: `github.com/footballcsv/cache.footballsquads` —
zamienia strony footballsquads.co.uk na proste pliki tekstowe/CSV,
zachowując oryginalne dane 1:1 ("What's unchanged: All names, dates, etc
get copied as is incl. typos and errors"). Ma folder `national/` z
dokładnie tą samą strukturą co źródło (`worldcup/wc1950`...`wc2022`,
`eurocham`, `copaamer`, `africacn`, `confed` — potwierdziłem to realnie,
przeglądając repo), plus osobne foldery per kraj dla lig klubowych (faza 2).
Buduj Fazę 1 tak jak `international_results.py` buduje się z martj42 —
`fetch_text()` na `raw.githubusercontent.com/footballcsv/cache.footballsquads/master/...`,
`pd.read_csv(StringIO(...))`, żadnego `BeautifulSoup`/scrapowania HTML.

**Jedna rzecz do zweryfikowania, nie zakładać:** przykładowy CSV w
README repo (Liverpool, klub) ma ostatnią kolumnę `Previous Club`, a
żywa strona footballsquads.co.uk dla kadry na MŚ (Polska 2022) miała
`Current Club`. To może się różnić między plikami klubowymi a
reprezentacyjnymi, albo być niespójne między latami — sprawdź nazwę
nagłówka faktycznie w plikach `national/`, nie zakładaj z tego przykładu.
Dokładnej nazwy/rozszerzenia pliku per drużyna (`poland.csv`? `.txt`?) też
nie potwierdziłem do końca przez GitHub UI — zajrzyj do środka jednego
realnego pliku przed pisaniem parsera.

## Ustalone indeksy (20 sierpnia 2026) — licz te cztery w Fazie 1

Po rozmowie z userem, konkretna lista, nie "co ci wpadnie do głowy":

1. **Wiek** — średni wiek kadry w dniu startu turnieju (główny cel, patrz
   niżej), plus odchylenie std wieku (balans doświadczenie/energia), plus
   średni wiek per grupa pozycji (GK/D/M/F osobno — starzejąca się
   defensywa czy starzejący się atak to różne sygnały).
2. **Fizyczność** — średni wzrost/waga całej kadry, plus **osobno per
   grupa pozycji** (średni wzrost obrońców/bramkarzy jako proxy zagrożenia
   przy stałych fragmentach gry jest sensowniejszy niż blanket average
   26 zawodników). Wskaźnik budowy (waga/wzrost²) jako czysto
   eksploracyjny "fajny stat", bez pretensji do sygnału predykcyjnego.
3. **Ciągłość kadry** — % zawodników z poprzedniego dużego turnieju tej
   samej drużyny (już opisane niżej w dokumencie), plus % debiutantów.
4. **Club Cohesion Index** ("chemia kadrowa, bezpieczna wersja") — patrz
   sekcja osobna poniżej, to jest nowe i wymaga własnego wyjaśnienia.

### Club Cohesion Index — czym jest i dlaczego to NIE jest zakazany "Team Chemistry"

`PROMPT_DATA_ENGINE.md` wprost mówi: *"Nie implementuj Team Chemistry
zależnego od przewidywanej XI, dopóki nie mamy wiarygodnego i legalnego
źródła aktualnych składów."* Ten indeks świadomie omija dokładnie ten
warunek graniczny: liczy się z **pełnej, ogłoszonej kadry** (26
zawodników, fakt, znany na dniach przed turniejem), nigdy z
**przewidywanego składu wyjściowego** (11 zawodników, spekulacja). To
różnica, nie technikalia — pełna kadra jest faktem z provenance, skład
wyjściowy jest zgadywaniem.

**Mechanika:** dla każdej pary zawodników w kadrze sprawdź, czy `Current
Club` się zgadza. Nie licz tylko płaskiej liczby par — policz **wielkość
największego skupiska** (np. "5 zawodników z Realu Madryt" to jeden węzeł
wielkości 5, dający C(5,2)=10 połączonych par, nie 5 osobnych,
niepowiązanych relacji). Klub z jednym dominującym klastrem gra inaczej
niż drużyna, w której każdy zawodnik jest z innego klubu, nawet przy tej
samej płaskiej liczbie par.

**Rozszerzenie (opcjonalne, Faza 1b):** to samo policzone **per grupa
pozycji** — ile obrońców w kadrze gra dziś razem w jednym klubie, ile
pomocników — bo to bliżej realnego skautingu ("czwórka obronna, która zna
się z klubu, traci mniej bramek razem") niż niezróżnicowana liczba dla
całej 26-osobowej kadry, i wciąż nie wymaga zgadywania składu wyjściowego.

**Jak to pokazać — najpierw fakt, nie cecha modelu.** Zanim to
kiedykolwiek trafi do `features/builder.py` jako liczba wejściowa do
modelu, niech to najpierw żyje w `insights/`+`narrative/` jako
odtwarzalny fakt z provenance — dokładnie w duchu reszty tego modułu:
np. *"6 zawodników kadry Hiszpanii na tym turnieju gra na co dzień w
Realu Madryt"* jako gotowe zdanie do podglądu meczu/PDF-a, nie surowa
liczba. To zero ryzyka (nic nie wchodzi do treningu), od razu użyteczne,
i pasuje do istniejącej infrastruktury faktów. Dopiero jeśli/kiedy okaże
się realnie sygnałowe, rozważcie uproszczoną liczbową wersję jako
`home_minus_away` diff cechę — osobna, jawna decyzja, nie automat.

---

## Zadanie — Faza 1: składy turniejowe reprezentacji

Nowy moduł `football_ai/scrapers/sources/footballsquads.py`, ten sam wzorzec
co istniejące źródła (patrz `international_results.py`): klasa dziedzicząca
po `scrapers.base.BaseDataSource`, `source_name="footballsquads"`,
uruchamialny jako `python -m football_ai.scrapers.sources.footballsquads`.

Co policzyć per drużyna per turniej (jeden wiersz w wyjściowej tabeli):

- **Średni wiek kadry w dniu rozpoczęcia turnieju** (nie w dniu
  scrapowania!) — to jest główny cel, bo bezpośrednio domyka
  udokumentowaną lukę w `features/builder.py`. Potrzebna mała referencyjna
  tabela dat rozpoczęcia turniejów (albo wyprowadzona z najwcześniejszej
  daty meczu tego turnieju w już posiadanych danych martj42).
- Średni wzrost.
- Rozkład pozycji (liczba GK/D/M/F).
- Ciągłość kadry względem poprzedniego dużego turnieju tej samej drużyny
  (% zawodników z poprzedniej kadry) — dopasowanie po nazwisku jest
  przybliżone (pisownia bywa niespójna), oznacz to wprost jako
  best-effort, nie twardą liczbę.

Kanonikalizacja nazw drużyn: użyj istniejącego
`data.normalization.canonicalize_team_name` / `config/team_aliases.py` —
ta sama historia RFN/ZSRR/Jugosławia co w `international_results.py`.

Zapis: nowy parquet, np.
`football_ai/data/processed/tournament_squads.parquet` — jeden wiersz per
zawodnik per turniej (surowe dane), plus albo osobna tabela zagregowana
per drużyna-turniej, albo agregacja liczona na żądanie — Twoja decyzja,
opisz w kodzie którą wybrałeś i dlaczego.

## Ważna komplikacja — bądź uczciwy co do zasięgu

Skład turniejowy to zdjęcie z **jednego momentu** (start turnieju), nie
skład na każdy mecz. To wypełnia `home_lineup_average_age` tylko dla
meczów **na dużych turniejach lub blisko nich w czasie** — nie dla zwykłych
eliminacji czy sparingów między turniejami. Nie udawaj większego pokrycia,
niż realnie jest — ten sam wzorzec uczciwości co reszta projektu (patrz
jak `home_lineup_market_value` jest dziś opisane jako "informational
context, not a claimed model input" w innych miejscach, dopóki nie jest
prawdziwe).

Starsze turnieje (lata 50.-60.) mogą mieć rzadsze dane — brakujące
wzrost/waga/data urodzenia dla części albo wszystkich zawodników.
Brakująca komórka = `None`, nigdy nie zgaduj i nie dociągaj z pamięci
modelu.

## Weryfikacja

1. `ruff check .` przechodzi.
2. Sanity-check na znanej kadrze: policzyłem ręcznie z danych, które
   faktycznie pobrałem (Polska, MŚ 2022, start turnieju 2022-11-20) —
   średni wiek kadry wychodzi **około 27.0 roku** (ręczne liczenie z 26
   zawodników, do przeliczenia dokładnie w kodzie, nie brać tej liczby
   jako pewnik co do dziesiątych). To ten sam rodzaj testu co
   "Warszawa-Londyn ~1450 km" w `PROMPT_GEOCODING.md` — rząd wielkości, nie
   dokładność co do dnia.
3. Test na drużynie z rzadszymi danymi (jakiś zespół z MŚ 1950 albo 1954)
   — sprawdź, że brakujące pola nie wywalają całego pipeline'u, tylko
   zostają `None`.

## Czego nie robić

- Nie ruszaj składów klubowych (`squads.htm`/`archive.htm`) w ogóle —
  nieuzasadnione, patrz sekcja "Ustalone indeksy" wyżej.
- Nie wpinaj tego automatycznie do `features/builder.py` /
  `TARGET_TASK_MAPPING`. Zbierz dane, policz agregaty, zweryfikuj jakość —
  decyzja o użyciu jako cechy modelu to osobny krok, świadomie, tak jak
  przy błędzie neutral-venue (żadnej zmiany w `features/builder.py` bez
  jawnej rozmowy).
- Nie hamruj serwisu równoległymi requestami bez opóźnień — to
  hobbystyczna strona od 1999, nie API klasy enterprise.
