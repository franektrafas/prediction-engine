# Kierunki odbudowy produktu football_ai

**Data:** 29 lipca 2026
**Cel dokumentu:** odpowiedź na pytanie „czy cały produkt jest do spalenia" + konkretne
warianty odbudowy w formacie zbliżonym do raportu źródłowego.

---

## 1. Nie, produkt nie jest do spalenia

Poprzednia analiza była krytyczna wobec **jednego zdania**: „sprzedajemy prawdopodobieństwa
1X2 jako produkt". To zdanie jest nie do obrony. Ale to jest może 10% projektu — warstwa
pozycjonowania, nie inżynierii.

### Co realnie mamy (aktywa przenośne do każdego wariantu)

| Aktywo | Wartość | Czy do wyrzucenia? |
|---|---|---|
| Silnik 19 targetów (1X2, gole, xG, rożne, kartki, strzały, posiadanie) | Wysoka — rzadkie, większość projektów robi tylko 1X2 | **Nie** |
| Leak-safe pipeline cech + chronologiczna walidacja | **Bardzo wysoka** — to jest rzecz, którą 90% projektów robi źle | **Nie** |
| Rejestr modeli, wersjonowanie artefaktów | Średnia — infrastruktura produkcyjna | **Nie** |
| FastAPI + PostgreSQL + SQLAlchemy | Średnia — standard, ale działa | **Nie** |
| Elo, pogoda, dystans podróży, cechy składu | Wysoka — gotowy wsad do każdego wariantu | **Nie** |
| Infrastruktura SHAP | Wysoka — fundament pod „wyjaśnialność jako produkt" | **Nie** |
| Testy, CI, struktura modułowa | Średnia | **Nie** |
| **Pozycjonowanie „B2B API z predykcjami"** | **Ujemna** — kieruje wysiłek w ślepy zaułek | **Tak** |
| Docelowa lista 6 typów klientów naraz | Ujemna — rozprasza | **Tak** |
| GNN (szkielet) | Niska — brak ścieżki do przychodu | Zamrozić |

**Podsumowanie: ~85% pracy jest do wykorzystania. Zmienia się to, co jest na wierzchu.**

Analogia: zbudowałeś dobry silnik. Problem polega na tym, że próbujesz sprzedawać sam silnik
ludziom, którzy mogą dostać lepszy za darmo. Silnik jest w porządku — trzeba zbudować wokół
niego pojazd i wybrać, kogo ma wozić.

---

## 2. Ograniczenia, które muszą kształtować wybór

Bycie uczciwym wobec realiów jest ważniejsze niż wybranie najciekawszego pomysłu.

| Ograniczenie | Konsekwencja |
|---|---|
| Jedna osoba, w trakcie nauki programowania | Odpada wszystko, co wymaga zespołu (CV/tracking, zbieranie danych ręcznie) |
| Brak licencji na dane | Odpada konkurowanie zakresem danych |
| Brak doświadczenia i sieci w sprzedaży B2B | Sprzedaż enterprise do mediów = bardzo trudna ścieżka |
| Brak kapitału / inwestorów | **To jest też przewaga** — patrz niżej |
| Trening w Colabie | Odpada nic wymagającego ciągłego, ciężkiego treningu |
| Rynek polski jako baza | Mały rynek, ale mała konkurencja i przewaga językowa |

**Przewaga wynikająca z braku kapitału:** Opta i Gracenote mają ogromne koszty stałe (zespoły
zbierające dane, zespoły redakcyjne, sprzedaż). Muszą wyceniać się enterprise'owo. Każdy rynek,
w którym całkowity możliwy przychód to <200 tys. € rocznie, jest dla nich **niewidzialny**.
Dla jednej osoby 40–60 tys. € rocznie to bardzo dobry wynik.

**To definiuje strefę, w której da się wygrać: rynki za małe dla gigantów, wystarczające dla jednej osoby.**

---

## 3. Trzy odkrycia rynkowe, które zmieniają rekomendacje

### 3.1 Kalendarz: piłka reprezentacyjna nie utrzyma produktu medialnego

Sprawdziłem realny kalendarz. Mundial 2026 właśnie się skończył (czerwiec–lipiec 2026).
Co dalej:

| Wydarzenie | Termin |
|---|---|
| Liga Narodów 2026/27 — faza grupowa | 24–26 września 2026 (start) |
| Ćwierćfinały LN | marzec 2027 |
| Finały LN | 9–13 czerwca 2027 |
| Eliminacje Euro 2028 | marzec 2027 – listopad 2027 |
| Baraże Euro 2028 | marzec 2028 |
| Euro 2028 (Anglia, Szkocja, Walia, Irlandia) | 9 czerwca – 9 lipca 2028 |

**Wniosek: od dziś do końca września nie ma praktycznie nic.** Piłka reprezentacyjna to
~30–40 dni meczowych rocznie, w skupiskach. Produkt medialny lub subskrypcyjny oparty
wyłącznie na reprezentacjach ma przychód zerowy przez większość roku.

Piłka klubowa: co tydzień, od sierpnia do maja.

**To odwraca część mojej poprzedniej rekomendacji.** Wcześniej odradzałem top 5 lig
(bo brak różnicowania — i to nadal prawda). Ale **klubowa piłka jest konieczna dla samej
częstotliwości produktu**, niezależnie od różnicowania. Rozstrzygnięcie: piłka klubowa tak,
ale przewaga musi pochodzić z czegoś innego niż „mamy statystyki o Realu".

### 3.2 Prawo: dane Ekstraklasy są mniej zamknięte, niż deklaruje Ekstraklasa

Wcześniej zapisałem to jako twardą barierę, bo Ekstraklasa SA deklaruje własność wszystkich
statystyk i wymóg pisemnej zgody. Sprawdziłem orzecznictwo — obraz jest inny.

Trybunał Sprawiedliwości UE w sprawach **BHB v William Hill** oraz **Fixtures Marketing** (2004)
orzekł, że organizatorzy zawodów **nie mają prawa sui generis do bazy danych** w odniesieniu
do danych, które sami *tworzą* (terminarze, listy uczestników). Kluczowe rozróżnienie:
prawo do bazy chroni inwestycję w *pozyskanie, weryfikację i prezentację* danych już
istniejących — a nie inwestycję w ich *wykreowanie*.

Wynik meczu jest faktem. Fakty nie podlegają prawu autorskiemu.

**Zastrzeżenie — nie jestem prawnikiem i to nie jest porada prawna.** Realne komplikacje:
- scraping strony może naruszać regulamin serwisu (to odrębna kwestia — prawo umów, nie IP),
- szczegółowe statystyki, w których zbieranie i weryfikację liga faktycznie inwestuje
  (posiadanie, podania, mapy strzałów), mogą być chronione inaczej niż same wyniki,
- polska implementacja dyrektywy (ustawa o ochronie baz danych) i 20 lat rozwoju orzecznictwa.

**Praktyczny wniosek:** przed komercjalizacją trzeba to skonsultować z prawnikiem od IP,
ale bariera jest prawdopodobnie znacznie niższa, niż sugeruje nota na stronie ligi.
To odblokowuje wariant „polskie ligi".

### 3.3 Rynek fantasy jest realny i ma udowodnione modele biznesowe

- Fantasy Premier League ma miliony graczy; sezon **2026/27 wystartował 23 lipca 2026**
  (sześć dni temu).
- Fantasy Football Scout: 350 000+ menedżerów FPL, 13 osób, model subskrypcyjny,
  narzędzia premium za opłatą.
- Obok działają mniejsze byty: Premier Fantasy Tools, LiveFPL, FPL Pulse.

To dowodzi, że **B2C w piłce jest opłacalne przy skali osiągalnej dla małego zespołu**,
i nie wymaga sprzedaży enterprise ani licencji na dane premium.

---

## 4. Sześć konkretnych wariantów odbudowy

Format zgodny z raportem źródłowym: cel, opis, dane, technologie, MVP, złożoność,
model przychodów, ryzyka.

---

### Wariant A — Silnik symulacji sezonu i turnieju („co jeszcze jest możliwe")

**Cel:** kibice + media, jako pierwszy krok i generator treści.

**Opis:** Monte Carlo nad istniejącym silnikiem prawdopodobieństw. Symuluje pozostałą część
sezonu/turnieju 10 000 razy i odpowiada na pytania, na które nie ma darmowej odpowiedzi:
szanse na mistrzostwo, awans, spadek, europejskie puchary; scenariusze „co jeśli"
(kontuzja kluczowego zawodnika, wygrana w derbach). Wyniki są z natury cytowalne
i nagłówkowe.

**Dane:** to, co już mamy — wyniki, Elo, siła zespołów. Zero nowych źródeł.

**Technologie:** NumPy, symulacja Monte Carlo, istniejący `predictions/service.py`.

**MVP:** strona z jedną ligą i jedną tabelą: „szanse każdej drużyny na mistrzostwo / spadek /
europejskie puchary", aktualizowana po każdej kolejce.

**Złożoność:** **Niska.** To jest najtańsza rzecz do zbudowania z tego, co już istnieje —
tygodnie, nie miesiące.

**Model przychodów:** sam w sobie słaby (reklama, ewentualnie licencja sezonowa dla portalu).
Jego prawdziwa rola to **magnes na ruch i dowód działania** dla wariantów B i C.

**Ryzyka:** niski bezpośredni przychód; FiveThirtyEight robił to za darmo przez lata
i ustawił oczekiwanie, że takie rzeczy są darmowe; poza sezonem martwy.

---

### Wariant B — Asystent decyzji do gier fantasy i typerów (B2C, subskrypcja)

**Cel:** gracze Fantasy Premier League, Fantasy Ekstraklasa, ligi typerskie.

**Opis:** narzędzie odpowiadające na jedno pytanie tygodniowo: „kogo wystawić / na kogo
postawić i dlaczego". Nasz silnik przewiduje gole, xG, strzały i kartki na poziomie drużyny —
to trzeba rozłożyć na zawodników przez historyczny udział w akcjach drużyny (mamy już
`lineups` i `player_reference`). Wyróżnik: **uzasadnienie** przy każdej rekomendacji
(SHAP → zdanie po polsku), zamiast gołego rankingu.

**Dane:** publiczne API FPL (darmowe), nasze predykcje drużynowe, historia występów zawodników.

**Technologie:** istniejący silnik + model rozkładu drużyna→zawodnik + prosty frontend.

**MVP:** strona z rankingiem zawodników na najbliższą kolejkę + „dlaczego" przy każdym,
+ planer transferów. Wersja darmowa ograniczona, płatna pełna.

**Złożoność:** **Średnia.** Nowy element to modelowanie na poziomie zawodnika; reszta istnieje.

**Model przychodów:** subskrypcja 15–25 zł/mies., freemium. Przy 500 płacących = ~10 tys. zł/mies.
Nie wymaga sprzedaży, rozmów, umów ani licencji.

**Ryzyka:** rynek anglojęzyczny zatłoczony (Fantasy Football Scout ma Optę i 13 osób);
zależność od API FPL, które może się zmienić; sezon FPL już wystartował — najlepszy moment
na pozyskanie użytkowników minął, kolejny za rok; wymaga marketingu, którego nie ćwiczyłeś.

**Dlaczego to mimo wszystko najmocniejsza ścieżka przychodowa:** to jedyny wariant, w którym
klient płaci sam, od razu, bez procesu zakupowego, umowy i działu prawnego po drugiej stronie.

---

### Wariant C — Widget „dlaczego" dla polskich portali sportowych (B2B2C)

**Cel:** Meczyki.pl, WP SportoweFakty, Weszło, Przegląd Sportowy, mniejsze portale klubowe.

**Opis:** osadzalny komponent przy zapowiedzi meczu: prognoza + **trzy powody w języku
naturalnym** („Polska 43% — przewaga Elo 68 pkt, 5 dni więcej odpoczynku, rywal pokonał
3 200 km"). Portale walczą o czas na stronie; wyjaśniona prognoza to treść angażująca,
a nie kolejna liczba.

**Dane:** nasze predykcje + SHAP.

**Technologie:** istniejący silnik, szablony narracyjne (deterministyczne, nie LLM),
lekki widget JS, endpoint publiczny.

**MVP:** jeden widget, jeden portal, za darmo przez sezon w zamian za logo i case study.

**Złożoność:** **Niska–średnia** technicznie. Trudność leży w sprzedaży, nie w kodzie.

**Model przychodów:** licencja miesięczna za portal (1–3 tys. zł), lub podział przychodu
reklamowego.

**Ryzyka:** **sprzedaż B2B bez doświadczenia i referencji jest bardzo trudna**; portale mogą
mieć już kontrakty z dostawcami danych; decyzja może wymagać zgody wydawnictwa (Ringier Axel
Springer przy Przeglądzie Sportowym); cykl zakupowy liczony w miesiącach.

---

### Wariant D — „Redakcja faktów" dla polskich lig (zawężony pomysł 5)

**Cel:** dziennikarze i komentatorzy Ekstraklasy, I ligi, rozgrywek młodzieżowych.

**Opis:** zweryfikowany pakiet faktów przed i po meczu — serie, rekordy, kamienie milowe,
nietypowe zbiegi okoliczności — dla lig, których Gracenote i Opta nie obsługują z polską
głębią. Rdzeniem jest **warstwa weryfikacji** (każdy fakt z odnośnikiem do konkretnych
meczów w bazie), a nie generowanie tekstu.

**Dane:** pełna historia polskich rozgrywek — **to jest główny problem**, bo trzeba ją zbudować
od zera; martj42 i schochastics nie mają głębi dla polskich lig.

**Technologie:** PostgreSQL, deterministyczne reguły wykrywania „nietypowości", testy
na znanych faktach historycznych (golden cases).

**MVP:** jedna liga, jeden sezon wstecz, pakiet 10 faktów na mecz w formacie JSON + PDF.

**Złożoność:** **Średnia–wysoka**, głównie z powodu pozyskania i weryfikacji danych.

**Model przychodów:** abonament dla redakcji (500–2000 zł/mies.).

**Ryzyka:** kwestia prawna (patrz 3.2) — wymaga konsultacji; budowa bazy historycznej to
miesiące żmudnej pracy; mały rynek (kilkanaście redakcji w Polsce); odpowiedzialność
za błąd rzeczowy opublikowany przez klienta.

**Największa wartość długoterminowa, najwolniejszy start.**

---

### Wariant E — Otwarty kod jako kapitał zawodowy

**Cel:** Ty sam — rynek pracy / zlecenia.

**Opis:** football_ai jako publiczne, dopracowane repozytorium pokazujące rzecz, którą
**mało kto potrafi udowodnić**: poprawny time-series ML bez przecieku danych, chronologiczna
walidacja, rejestr modeli, testy. Naprawione przez nas błędy (kolizja kolumn, losowy CV
w stackingu) to gotowy materiał na artykuł techniczny — bo to są błędy, które popełnia
większość projektów w tej dziedzinie.

**Dane:** publiczne (martj42, schochastics) — żadnych kwestii licencyjnych.

**Technologie:** to, co jest, plus dokumentacja i przykłady.

**MVP:** repozytorium z czytelnym README, notebookiem demonstracyjnym i jednym wpisem
blogowym „jak wykryliśmy przeciek danych w silniku predykcji piłkarskich".

**Złożoność:** **Niska.**

**Model przychodów:** pośredni — praca, staż, zlecenia freelance, reputacja.

**Ryzyka:** zero bezpośredniego przychodu; nie jest to „produkt".

**Uczciwie: dla osoby uczącej się programowania to może być najwyższa oczekiwana wartość
z całej listy.** Dobre publiczne repozytorium z rzadką kompetencją daje konkretne oferty
pracy. Nieudany startup nie daje nic poza doświadczeniem.

---

### Wariant F — Modele rynków niszowych (rożne, kartki) jako feed danych

**Cel:** serwisy analityczne i porównywarki.

**Opis:** wykorzystanie faktu, że nasz silnik przewiduje rzeczy, których prawie nikt nie
modeluje — rożne, kartki, strzały celne. Rynki mniej płynne niż 1X2, więc kursy są mniej
efektywne, a konkurencja modelowa rzadsza.

**Złożoność:** Niska (już mamy).

**Model przychodów:** feed API.

**Ryzyka — i są poważne:** odbiorcami są w praktyce podmioty okołobukmacherskie.
Polskie prawo hazardowe jest restrykcyjne, a reklama i promocja zakładów obwarowana.
Doszłaby też kwestia odpowiedzialności — produkty zwiększające zaangażowanie w hazard mają
realny potencjał szkody dla użytkowników. **Wymienić dla kompletności, ale odradzam**,
zwłaszcza że sam wykluczyłeś operatorów zakładów z listy klientów.

---

## 5. Tabela porównawcza

| Wariant | Złożoność | Wykorzystanie obecnego kodu | Ścieżka przychodu | Wymaga sprzedaży B2B | Ryzyko | Ocena |
|---|---|---|---|---|---|---|
| **A. Symulacje** | Niska | ~90% | Słaba bezpośrednio | Nie | Niskie | Zrób pierwszy |
| **B. Fantasy/typer** | Średnia | ~70% | **Najlepsza** | Nie | Średnie | **Główny kandydat** |
| **C. Widget dla portali** | Niska–śr. | ~85% | Średnia | **Tak** | Średnie | Po walidacji A |
| **D. Redakcja faktów PL** | Śr.–wysoka | ~50% | Średnia | **Tak** | Wysokie (dane, prawo) | Cel długoterminowy |
| **E. Open source** | Niska | 100% | Pośrednia | Nie | Bardzo niskie | **Zrób równolegle** |
| **F. Rynki niszowe** | Niska | ~95% | Niejasna | Tak | Wysokie (prawne, etyczne) | Odradzam |

---

## 6. Rekomendacja: ścieżka etapowa, nie jeden zakład

Nie wybieraj jednego wariantu na ślepo. Kolejność, w której każdy krok zmniejsza niepewność
przed następnym:

**Etap 0 (teraz, ~1 tydzień) — rozmowy, zero kodu.**
Pięć rozmów: dwóch dziennikarzy sportowych, trzech aktywnych graczy fantasy/typera.
Pytanie do dziennikarzy: jak przygotowujesz się do meczu, co jest najbardziej upierdliwe.
Pytanie do graczy: za co zapłaciłbyś 20 zł miesięcznie. To rozstrzyga między ścieżką B a C/D
**zanim** napiszesz linijkę kodu.

**Etap 1 (~3–4 tygodnie) — Wariant A.**
Symulacje Monte Carlo nad istniejącym silnikiem. Powód: najtańsze, wykorzystuje 90% tego,
co jest, i daje **publiczny, pokazywalny artefakt**. Start Ligi Narodów 24 września to
naturalny termin.

**Etap 2 (równolegle, ciągle) — Wariant E.**
Porządkowanie repozytorium jako publicznego portfolio. Zero dodatkowej pracy — to głównie
dokumentacja tego, co i tak robisz. Zabezpiecza wartość projektu niezależnie od wyniku biznesowego.

**Etap 3 (~2–3 miesiące) — Wariant B albo C, w zależności od wyniku Etapu 0.**
Jeśli rozmowy z graczami fantasy pokażą gotowość do płacenia → B.
Jeśli któryś portal wykaże realne zainteresowanie → C (i wtedy A staje się demem sprzedażowym).

**Etap 4 (rok+) — Wariant D**, jeśli wcześniejsze etapy dadzą przychód i referencje,
i jeśli konsultacja prawna wypadnie pomyślnie.

---

## 7. Co porzucić definitywnie

- **„B2B API z predykcjami" jako pozycjonowanie** — powód w poprzedniej analizie.
- **Sześć segmentów klientów naraz** — wybrać jeden.
- **GNN** — zamrozić, brak ścieżki do przychodu przy jednoosobowym zespole.
- **Dalsze dostrajanie modelu jako główna aktywność** — model jest wystarczająco dobry
  dla każdego z powyższych wariantów. Kolejne 2% skuteczności nie zmieni niczego biznesowo.
- **Rozszerzanie na kolejne ligi „bo można"** — tylko pod konkretny wariant.

---

## 8. Jedno zdanie na koniec

Silnik jest dobry i został właśnie naprawiony z dwóch poważnych błędów — to realna wartość.
Problem nigdy nie leżał w kodzie, tylko w tym, że nie było zdefiniowanego człowieka,
który ma za to zapłacić i dlaczego. Etap 0 kosztuje tydzień i pięć rozmów, a rozstrzyga
pytanie, na które sam kod nie odpowie.

---

## Źródła

- [Fantasy Football Scout — cennik i model subskrypcyjny](https://www.fantasyfootballscout.co.uk/pricing)
- [Fantasy Football Scout — start sezonu FPL 2026/27 (23.07.2026)](https://www.fantasyfootballscout.co.uk/2026/07/23/fantasy-premier-league-fpl-2026-27-is-live)
- [Best FPL Tools 2026/27 — przegląd konkurencji](https://www.fplpulse.com/blog/best-fpl-tools-2025-26)
- [Terminarz Ligi Narodów UEFA 2026/27](https://www.meczyki.pl/liga/liga-narodow-uefa/1661/terminarz)
- [Kalendarz rozgrywek 2026/27](https://rankinguefa.pl/kalendarz-2026-27/)
- [Terminy Euro 2028, Copa América i innych turniejów](https://www.foxsports.com/stories/soccer/world-cup-euro-copa-america-gold-cup-afcon-afc-cup)
- [Database Right — wyroki ECJ w BHB v William Hill i Fixtures Marketing](https://www.mondaq.com/uk/it-and-internet/44318/database-right-and-the-ecj-judgments-in-bhb-v-william-hill-and-the-fixtures-marketing-cases)
- [Bazy danych — ochrona sui generis a prawo autorskie (Kluwer)](https://legalblogs.wolterskluwer.com/copyright-blog/databases-sui-generis-protection-and-copyright-protection/)
- [EUR-Lex — wyrok C-203/02 (BHB v William Hill)](https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:62002CJ0203)
- [Ekstraklasa SA — statystyki i warunki wykorzystania](https://www.ekstraklasa.org/statystyki/)
- [Meczyki.pl](https://www.meczyki.pl/) · [WP SportoweFakty](https://sportowefakty.wp.pl/) · [Weszło](https://weszlo.com/)
