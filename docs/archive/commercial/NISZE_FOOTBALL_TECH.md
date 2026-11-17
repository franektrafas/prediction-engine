# Niezamknięte nisze w football tech

**Data:** 29 lipca 2026
**Zakres:** analiza niezależna od naszego projektu. Szukam realnych luk — niekoniecznie
statystycznych, niekoniecznie dużych.
**Zasada:** luka musi wynikać z dowodu (wydarzenie, badanie, dane rynkowe), a nie z wyobraźni.

---

## 1. Dwie zasady, którymi się kierowałem

Zanim przejdę do listy — sposób szukania, bo on decyduje o jakości wyniku.

**Zasada 1: najlepsze nisze tworzy niedawne wydarzenie albo niedawno powstała rola.**
Nie „obszar, który wygląda na słaby", tylko konkretna zmiana, po której rynek nie zdążył
się przestawić. Incumbent ma wtedy złą pozycję, bo zbudował się pod stary stan świata.
Znalazłem dwa takie przypadki (sekcje 2 i 5).

**Zasada 2: szukaj segmentu użytkowników, dla którego nikt nic nie buduje.**
Cała branża buduje dla zawodników, trenerów, klubów, kibiców i analityków. Jest jedna
grupa uczestników meczu, dla której nie buduje prawie nikt — i akurat ta grupa przeżywa
kryzys egzystencjalny (sekcja 3).

---

## 2. Odkrycie główne: zawalił się „commons" danych piłkarskich

**Co się stało:** 20 stycznia 2026 Stats Perform (właściciel Opty) zerwał umowę z FBref
i zażądał natychmiastowego usunięcia wszystkich zaawansowanych statystyk. Powód oficjalny:
naruszenie warunków umowy. Kontekst: stało się to **osiem dni po** ogłoszeniu Stats Perform
wyłącznym dystrybutorem danych bukmacherskich i praw streamingowych FIFA na Mundial 2026.
Środowisko powszechnie odczytało to jako ruch komercyjny — usunięcie najlepszego darmowego
źródła tych samych danych.

**Skutek:** FBref był przez dekadę fundamentem amatorskiej i akademickiej analityki
piłkarskiej. Nic go nie zastąpiło jeden do jednego, bo nie ma już żadnego darmowego źródła
rozdającego aktualne dane Opty. Cytat ze środowiska: *„społeczność analityki piłkarskiej
straciła swoje dobro wspólne, dane przeszły za drogie korporacyjne paywalle"*.

**Dlaczego to jest nisza, a nie tylko zła wiadomość:**

Powstała próżnia, w której są dziesiątki tysięcy ludzi — studenci, dziennikarze, blogerzy,
niezależni analitycy, skauci małych klubów, twórcy narzędzi — którzy stracili narzędzie pracy
i nie mają budżetu na Optę. To jest zdefiniowana, zmotywowana, łatwa do znalezienia grupa
odbiorców. Sześć miesięcy później wciąż bez rozwiązania.

**Kluczowa obserwacja strategiczna:** próbę odtworzenia FBref przez scraping cudzych danych
zabije ten sam mechanizm — wystarczy jedno pismo od prawników. Obronne jest tylko rozwiązanie,
w którym **liczysz metryki samodzielnie z otwartych źródeł**, bo wtedy nie ma licencji do
cofnięcia. To jest różnica między „kolejnym agregatorem" a czymś, czego nie da się wyłączyć.

**Szczególnie dotkliwy podprzypadek — piłka kobieca.** Utrata FBref została opisana wprost
jako katastrofa dla danych o piłce kobiecej, bez gotowego następcy, a dysproporcja wobec
męskiej dopiero się pogłębi. Jednocześnie 76% kibiców piłki kobiecej deklaruje, że widzi
za mało jej pokrycia. Częściowe złagodzenie: Hudl StatsBomb udostępnił za darmo pięć pełnych
lig kobiecych — 771 meczów, 62 zespoły, 1500 zawodniczek.

To jest najlepszy punkt wejścia z całego dokumentu: **świeża luka + darmowe dane pod ręką
+ życzliwa społeczność + zero konkurencji**.

---

## 3. Segment bez produktów: sędziowie

Skala problemu, w liczbach:

- **Jeden na siedmiu sędziów rzuca funkcję co roku.**
- W 55 federacjach UEFA brakuje **około 40 000 sędziów**.
- Główna przyczyna to przemoc słowna i fizyczna, dalej: obciążenie finansowe, brak szacunku,
  brak wsparcia organizacyjnego.
- Skutek wtórny: odwoływane mecze, a w dłuższym terminie brak kadry na poziomie elity,
  bo elita rekrutuje się z podstaw.

**Dlaczego to jest interesujące produktowo:** przejrzyj dowolne zestawienie startupów
football tech — wszystko jest dla zawodnika, trenera, klubu, skauta, kibica albo analityka.
Sędzia jest uczestnikiem każdego meczu na świecie i praktycznie nikt nie buduje dla niego
narzędzi. To jest rzadka sytuacja: duży, zdefiniowany segment bez oferty.

**Uczciwa trudność:** sędzia indywidualny nie ma pieniędzy. Płacić musiałby związek, liga
albo federacja — czyli instytucja, a sprzedaż do instytucji jest wolna i polityczna.
To jest prawdziwy powód, dla którego nikt tego nie robi, i trzeba to przyjąć do wiadomości,
zamiast zakładać, że „nikt na to nie wpadł".

---

## 4. Luka wdrożeniowa, nie wiedzowa: prewencja urazów

Znalezisko z badań, które lubię najbardziej, bo jest kontrintuicyjne:

**Większość amatorskich zespołów nie stosuje podstawowych elementów treningu
nerwowo-mięśniowego zapobiegającego zerwaniom ACL i skręceniom kostki** — mimo że programy
te są znane, darmowe, przebadane i skuteczne.

To znaczy, że problemem **nie jest brak wiedzy ani brak modelu predykcyjnego**. Wiedza istnieje
i jest za darmo. Problem jest czysto wdrożeniowy: nikt tego nie robi w praktyce, bo brakuje
nawyku, przypomnienia, struktury i kontroli.

Do tego, z badań nad amatorami: kluby nie mają zaplecza medycznego, a decyzję „czy zawodnik
z bólem może grać" podejmuje trener bez kompetencji medycznych i bez wsparcia.

**To jest wzorzec do zapamiętania i szukania w innych miejscach: sprawdzone rozwiązanie
istnieje, jest darmowe, i nikt go nie używa.** Zamknięcie takiej luki nie wymaga wynalazku —
wymaga produktu, który zmienia zachowanie. To jest znacznie łatwiejsze technicznie
i znacznie trudniejsze produktowo.

---

## 5. Rola, która powstała, zanim powstały narzędzia: set pieces

- Ze stałych fragmentów pada **25–35% wszystkich bramek** w czołowych ligach Europy.
- Przed sezonem 2025/26: **10 klubów Premier League ma dedykowanego trenera od stałych
  fragmentów, dwa rekrutują, osiem nie ma nikogo.**
- Rola powstała realnie w ostatnich ~3 latach i szybko się upowszechnia.

Poniżej Premier League specjalistów praktycznie nie ma, a analiza stałych fragmentów odbywa
się klatka po klatce, ręcznie.

**Nisza:** nowa rola zawodowa, która jeszcze nie ma własnego oprogramowania. Trenerzy od
stałych fragmentów pracują dziś w narzędziach nieprzeznaczonych do tego (edytory wideo,
tablice taktyczne, Excel). Historycznie każda nowa rola w sporcie w końcu dostaje swój
dedykowany software — tu jest okno.

---

## 6. Pozostałe zidentyfikowane luki

| Luka | Dowód | Wielkość | Trudność |
|---|---|---|---|
| Pustynie danych poza Europą | Z 54 krajów Afryki niektórzy dostawcy pokrywają tylko RPA | Duża | Wysoka (trzeba tworzyć dane) |
| Fragmentacja narzędzi grassroots | 68% z 400 badanych klubów w UK używa 3+ niepołączonych platform | Duża | **Rynek zajęty** — patrz sekcja 8 |
| Brak wsparcia psychospołecznego przy końcu kariery amatorskiej | Badania: „ogólny brak wsparcia psychospołecznego w doświadczeniu tranzycji" | Średnia | Wysoka (nie jest to problem software'owy) |
| Bariery finansowe uczestnictwa (sprzęt, składki, transport) | Badania nad uczestnictwem mniejszości w klubach amatorskich | Duża | Wysoka (problem społeczny, nie produktowy) |
| Brak taniego wsparcia sędziowskiego (odpowiednik VAR) | Pojedynczy stadion z SAOT to koszt milionów; poniżej elity nie istnieje nic | Duża | Bardzo wysoka (CV, sprzęt) |

---

## 7. Osiem konkretnych pomysłów

Format zbliżony do raportu źródłowego. Świadomie mieszam duże i bardzo małe.

---

### 1. Otwarty, niezależny „commons" statystyk piłkarskich

**Luka:** sekcja 2. **Odbiorcy:** analitycy niezależni, dziennikarze, studenci, blogerzy, twórcy narzędzi.

**Opis:** nie agregator cudzych danych, tylko **własna warstwa obliczeniowa** na otwartych
źródłach (StatsBomb Open Data, schochastics, martj42, otwarte dane zdarzeniowe). Publikuje
metryki, które zniknęły z FBref, ale liczone samodzielnie i opisane metodologicznie.
Kluczowa cecha: **nie ma licencji, którą można cofnąć** — to jedyna obrona przed powtórką
stycznia 2026.

**Technologie:** Python, parquet, publiczny API + prosty frontend, dokumentacja metodologii.

**MVP:** jedna liga, jeden sezon, 15 metryk, publiczny endpoint + strona z tabelą.

**Złożoność:** średnia. **Model:** open core (darmowe dane, płatne API/eksporty), sponsoring,
Patreon, konsulting. **Ryzyka:** monetyzacja dobra wspólnego jest notorycznie trudna;
utrzymanie wymaga ciągłej pracy; jakość musi być bardzo wysoka, bo porównanie z FBref
jest nieuniknione.

---

### 2. To samo, ale wyłącznie piłka kobieca ⭐ *wersja mała*

**Luka:** sekcja 2, podprzypadek. **Odbiorcy:** ta sama grupa, ale znacznie mniejsza i bardziej zdesperowana.

**Opis:** identyczna idea zawężona do lig kobiecych. Przewaga: Hudl StatsBomb właśnie
udostępnił za darmo pięć pełnych lig (771 meczów, 62 zespoły, 1500 zawodniczek), więc
**dane wejściowe są gotowe i legalne**. Konkurencja: praktycznie zero. Społeczność:
życzliwa i głośna.

**MVP:** strona ze statystykami zaawansowanymi dla pięciu lig z darmowego zbioru StatsBomb,
odświeżana po każdej kolejce.

**Złożoność:** **niska–średnia** — dane są gotowe, robota jest w obliczeniach i prezentacji.

**Model:** sponsoring, granty (piłka kobieca ma dostęp do finansowania rozwojowego),
później API. **Ryzyka:** mały rynek komercyjny; ryzyko, że StatsBomb rozszerzy własne darmowe
udostępnianie i zajmie miejsce.

**To jest mój typ numer jeden na „małą, realnie zamykalną lukę".**

---

### 3. Towarzysz pierwszego sezonu młodego sędziego ⭐ *wersja mała*

**Luka:** sekcja 3. **Odbiorcy:** sędziowie w pierwszym roku, związki sędziowskie.

**Opis:** nie „aplikacja dla sędziów" w ogóle, tylko wąsko: **pierwsze 10–20 meczów**, kiedy
odpada najwięcej osób. Przygotowanie przed meczem, checklisty, szybkie zgłoszenie incydentu
jednym kliknięciem (dziś: papierowe formularze, których nikt nie wypełnia), kontakt z mentorem,
krótka rozmowa po meczu. Cel produktu nie jest analityczny — jest retencyjny.

**MVP:** aplikacja mobilna dla jednego okręgu sędziowskiego, funkcja zgłaszania incydentu
+ mentor.

**Złożoność:** niska technicznie. **Model:** licencja dla związku/okręgu; potencjalnie
finansowanie z programów federacyjnych, bo brak sędziów to ich bezpośredni problem operacyjny.

**Ryzyka:** sprzedaż wyłącznie instytucjonalna (wolna, polityczna); sędzia indywidualny nie
zapłaci; wymaga zaufania i współpracy związku od pierwszego dnia. **Ale:** to jest problem,
który federacje realnie próbują rozwiązać i mają na to budżety.

---

### 4. Narzędzie wdrażania prewencji urazów (nie predykcji)

**Luka:** sekcja 4. **Odbiorcy:** kluby amatorskie i młodzieżowe, federacje, **ubezpieczyciele**.

**Opis:** 10-minutowa rozgrzewka oparta na sprawdzonych programach neuromięśniowych, wpięta
w rutynę treningu: wideo, kolejność ćwiczeń, śledzenie tego, czy zespół faktycznie to robi.
Zero modelowania — cała wartość leży w zmianie zachowania i mierzeniu adherencji.

**Ciekawy trop monetyzacyjny:** ubezpieczyciel ma bezpośredni interes finansowy w mniejszej
liczbie zerwanych więzadeł. To potencjalnie płatnik, o którym nikt nie myśli, bo wszyscy
patrzą na kluby.

**Złożoność:** niska. **Ryzyka:** adherencja to znany, twardy problem (aplikacje fitness mają
z tym to samo); wymaga treści wideo; efekt widać po sezonach, nie tygodniach.

---

### 5. Edytor i biblioteka stałych fragmentów dla nie-elity

**Luka:** sekcja 5. **Odbiorcy:** trenerzy poniżej top lig, akademie, drugie ligi.

**Opis:** narzędzie rzemieślnicze, nie analityczne: projektowanie wariantów rożnych i wolnych,
biblioteka schematów z opisem ról, animacja, eksport do wydruku i na telefon zawodnika.
Rola set-piece coacha powstała szybciej niż jej oprogramowanie.

**Złożoność:** niska–średnia (edytor 2D + animacja). **Model:** subskrypcja klubowa.
**Ryzyka:** trenerzy niższych szczebli mają minimalne budżety; ryzyko, że wystarczy im tablica.

---

### 6. Warstwa integracyjna dla klubów grassroots (nie kolejna platforma)

**Luka:** 68% klubów używa 3+ niepołączonych systemów. **Uwaga:** rynek platform jest zajęty
(Spond, TeamStats, LoveAdmin i inni). Dlatego jedyny sensowny wariant to **nie budować kolejnej
platformy, tylko spinać istniejące**.

**Złożoność:** średnia (integracje to zawsze więcej pracy, niż się wydaje).
**Ryzyko główne:** uzależnienie od cudzych API, które mogą zostać zamknięte — dokładnie ten
sam mechanizm, który zabił FBref.

**Ocena: raczej pułapka niż nisza.** Wymieniam dla porządku.

---

### 7. Relacja z meczu, którego nikt nie transmituje

**Luka:** niższe ligi i młodzież nie mają żadnego pokrycia. Rodzic czy kibic nie ma jak
dowiedzieć się, co się stało.

**Opis:** narzędzie dla jednego wolontariusza na trybunie: kilka przycisków (gol, kartka,
zmiana, sytuacja), z których automatycznie powstaje czytelna relacja tekstowa do udostępnienia.
Człowiek dostarcza obserwację, system dostarcza formę.

**Złożoność:** niska. **Model:** freemium dla klubów, licencja dla związków.
**Ryzyka:** zależność od tego, że komuś będzie się chciało klikać przez 90 minut —
to jest całe ryzyko produktu i trzeba je przetestować w pierwszej kolejności.

---

### 8. „Dlaczego przegraliśmy" dla zespołów bez analityka

**Opis:** jeden ekran po meczu, w języku naturalnym, bez wykresów, dla trenera, który nie
umie i nie chce czytać dashboardów. Badania wskazują, że problemem jest tłumaczenie analiz
na praktyczne kroki, a narzędzia wymagające ręcznego wgrywania nagrań i tagowania zdarzeń
tworzą „drugą pracę".

**Złożoność:** zależna od źródła danych — i to jest główny problem, bo na tym poziomie
danych zwykle nie ma. **Ryzyko:** bez wsadu danych to jest wydmuszka.

---

## 8. Pozorne luki — czyli gdzie nie iść

Uczciwość wymaga wskazania rzeczy, które wyglądają na nisze, a nie są:

| Wygląda na lukę | Dlaczego nie |
|---|---|
| Zarządzanie klubem grassroots | Rynek zajęty przez Spond, TeamStats, LoveAdmin i innych. Realny problem to integracja, nie brak platformy |
| Automatyczna analiza wideo z telefonu | Zajęte (BigBro.AI i podobni), a wejście wymaga zespołu CV i kapitału |
| Kolejne API z predykcjami | Sufit dokładności + darmowy lepszy substytut w postaci kursów |
| Scraping cudzych danych jako fundament produktu | Styczeń 2026 pokazał dokładnie, jak to się kończy |
| Tanie VAR dla amatorów | Koszt sprzętu i CV poza zasięgiem małego zespołu, mimo realnej potrzeby |

---

## 9. Moje trzy typy i uzasadnienie

Gdybym miał wybrać, w kolejności:

**1. Statystyki piłki kobiecej (pomysł 2).**
Świeża luka po konkretnym wydarzeniu, darmowe legalne dane już przygotowane przez StatsBomb,
zerowa konkurencja, życzliwa i głośna społeczność, mały zakres na start. Największy stosunek
realności do wysiłku z całej listy.

**2. Towarzysz młodego sędziego (pomysł 3).**
Największy niezagospodarowany segment użytkowników w całej piłce, problem o skali 40 000 osób,
instytucje z budżetem i realną motywacją. Najtrudniejsza sprzedaż, ale najmniej zatłoczone pole.

**3. Wdrażanie prewencji urazów (pomysł 4).**
Bo pokazuje wzorzec, który warto umieć rozpoznawać: sprawdzone, darmowe rozwiązanie, którego
nikt nie stosuje. Plus nietypowy płatnik (ubezpieczyciel).

---

## 10. Najsłabszy punkt tej analizy

Wszystko powyżej pochodzi z badań, publikacji i danych rynkowych — **czyli z drugiej ręki**.
Ani jedna z tych nisz nie została potwierdzona rozmową z człowiekiem, który ma ten problem.

To jest dokładnie ten sam błąd, przed którym ostrzegałem przy naszym własnym produkcie:
analiza zastępująca kontakt z użytkownikiem. Dokument tego typu jest dobry do **zawężenia
listy z dwudziestu do trzech**. Nie nadaje się do wyboru jednej i zaczęcia kodowania.

Najtańszy następny krok dla dowolnego z tych pomysłów: znaleźć pięć osób z danym problemem
i zapytać, co robią dzisiaj zamiast. Jeśli odpowiedź brzmi „nic, bo to nie jest aż taki
problem" — nisza jest pozorna, niezależnie od tego, jak dobrze wygląda w raporcie.

---

## Źródła

**Upadek FBref / Opta:**
- [An Ecological Solution to Opta Killing FBRef's Football Statistics](https://medium.com/@johncomonitski/an-ecological-solution-to-opta-killing-fbrefs-football-statistics-1b5231dcc076)
- [FBref loses Opta data access — omówienie skutków](https://www.linkedin.com/posts/theapfa_the-news-regarding-fbref-losing-access-to-activity-7419767688773746690-CTdr)
- [FBRef Alternatives](https://footyapps.com/guide/fbref-alternatives)
- [Where to Find Football Data in 2026 — Liam Henshaw](https://www.liamhenshaw.com/writing/where-to-find-football-data)

**Piłka kobieca:**
- [Loss of FBref advanced stats is disaster for women's soccer data — The IX](https://www.theixsports.com/the-ix-soccer/fbrefs-loss-advanced-stats-womens-soccer-data-accessibility/)
- [Hudl StatsBomb — darmowe dane pięciu lig kobiecych](https://www.hudl.com/blog/statsbomb-free-womens-data-wsl-ligaf-bundesliga-seriea-nwsl)
- [Women's football State of Play 2026](https://www.sgieurope.com/consumer/womens-football-state-of-play/120009.article)

**Kryzys sędziowski:**
- [Football's referee crisis — badanie na tysiącach sędziów (The Conversation)](https://theconversation.com/footballs-referee-crisis-we-asked-thousands-of-refs-about-the-abuse-and-violence-thats-driving-them-out-of-the-game-231886)
- [European football faces a referee shortage — ESPN](https://www.espn.com/soccer/story/_/id/38452288/european-soccer-elite-referees-premier-league-etc-blame)
- [What Keeps Soccer Referees from Renewing Their Licenses](https://thesportjournal.org/article/for-the-good-of-the-game-what-keeps-soccer-referees-from-renewing-their-licenses/)

**Prewencja urazów i amatorzy:**
- [Most amateur football teams do not implement essential neuromuscular training (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9007793/)
- [Amateur football coaches on perceptions of pain during participation (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11243123/)
- [Predicting Dropout From Organized Football (Frontiers)](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2021.752884/pdf)

**Stałe fragmenty:**
- [Premier League 2025–26: Set Play Coaches](https://www.creativesetplays.com/post/premier-league-set-piece-coaches-2025-26)
- [The Role of the Set-Piece Coach In Modern Football](https://the-footballanalyst.com/the-role-of-the-set-piece-coach-in-modern-football/)

**Grassroots i pokrycie danych:**
- [Europe Just Told Grassroots Football It Deserves Better Digital Tools](https://tacticosport.com/news/eu-grassroots-football-digital-tools)
- [Emerging Football Markets: Why Data Coverage Beyond Europe Matters](https://www.sportmonks.com/blogs/emerging-football-markets-why-data-coverage-beyond-europe-matters/)
- [Football Analytics in 2026: The Tools Reshaping the Game](https://firsttouchonline.com/football-analytics-in-2026/)

**Sędziowanie / technologia:**
- [R-VARS: From Pro Leagues to Grassroots (ACM)](https://dl.acm.org/doi/10.1145/3730436.3730473)
