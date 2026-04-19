#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

#set page(paper: "a4", margin: 2cm, footer: context [
  #grid(
    columns: (1fr, 1fr, 1fr),
    // Creates 3 equal zones
    align: (left, center, right),
    // Aligns content within those zones

    [],
    // 1. EMPTY LEFT (Crucial for balancing)

    // 2. CENTER Page Number
    smallcaps[#context counter(page).display()],

    // 3. RIGHT Location & Date
    smallcaps[Rzeszów #datetime.today().display()],
  )
])

#set heading(numbering: "1.1.1.")
#set text(lang: "pl")
#set math.equation(numbering: "(1.1)")
#show figure: set block(breakable: true)

//Title page
#page(numbering: none, header: none, footer: none)[
  //TODO: Add prz images

  #align(center)[
    #text(weight: "extrabold", size: 1cm)[Sztuczna Inteligencja] //Subject

    #text(size: 0.5cm)[Projekt] // Laboratory number

    #text(size: 0.5cm)[Probabilistyczna sieć neuronowa] // Laboratory subject

    #align(bottom)[
      Hubert Dec $-$ 179474 $-$ L04 //Author information
      #linebreak()
      #datetime.today().display()
    ]

  ]
]

#outline()
#pagebreak()

= Opis problemu

= Część teoretyczna <teoria>

Probabilistyczna sieć neuronowa (PNN) #cite(<specht1990pnn>) jest siecią neuronową,
w której wyjście każdego neuronu interpretowane jest jako prawdopodobieństwo przynależności
do danej klasy. PNN opiera się na teorii estymacji gęstości jądrowej (ang. *Kernel Density
Estimation*, KDE) i jest szeroko stosowana w zadaniach klasyfikacji nadzorowanej.

== Model probabilistycznej sieci neuronowej

#figure(
  diagram(
    node-stroke: 1pt,
    edge-stroke: 1pt,
    node-fill: white,
    spacing: (1.5cm, 1cm),

    // Etykiety warstw
    node((0, -1), [*Input Layer* \ *Warstwa wejściowa*], stroke: none),
    node((1, -1), [*Pattern Layer* \ *Warstwa wzorców*], stroke: none),
    node((2, -1), [*Summation Layer* \ *Warstwa sumowania*], stroke: none),
    node((3, -1), [*Output Layer* \ *Warstwa wyjściowa*], stroke: none),

    // Neurony wejściowe
    node((0, 0.5), [$x_1$], shape: "circle", name: <i1>, width: 2.5em),
    node((0, 2.5), [$x_2$], shape: "circle", name: <i2>, width: 2.5em),

    // Neurony wzorcowe -- klasa A
    node((1, 0), [$P_("A1")$], shape: "circle", name: <pa1>, width: 2.5em),
    node((1, 1), [$P_("A2")$], shape: "circle", name: <pa2>, width: 2.5em),

    // Neurony wzorcowe -- klasa B
    node((1, 2), [$P_("B1")$], shape: "circle", name: <pb1>, width: 2.5em),
    node((1, 3), [$P_("B2")$], shape: "circle", name: <pb2>, width: 2.5em),

    // Neurony sumujące
    node((2, 0.5), [$Sigma_A$], shape: "circle", name: <sa>, width: 2.5em),
    node((2, 2.5), [$Sigma_B$], shape: "circle", name: <sb>, width: 2.5em),

    // Neuron wyjściowy
    node((3, 1.5), [Decision \ (Argmax)], shape: "circle", name: <out>, width: 5em),

    // Połączenia: wejście → wzorce
    edge(<i1>, <pa1>, "->", stroke: gray),
    edge(<i1>, <pa2>, "->", stroke: gray),
    edge(<i1>, <pb1>, "->", stroke: gray),
    edge(<i1>, <pb2>, "->", stroke: gray),

    edge(<i2>, <pa1>, "->", stroke: gray),
    edge(<i2>, <pa2>, "->", stroke: gray),
    edge(<i2>, <pb1>, "->", stroke: gray),
    edge(<i2>, <pb2>, "->", stroke: gray),

    // Połączenia: wzorce → sumatory (tylko w obrębie tej samej klasy)
    edge(<pa1>, <sa>, "->"),
    edge(<pa2>, <sa>, "->"),
    edge(<pb1>, <sb>, "->"),
    edge(<pb2>, <sb>, "->"),

    // Połączenia: sumatory → wyjście
    edge(<sa>, <out>, "->"),
    edge(<sb>, <out>, "->"),
  ),
  caption: [Model probabilistycznej sieci neuronowej (PNN)],
) <Diagram_PNN>

Poszczególne symbole użyte na @Diagram_PNN oznaczają:

- $x_1, x_2$ -- wejścia sieci (cechy danych wejściowych),
- $P_("A1"), P_("A2")$ -- neurony wzorcowe dla klasy $A$,
- $P_("B1"), P_("B2")$ -- neurony wzorcowe dla klasy $B$,
- $Sigma_A, Sigma_B$ -- neurony sumujące odpowiednio dla klas $A$ i $B$,
- Decision (Argmax) -- neuron decyzyjny przypisujący obserwację do klasy
  o najwyższej estymowanej gęstości prawdopodobieństwa.

=== Warstwa wejściowa (Input Layer)

Warstwa wejściowa nie wykonuje żadnych operacji obliczeniowych. Jej jedynym zadaniem jest
wprowadzenie wektora cech $bold(x) = [x_1, x_2, dots, x_p]^T in RR^p$ do sieci oraz
dystrybucja jego składowych do wszystkich neuronów warstwy wzorców. Liczba neuronów w tej
warstwie jest równa wymiarowości $p$ przestrzeni wejściowej.

=== Warstwa wzorców (Pattern Layer)

Warstwa wzorców stanowi bezpośrednią reprezentację zbioru uczącego -- każdy neuron
przechowuje dokładnie jeden wzorzec treningowy $bold(x)_(k j)$, gdzie $k$ oznacza indeks
klasy, a $j$ -- numer wzorca w obrębie tej klasy.

Dla danego wektora wejściowego $bold(x)$ neuron oblicza odległość euklidesową między
wejściem a zapamiętanym wzorcem, a następnie przetwarza ją przez funkcję jądra
#cite(<specht1990pnn>). Domyślnie stosuje się wielowymiarowe jądro Gaussa (zob. @kernel):

$ phi_(k j)(bold(x)) = exp lr(( -frac(||bold(x) - bold(x)_(k j)||^2, 2 sigma^2) )) $

gdzie:
- $||bold(x) - bold(x)_(k j)||$ -- odległość euklidesowa w przestrzeni cech (_#ref(<eq_distance>)_),
- $sigma$ -- parametr wygładzania (*smoothing parameter* lub *bandwidth*),
  decydujący o szerokości funkcji dzwonowej (zob. @sigma).

Odległość euklidesową między wektorem $x = [x_1, x_2, dots, x_p]^T$ a wzorcem $bold(x)_(k j)$ można wyrazić wzorem:

$ ||bold(x) - bold(x)_(k j)|| = sqrt(sum_(i=1)^p (x_i - (x)_(k j i))^2) $ <eq_distance>

=== Warstwa sumowania (Summation Layer)

Każdy neuron sumujący agreguje wyjścia neuronów wzorcowych należących do tej samej klasy
#cite(<bishop2006prml>):

$ S_k (bold(x)) = sum_(j=1)^(N_k) phi_(k j)(bold(x)) $

gdzie $N_k$ to liczba wzorców treningowych klasy $k$. Wynik $S_k$ jest proporcjonalny
do estymowanej gęstości prawdopodobieństwa klasy $k$ -- czynnik normalizujący jest
wspólny dla wszystkich klas i nie wpływa na wynik decyzji w warstwie wyjściowej.

=== Warstwa wyjściowa (Output Layer)

Warstwa wyjściowa implementuje optymalną regułę decyzyjną Bayesa #cite(<hastie2009elements>).
Zakładając równe prawdopodobieństwa *a priori* klas oraz jednakowe koszty błędnej
klasyfikacji, obserwacja $bold(x)$ zostaje przypisana do klasy $hat(k)$ o najwyższej
estymowanej gęstości prawdopodobieństwa:

$ hat(k) = op("arg max")_k , f_k (bold(x)) $

Architektura PNN zapewnia brak problemu minimów lokalnych (sieć nie jest trenowana
gradientowo) oraz asymptotyczną zbieżność do optymalnego klasyfikatora bayesowskiego
wraz ze wzrostem liczebności zbioru uczącego #cite(<specht1990pnn>).

== Funkcje jądra <kernel>

Funkcja jądra $K(bold(x), bold(x)_i)$ określa, jak silnie wzorzec $bold(x)_i$
wpływa na estymowaną gęstość w punkcie $bold(x)$ #cite(<silverman1986density>).
Wybór funkcji jądra wpływa na kształt granicy decyzyjnej i odporność modelu na szum.
Nie ma jednej uniwersalnej funkcji jądra, która byłaby najlepsza dla wszystkich problemów -- dobór powinien być dostosowany do charakterystyki danych #cite(<specht1990pnn>).

W każdej funkcji pomijamy stały czynnik normalizujący, ponieważ nie wpływa on na decyzję klasyfikacyjną. #cite(<PatternClassification>)

=== Gaussowska (Gaussian Kernel)

$ K(bold(x), bold(x)_i) = exp lr(( -frac(||bold(x) - bold(x)_i||^2, 2 sigma^2) )) $

Najczęściej stosowana funkcja jądra #cite(<specht1990pnn>). Gładka, różniczkowalna,
symetrycznie zanika wraz z odległością. Wrażliwa na wartości odstające ze względu
na szybki zanik gaussowski.

=== Laplasjańska (Laplacian Kernel)

$ K(bold(x), bold(x)_i) = exp lr(( -frac(||bold(x) - bold(x)_i||, sigma) )) $

Zanika wolniej niż jądro gaussowskie -- jest mniej czułe na wartości odstające
i lepiej sprawdza się przy danych z rozkładami o grubych ogonach.

=== Cauchy'ego (Cauchy Kernel)

$ K(bold(x), bold(x)_i) = frac(1, 1 + frac(||bold(x) - bold(x)_i||^2, sigma^2)) $

Posiada bardzo grube ogony, przez co silnie uwzględnia odległe wzorce.
Przydatne przy danych silnie zaszumionych lub heterogenicznych.

=== Odwrotna multikwadratowa (Inverse Multiquadric Kernel)

$ K(bold(x), bold(x)_i) = frac(1, sqrt(||bold(x) - bold(x)_i||^2 + sigma^2)) $

Funkcja zawsze dodatnia o łagodnym zaniku. Stosowana jako alternatywa dla jądra
gaussowskiego, gdy pożądana jest mniejsza czułość na dokładną wartość $sigma$.

=== Epanecznika (Epanechnikov Kernel)
$ K(bold(x), bold(x)_i) = max (lr(0, 1 - frac(||bold(x) - bold(x)_i||^2, sigma^2))) $
Jądro to jest gładkie wewnątrz obszaru wsparcia, ale na jego brzegu ma nieciągłość pochodnej. Z tego powodu dobrze sprawdza się w zadaniach, w których zależy nam na lokalnym uśrednianiu i ograniczeniu wpływu odległych obserwacji. W porównaniu z jądrem gaussowskim daje bardziej „lokalny” charakter estymacji i może lepiej tłumić słabo istotne, odległe wzorce.

Używamy funkcji $max$ do zapewnienia, że jądro nie zwróci wartości $< 0$. Ponieważ wartości ujemne nie mają sensu w kontekście estymacji gęstości, funkcja $max$ gwarantuje, że jądro będzie miało wartość zero dla obserwacji znajdujących się poza obszarem wsparcia (gdzie $||bold(x) - bold(x)_i||^2 > sigma^2$).

=== Trójkątna (Triangular Kernel)
$ K(bold(x), bold(x)_i) = max (lr(0, 1 - frac(||bold(x) - bold(x)_i||, sigma))) $
Jądro trójkątne jest funkcją liniową, która maleje wraz z odległością, osiągając wartość zero w odległości $sigma$. Jest to jądro o ograniczonym zasięgu, które uwzględnia tylko obserwacje znajdujące się w promieniu $sigma$ od punktu estymacji. Dzięki temu jest szczególnie skuteczne w zadaniach, gdzie ważne jest uwzględnienie tylko lokalnych informacji i ograniczenie wpływu odległych obserwacji.

Ponieważ jest ono bardzo podobne do jądra epanecznika, ale z liniowym zamiast kwadratowym spadkiem, może być bardziej odpowiednie w sytuacjach, gdzie chcemy jeszcze bardziej ograniczyć wpływ odległych obserwacji i skupić się na bardzo lokalnych strukturach danych. #cite(<wikipedia_kernels>)

== Parametr $sigma$ <sigma>

Parametr $sigma$ (parametr wygładzania, ang. *bandwidth*, *smoothing parameter*) jest kluczowym hiperparametrem
PNN -- kontroluje szerokość funkcji jądra i tym samym zasięg wpływu każdego wzorca
treningowego na estymowaną gęstość #cite(<specht1990pnn>):

- *zbyt małe $sigma$* -- sieć dopasowuje się ściśle do zbioru uczącego (overfitting),
  granica decyzyjna jest nieregularna i wrażliwa na szum,
- *zbyt duże $sigma$* -- sieć nadmiernie wygładza rozkład (underfitting),
  tracąc lokalne struktury w danych.

#figure(
  image("Images/kernels.png"),
  caption: [Porównanie kształtu różnych funkcji jądra dla kilku wartości $sigma$],
)


Do wyznaczenia optymalnej wartości $sigma$ stosuje się zazwyczaj jedną z poniższych metod:

- *$k$-krotna walidacja krzyżowa* (ang. $k$-fold cross-validation) -- wartość $sigma$
  dobierana jest tak, aby minimalizować błąd klasyfikacji na zbiorze walidacyjnym
  #cite(<hastie2009elements>),
- *reguła Silvermana* #cite(<silverman1986density>) -- heurystyczny estymator oparty
  na odchyleniu standardowym danych:
  $sigma^* = 1.06 , hat(s) , N^(-1\/5)$, gdzie $hat(s)$ to odchylenie standardowe
  cech, a $N$ to liczba wzorców,
- *przeszukiwanie siatki* (ang. grid search) -- systematyczne sprawdzenie zbioru
  kandydujących wartości $sigma$ z oceną na zbiorze testowym #cite(<hastie2009elements>).

W praktyce zaleca się stosowanie walidacji krzyżowej jako metody najbardziej odpornej
na specyfikę konkretnego zbioru danych.

== Różnice między PNN a innymi sieciami neuronowymi
PNN różni się od tradycyjnych sieci neuronowych, takich jak perceptron wielowarstwowy (MLP) czy sieci konwolucyjne (CNN), przede wszystkim sposobem reprezentacji i przetwarzania danych:
- *Brak procesu uczenia gradientowego* -- PNN nie wymaga iteracyjnego dostosowywania wag, co eliminuje problem minimów lokalnych i przyspiesza proces budowy modelu,
- *Bezpośrednia reprezentacja zbioru uczącego* -- każdy neuron wzorcowy przechowuje dokładnie jeden przykład treningowy, co pozwala na natychmiastowe wykorzystanie całego zbioru danych bez konieczności jego kompresji,
- *Estymacja gęstości zamiast funkcji decyzyjnej* -- PNN opiera się na estymacji gęstości prawdopodobieństwa klas, podczas gdy MLP i CNN uczą bezpośrednio funkcję decyzyjną, co wpływa na charakter granic decyzyjnych i odporność modelu na szum

PNN jest szczególnie efektywny w zadaniach klasyfikacji z niewielką ilością danych, gdzie tradycyjne sieci mogą mieć trudności z generalizacją. Jednakże, ze względu na bezpośrednią reprezentację zbioru uczącego, PNN może być niepraktyczny dla bardzo dużych zbiorów danych ze względu na wysokie wymagania pamięciowe i czasowe podczas predykcji.

= Analiza danych
TODO: ZROBIC
//TODO
= Skrypt programu
#figure(
  block(
    align(left, raw(read("../main.py"), lang: "python")),
    breakable: true,
    width: 100%,
  ),
  caption: [Główny program funkcji],
  kind: raw,
  placement: none,
)
#figure(
  block(
    align(left, raw(read("../src/pnn.py"), lang: "python")),
    breakable: true,
    width: 100%,
  ),
  caption: [Klasa implementująca probabilistyczną sieć neuronową],
  kind: raw,
  placement: none,
)

#figure(
  block(
    align(left, raw(read("../src/kernels.py"), lang: "python")),
    breakable: true,
    width: 100%,
  ),
  caption: [Klasa implementująca funkcje jądra używane w probabilistycznej sieci neuronowej z _#ref(<kernel>, supplement: [Sekcji])_],
  kind: raw,
  placement: none,
)

= Eksperymenty
W Probablistycznej sieci neuronowej kluczową rolę odgrywa parametr $sigma$ _#ref(<sigma>)_, który kontroluje szerokość funkcji jądra używanej do estymacji gęstości. Różne wartości $sigma$ mogą znacząco wpłynąć na dokładność klasyfikacji, dlatego ważne jest przeprowadzenie eksperymentów w celu znalezienia optymalnej wartości tego parametru.
Możemy również porównać różne funkcje jądra, takie jak gaussowska, laplasjańska, cauchy'ego, epańczykow, trójkątna i odwrotna multikwadratowa, aby zobaczyć, która z nich najlepiej sprawdza się w naszym zadaniu klasyfikacji.

Zrobimy to za pomocą następującego skryptu:
#figure(
  ```py
    # Best sigma and kernel search
    sigma = 0.001
    accuracies: dict[str, dict[float, np.floating]] = {} # {kernel_name: {sigma_value: accuracy_value}}
    kerns = [PnnKernels.gaussian_kernel, PnnKernels.laplacian_kernel, PnnKernels.cauchy_kernel, PnnKernels.inverse_multiquadric_kernel, PnnKernels.epanechnikov_kernel, PnnKernels.triangular_kernel]
    for kernel in kerns:
        p.kernel = kernel
        accuracies.setdefault(f"{kernel.__name__}", {})
        while sigma < 1:
            p.sigma = sigma
            predictions = p.predict(x_test[:100])
            accuracy = np.mean(predictions == y_test[:100])
            print(f"Test Accuracy: {accuracy:.2f}, Sigma: {sigma:.3f}, Kernel: {kernel.__name__}")
            accuracies[f"{kernel.__name__}"][sigma] = float(accuracy)
            sigma += 0.005
        sigma = 0.001
  ```,
  caption: [Część skryptu odpowiedzialna za znalezienie najlepszej konfiguracji jądra i $sigma$],
)
Badamy wartości $sigma$ w zakresie od $0.001$ do $1$ z krokiem $0.005$. Dla każdej funkcji jądra obliczamy dokładność klasyfikacji na zbiorze testowym i zapisujemy wyniki w słowniku `accuracies`, który mapuje nazwę funkcji jądra na słownik wartości $sigma$ i odpowiadających im dokładności.
Dla przyśpieszenia procesu testowania, ograniczamy zbiór testowy do pierwszych 100 próbek.

Po uruchomieniu możemy stwierdzić, że najlepszą dokładność ($0.92$) otrzymaliśmy dla funkcji jądra laplasjańskiej przy $sigma = 0.236$.
#figure(
  image("Images/pnn.png", width: 80%),
  caption: [Wykres dokładności dla różnych konfiguracji jądra i wartości sigma],
)
Jak widać na powyższym wykresie, najlepsze wyniki osiągnięto dla funkcji jądra laplasjańskiej. Jądro gaussowskie również osiągnęło dobrą dokładność, ale było bardziej wrażliwe na wybór $sigma$. Jądra cauchy'ego i odwrotna multikwadratowa wykazały niższą dokładność, co może sugerować, że są mniej odpowiednie dla tego konkretnego zbioru danych. Jądro epanecznika i trójkątne osiągneły praktycznie te same wyniki, co może być spowodowane ich podobnym charakterem (oba są jądrami o ograniczonym zasięgu). Ogólnie rzecz biorąc, wyniki te podkreślają znaczenie doboru odpowiedniej funkcji jądra i wartości $sigma$ dla osiągnięcia optymalnej wydajności PNN.

Warto również zauważyć, że dokładność klasyfikacji różni się w zależności od wyboru datasetów testowych (od $0.92$ do $0.97$), co podkreśla potrzebę stosowania walidacji krzyżowej lub innych metod oceny modelu, aby uzyskać bardziej wiarygodne oszacowanie jego wydajności.

= Podsumowanie i wnioski

W ramach projektu przedstawiono teoretyczne podstawy probabilistycznej sieci neuronowej (PNN), jej architekturę oraz rolę funkcji jądra i parametru $sigma$. Następnie przeprowadzono eksperyment porównujący kilka funkcji jądra dla różnych wartości $sigma$.

Uzyskane wyniki potwierdzają, że skuteczność PNN silnie zależy od doboru hiperparametrów. Najlepszą dokładność klasyfikacji osiągnięto dla jądra laplasjańskiego przy odpowiednio dobranym $sigma$, natomiast pozostałe jądra wykazywały większą wrażliwość lub niższą stabilność wyników.

Najważniejsze wnioski:
- dobór funkcji jądra ma istotny wpływ na jakość klasyfikacji,
- parametr $sigma$ powinien być strojon y eksperymentalnie (np. walidacją krzyżową),
- PNN jest prostą i skuteczną metodą dla zadań klasyfikacji, szczególnie przy mniejszych zbiorach danych,
- ograniczeniem PNN pozostaje koszt pamięciowy i obliczeniowy przy dużej liczbie wzorców.

#bibliography("bibliography.bib")
