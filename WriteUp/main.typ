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

#outline(depth: 2)
#pagebreak()

= Opis problemu
Zadaniem projektu jest implementacja i analiza probabilistycznej sieci neuronowej (PNN) #cite(<specht1990pnn>) w kontekście klasyfikacji wiadomości e-mail jako spam lub nie-spam. Na podstawie zbioru danych spamebase #cite(<spambase>). Zostaną również przeprowadzone eksperymenty badające wpływ różnych funkcji jądra i wartości parametru wygładzania $sigma$ na dokładność klasyfikacji.

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
  caption: [Model probabilistycznej sieci neuronowej (PNN) \ Dla przejrzystości pokazano tylko 2 wzorce treningowe dla każdej z 2 klas oraz 2 cechy wejściowe],
) <Diagram_PNN>

Poszczególne symbole użyte na #ref(<Diagram_PNN>, supplement: [Rysunku]) oznaczają:

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
warstwie jest równa wymiarowości $p$ przestrzeni wejściowej $x$.

=== Warstwa wzorców (Pattern Layer) <patternlayer>

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
gdzie:
- $p$ -- liczba cech (wymiarowość przestrzeni wejściowej),
- $x_i$ -- $i$-ta cecha wektora wejściowego,
- $(x)_(k j i)$ -- $i$-ta cecha wzorca $bold(x)_(k j)$.
=== Warstwa sumowania (Summation Layer)

Każdy neuron sumujący agreguje wyjścia neuronów wzorcowych należących do tej samej klasy
#cite(<bishop2006prml>):

$ S_k (bold(x)) = sum_(j=1)^(N_k) phi_(k j)(bold(x)) $

gdzie $N_k$ to liczba wzorców treningowych klasy $k$. Wynik $S_k$ jest proporcjonalny
do estymowanej gęstości prawdopodobieństwa klasy $k$, $phi$ to funkcja jądra (#ref(<kernel>)), a $bold(x)$ to wektor cech obserwacji.

=== Warstwa wyjściowa (Output Layer)

Warstwa wyjściowa implementuje optymalną regułę decyzyjną Bayesa #cite(<hastie2009elements>).
Zakładając równe prawdopodobieństwa *a priori* #footnote([Każda klasa na początku ma takie samo prawdopodobieństwo]) klas oraz jednakowe koszty błędnej
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

W każdej funkcji pomijamy stały czynnik normalizujący, *ponieważ jest on identyczny dla wszystkich wzorców*, więc nie wpływa on na decyzję klasyfikacyjną. #cite(<PatternClassification>)

=== Gaussowska (Gaussian Kernel)

$ K(bold(x), bold(x)_i) = exp lr(( -frac(||bold(x) - bold(x)_i||^2, 2 sigma^2) )) $

Najczęściej stosowana funkcja jądra #cite(<specht1990pnn>) #cite(<Kusy2014ProbabilisticNN>). Gładka, różniczkowalna,
symetrycznie zanika wraz z odległością. Wrażliwa na wartości odstające ze względu
na szybki zanik gaussowski.

=== Laplasjańska (Laplacian Kernel)

$ K(bold(x), bold(x)_i) = exp lr(( -frac(||bold(x) - bold(x)_i||, sigma) )) $

Zanika wolniej niż jądro gaussowskie -- jest mniej czułe na wartości odstające
i lepiej sprawdza się przy danych z rozkładami o grubych ogonach.

=== Cauchy'ego (Cauchy Kernel) #cite(<cursedKernels>)

$ K(bold(x), bold(x)_i) = frac(1, 1 + frac(||bold(x) - bold(x)_i||^2, sigma^2)) $

Posiada bardzo grube ogony, przez co silnie uwzględnia odległe wzorce.
Przydatne przy danych silnie zaszumionych lub heterogenicznych.

=== Odwrotna multikwadratowa (Inverse Multiquadric Kernel) #cite(<commonKernels>)

$ K(bold(x), bold(x)_i) = frac(1, sqrt(||bold(x) - bold(x)_i||^2 + sigma^2)) $

Funkcja zawsze dodatnia o łagodnym zaniku. Stosowana jako alternatywa dla jądra
gaussowskiego, gdy pożądana jest mniejsza czułość na dokładną wartość $sigma$.

=== Epanecznika (Epanechnikov Kernel) #cite(<wikipedia_kernels>)
$ K(bold(x), bold(x)_i) = max (lr(0, 1 - frac(||bold(x) - bold(x)_i||^2, sigma^2))) $
Jądro to jest gładkie wewnątrz obszaru wsparcia, ale na jego brzegu ma nieciągłość pochodnej. Z tego powodu dobrze sprawdza się w zadaniach, w których zależy nam na lokalnym uśrednianiu i ograniczeniu wpływu odległych obserwacji. W porównaniu z jądrem gaussowskim daje bardziej „lokalny” charakter estymacji i może lepiej tłumić słabo istotne, odległe wzorce.

Używamy funkcji $max$ do zapewnienia, że jądro nie zwróci wartości $< 0$. Ponieważ wartości ujemne nie mają sensu w kontekście estymacji gęstości, funkcja $max$ gwarantuje, że jądro będzie miało wartość zero dla obserwacji znajdujących się poza obszarem wsparcia (gdzie $||bold(x) - bold(x)_i||^2 > sigma^2$).

=== Trójkątna (Triangular Kernel)
$ K(bold(x), bold(x)_i) = max (lr(0, 1 - frac(||bold(x) - bold(x)_i||, sigma))) $
Jądro trójkątne jest funkcją liniową, która maleje wraz z odległością, osiągając wartość zero w odległości $sigma$. Jest to jądro o ograniczonym zasięgu, które uwzględnia tylko obserwacje znajdujące się w promieniu $sigma$ od punktu estymacji. Dzięki temu jest szczególnie skuteczne w zadaniach, gdzie ważne jest uwzględnienie tylko lokalnych informacji i ograniczenie wpływu odległych obserwacji.

=== Jednostkowa (Uniform Kernel)
$ K(x, x_i) = cases(0.5 "kiedy" ||x - x_i|| <= sigma, 0 "w przeciwnym wypadku") $
Jądro jednostkowe jest funkcją o stałej wartości wewnątrz obszaru wsparcia i zerowej wartości poza nim. Jest to najprostsze jądro, które uwzględnia tylko obserwacje znajdujące się w promieniu $sigma$ od punktu estymacji, przypisując im jednakową wagę. Ze względu na swoją prostotę, jądro jednostkowe może być mniej skuteczne w zadaniach, gdzie ważne jest uwzględnienie różnic w odległości między obserwacjami a punktem estymacji.

Porównanie kształtu różnych funkcji jądra dla kilku wartości $sigma$ przedstawiono na #ref(<PorownanieJader>).

== Parametr wygładzania $sigma$ <sigma>

Parametr $sigma$ (parametr wygładzania, ang. *bandwidth*, *smoothing parameter*) jest kluczowym hiperparametrem
PNN -- kontroluje szerokość funkcji jądra i tym samym zasięg wpływu każdego wzorca
treningowego na estymowaną gęstość #cite(<specht1990pnn>):

- *zbyt małe $sigma$* -- sieć dopasowuje się ściśle do zbioru uczącego (overfitting),
  granica decyzyjna jest nieregularna i wrażliwa na szum,
- *zbyt duże $sigma$* -- sieć nadmiernie wygładza rozkład (underfitting),
  tracąc lokalne struktury w danych.

#figure(
  image("Images/porownaniejader2.png"),
  caption: [Porównanie kształtu różnych funkcji jądra dla kilku wartości $sigma$],
) <PorownanieJader>


=== Dobór optymalnej wartości $sigma$ <sigmasearch>
Do wyznaczenia optymalnej wartości $sigma$ stosuje się zazwyczaj jedną z poniższych metod:

- *$k$-krotna walidacja krzyżowa* (ang. $k$-fold cross-validation) -- wartość $sigma$
  dobierana jest tak, aby minimalizować błąd klasyfikacji na zbiorze walidacyjnym
  #cite(<hastie2009elements>),
- *reguła Silvermana* #cite(<silverman1986density>) -- heurystyczny estymator oparty
  na odchyleniu standardowym danych:
  $sigma^* = 1.06 * hat(s) * N^(-1\/5)$, gdzie $hat(s)$ <silvermanrule>
  to odchylenie standardowe
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
Zbiór danych spambase #cite(<spambase>) zawiera 4601 próbek wiadomości e-mail, z których 57% to spam. Każda próbka jest reprezentowana przez 57 cech numerycznych, które opisują różne aspekty wiadomości, takie jak częstotliwość występowania określonych słów i znaków specjalnych, długość wiadomości oraz inne statystyczne właściwości tekstu. Celem jest klasyfikacja wiadomości jako spam lub nie-spam na podstawie tych cech.

Zbiór nie zawiera brakujących wartości, a cechy są numeryczne, co ułatwia ich bezpośrednie wykorzystanie w modelu PNN.

Dane przed przetwarzaniem są normalizowane, ponieważ cechy zbioru danych mają różne zakresy wartości (częstość występowania słów to zakres od $0$ do $1$, natomist  długość wiadomości to zakres od $0$ do $10000$).
Jeżeli nie przeprowadzimy normalizacji, odległości euklidesowe obliczane przez PNN będą zdominowane przez cechy o większych zakresach, co może prowadzić do błędnych klasyfikacji. Normalizacja zapewnia, że wszystkie cechy mają równy wpływ na estymację gęstości i poprawia wydajność modelu.

#figure(
  ```py
  def normalize_data(X: np.ndarray) -> np.ndarray:
    """
    Normalize the features in the dataset to have zero mean and unit variance.
    :param X: The input feature matrix to be normalized.
    :return: The normalized feature matrix.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std
  ```,
  caption: [Część skryptu odpowiedzialna za normalizację danych],
)
Po znormalizowaniu danych, każda cecha ma wartość średnią równą $0$ i odchylenie standardowe równe $1$. Dzięki temu PNN może efektywnie obliczać odległości euklidesowe i estymować gęstość prawdopodobieństwa bez dominacji jednej cechy nad innymi.

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
W Probablistycznej sieci neuronowej kluczową rolę odgrywa parametr $sigma$ (_#ref(<sigma>)_), który kontroluje szerokość funkcji jądra używanej do estymacji gęstości. Różne wartości $sigma$ mogą znacząco wpłynąć na dokładność klasyfikacji, dlatego ważne jest przeprowadzenie eksperymentów w celu znalezienia optymalnej wartości tego parametru.
Porównamy również różne funkcje jądra, aby zobaczyć, jak wpływają one na wydajność modelu. (_#ref(<kernel>)_).

#figure(
  ```py
      def kFold_search(
      x: np.ndarray,
      y: np.ndarray,
      min_sigma=0.001,
      max_sigma=2,
      diff_sigma=0.005,
      splits=5,
      kerns=[
          PnnKernels.gaussian_kernel
      ],
  ) -> dict[str, dict[float, np.ndarray]]:
      """
      Perform k-fold cross-validation to search for the best sigma and kernel configuration.
      :param x: The input feature matrix.
      :param y: The target labels.
      :param min_sigma: The minimum value for sigma.
      :param max_sigma: The maximum value for sigma.
      :param diff_sigma: The step size for sigma.
      :return: A dictionary containing the accuracies for each kernel and sigma combination.
      """
      accuracies: dict[
          str, dict[float, np.ndarray]
      ] = {}  # {kernel_name: {sigma_value: (min_acc, max_acc, avg_acc)}}

      kf = KFold(n_splits=splits, shuffle=True, random_state=67)
      p = PNN()
      for kernel in kerns:
          kernel_name = kernel.__name__
          accuracies[kernel_name] = {}
          for sigma in np.arange(min_sigma, max_sigma + diff_sigma, diff_sigma):
              p.kernel = kernel
              p.sigma = sigma
              fold_accuracies = []
              for train_index, test_index in kf.split(x):
                  x_train, x_test = x[train_index], x[test_index]
                  y_train, y_test = y[train_index], y[test_index]
                  p.fit(x_train, y_train)
                  y_pred = np.array([p.predict_single(x) for x in x_test])
                  fold_accuracy = np.mean(y_pred == y_test)
                  fold_accuracies.append(fold_accuracy)
              accuracies[kernel_name][sigma] = np.array(fold_accuracies)
              print(
                  f"Kernel: {kernel_name}, Sigma: {sigma:.3f}, Accuracy: {np.mean(accuracies[kernel_name][sigma]):.4f} (min: {np.min(accuracies[kernel_name][sigma]):.4f}, max: {np.max(accuracies[kernel_name][sigma]):.4f})"
              )
      return accuracies
  ```,
  caption: [Część skryptu odpowiedzialna za poszukiwanie najlepszej konfiguracji jądra i $sigma$ z użyciem walidacji krzyżowej (_#ref(<sigmasearch>)_)],
)
Korzystamy z z $5$-krotnej walidacji krzyżowej, aby uzyskać bardziej wiarygodne oszacowanie dokładności dla każdej kombinacji jądra i $sigma$.

Badamy wartości $sigma$ w zakresie od $0.001$ do $2$ z krokiem $0.005$, co pozwala nam zobaczyć, jak dokładność zmienia się w szerokim zakresie wartości tego parametru.

Po przeprowadzeniu eksperymentów, możemy zwizualizować wyniki na wykresie, który pokazuje dokładność dla różnych konfiguracji jądra i wartości $sigma$.
#figure(
  image("Images/pnn2.png", width: 95%),
  caption: [Wykres dokładności dla różnych konfiguracji jądra i wartości sigma],
)
Najlepszą dokładność klasyfikacji osiągnięto dla jądra laplasjańskiego przy $sigma = 0.611$, (średnia dokładność $0.9244$ , min: $0.9121$, max: $0.9359$). Ze względu na podobną charakterystykę jądra gaussowskiego, osiągneło ono podobne wyniki. Jądra trójkątne, epańczykowa i jednostkowe wykazały prawie taką samą dokładność na całym zakresie $sigma$, co może być spowodowane ich podobną charakterystyką (_#ref(<PorownanieJader>)_).
Jądra cauchy'ego i odwrotna multikwadratowa nie są odpowiednie dla tego zbioru danych, ich dokładność nie przekroczyła $0.8$ dla żadnej wartości $sigma$.

Wykorzystaliśmy również regułę Silvermana (_#ref(<sigmasearch>)_) do oszacowania optymalnej wartości $sigma$ na podstawie odchylenia standardowego cech. Dla naszego zbioru danych, reguła Silvermana zasugerowała wartość $sigma^* = 0.1962$, sprawdzając tą wartość z każdym z jąder uzyskaliśmy największą średnią dokładność dla jądra laplasjańskiego ($0.9141$), co jest zgodne z wynikami uzyskanymi podczas eksperymentu z walidacją krzyżową.

= Podsumowanie i wnioski

W ramach projektu przedstawiono teoretyczne podstawy probabilistycznej sieci neuronowej (PNN), jej architekturę oraz rolę funkcji jądra i parametru $sigma$. Następnie przeprowadzono eksperyment porównujący kilka funkcji jądra dla różnych wartości $sigma$.

Uzyskane wyniki potwierdzają, że skuteczność PNN silnie zależy od doboru hiperparametrów. Najlepszą dokładność klasyfikacji osiągnięto dla jądra laplasjańskiego przy odpowiednio dobranym $sigma$, natomiast pozostałe jądra wykazywały większą wrażliwość lub niższą stabilność wyników.

Reguła Silvermana okazała się użytecznym narzędziem do oszacowania początkowej wartości $sigma$, która następnie mogła być dostrojona za pomocą walidacji krzyżowej, co potwierdziło jej praktyczną wartość w kontekście PNN.

Probabilistyczne sieci neuronowe są skuteczną metodą klasyfikacji, podczas naszych eksperymentów osiągnęły wysoką dokładność porównywalną z innymi metodami klasyfikacji #cite(<spambase>). Mimo bardzo krótkiego czasu uczenia, PNN może być konkurencyjną alternatywą dla bardziej złożonych modeli, zwłaszcza w zadaniach z niewielką ilością danych.

Najważniejsze wnioski:
- dobór funkcji jądra ma istotny wpływ na jakość klasyfikacji,
- parametr $sigma$ powinien być strojony eksperymentalnie (np. walidacją krzyżową),
- PNN jest prostą i skuteczną metodą dla zadań klasyfikacji, szczególnie przy mniejszych zbiorach danych,
- ograniczeniem PNN pozostaje koszt pamięciowy i obliczeniowy przy dużej liczbie wzorców.

#bibliography("bibliography.bib")
