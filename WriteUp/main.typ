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

#set heading(numbering: "1.1.a.")
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
Probabilistyczna sieć neuronowa (PNN) #cite(<specht1990pnn>) -- sieć nueronowa, w której wyjście każdego neuronu jest interpretowane jako prawdopodobieństwo przynależności do danej klasy. PNN jest oparta na teorii estymacji gęstości jądrowej i jest często stosowana w zadaniach klasyfikacji.

== Model probablistycznej sieci neuronowej

#figure(
  diagram(
    node-stroke: 1pt,
    edge-stroke: 1pt,
    node-fill: white,
    spacing: (1.5cm, 1cm), // Controls (horizontal, vertical) spacing between nodes

    // ---------------- Layer Labels ----------------
    node((0, -1), [*Input Layer* #linebreak() *Wejścia sieci*], stroke: none),
    node((1, -1), [*Pattern Layer* #linebreak() *Neurony wzorcowe*], stroke: none),
    node((2, -1), [*Summation Layer* #linebreak() *Neurony sumujące*], stroke: none),
    node((3, -1), [*Output Layer* #linebreak() *Neuron decyzyjny*], stroke: none),

    // ---------------- Input Neurons ----------------
    node((0, 0.5), [$x_1$], shape: "circle", name: <i1>, width: 2.5em),
    node((0, 2.5), [$x_2$], shape: "circle", name: <i2>, width: 2.5em),

    // ---------------- Pattern Neurons ----------------
    // Class A Patterns
    node((1, 0), [$P_("A1")$], shape: "circle", name: <pa1>, width: 2.5em),
    node((1, 1), [$P_("A2")$], shape: "circle", name: <pa2>, width: 2.5em),

    // Class B Patterns
    node((1, 2), [$P_("B1")$], shape: "circle", name: <pb1>, width: 2.5em),
    node((1, 3), [$P_("B2")$], shape: "circle", name: <pb2>, width: 2.5em),

    // ---------------- Summation Neurons ----------------
    node((2, 0.5), [$Sigma_A$], shape: "circle", name: <sa>, width: 2.5em),
    node((2, 2.5), [$Sigma_B$], shape: "circle", name: <sb>, width: 2.5em),

    // ---------------- Output Neuron ----------------
    node((3, 1.5), [Decision\ (Argmax)], shape: "circle", name: <out>, width: 5em),

    // ---------------- Edges ----------------

    // 1. Input to Pattern (Fully connected, standard PNN behavior)
    edge(<i1>, <pa1>, "->", stroke: gray),
    edge(<i1>, <pa2>, "->", stroke: gray),
    edge(<i1>, <pb1>, "->", stroke: gray),
    edge(<i1>, <pb2>, "->", stroke: gray),

    edge(<i2>, <pa1>, "->", stroke: gray),
    edge(<i2>, <pa2>, "->", stroke: gray),
    edge(<i2>, <pb1>, "->", stroke: gray),
    edge(<i2>, <pb2>, "->", stroke: gray),

    // 2. Pattern to Summation (Class specific: Patterns only connect to their respective class sum)
    edge(<pa1>, <sa>, "->"),
    edge(<pa2>, <sa>, "->"),

    edge(<pb1>, <sb>, "->"),
    edge(<pb2>, <sb>, "->"),

    // 3. Summation to Output
    edge(<sa>, <out>, "->"),
    edge(<sb>, <out>, "->"),
  ),
  caption: "Model Probablistycznej sieci neuronowej",
) <Diagram_PNN>
Gdzie:
- $x_1$, $x_2$ -- wejścia sieci (cechy danych)
- $P_("A1")$, $P_("A2")$ -- neurony wzorcowe dla klasy A
- $P_("B1")$, $P_("B2")$ -- neurony wzorcowe dla klasy B
- $Sigma_A$, $Sigma_B$ -- neurony sumujące dla klas A i B
- Output (Argmax) -- neuron decyzyjny, który wybiera klasę z największym prawdopodobieństwem.
== Funkcje jądra <kernel>

== Parametr sigma <sigma>

= Analiza danych

//TODO
= Skrypt programu
#figure(
  block(
    align(left, raw(read("../main.py"), lang: "python")),
    breakable: true,
    width: 100%,
  ),
  caption: "Główny program funkcji",
  kind: raw,
  placement: none,
)
#figure(
  block(
    align(left, raw(read("../src/pnn.py"), lang: "python")),
    breakable: true,
    width: 100%,
  ),
  caption: "Klasa implementująca probabilistyczną sieć neuronową",
  kind: raw,
  placement: none,
)

#figure(
  block(
    align(left, raw(read("../src/kernels.py"), lang: "python")),
    breakable: true,
    width: 100%,
  ),
  caption: "Klasa implementująca funkcje jądra używane w probabilistycznej sieci neuronowej",
  kind: raw,
  placement: none,
)

= Eksperymenty
W Probablistycznej sieci neuronowej kluczową rolę odgrywa parametr $sigma$ _#ref(<sigma>)_, który kontroluje szerokość funkcji jądra używanej do estymacji gęstości. Różne wartości $sigma$ mogą znacząco wpłynąć na dokładność klasyfikacji, dlatego ważne jest przeprowadzenie eksperymentów w celu znalezienia optymalnej wartości tego parametru.
Możemy również porównać różne funkcje jądra, takie jak gaussowska, laplasjańska, cauchy'ego i odwrotna multikwadratowa, aby zobaczyć, która z nich najlepiej sprawdza się w naszym zadaniu klasyfikacji.

Zrobimy to za pomocą następującego skryptu:
#figure(
  ```py
    sigma = 0.001
    accuracies: dict[str, dict[float, np.floating]] = {} # {kernel_name: {sigma_value: accuracy_value}}
    kerns = [PnnKernels.gaussian_kernel, PnnKernels.laplacian_kernel, PnnKernels.cauchy_kernel, PnnKernels.inverse_multiquadric_kernel]
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
  caption: [Część skryptu odpowiedzialna za znlezienie najlepszej konfiguracji jądra i $sigma$],
)
Po uruchomieniu możemy stwierdzić, że najlepszą dokładność ($0.92$) otrzymaliśmy dla funkcji jądra laplasjańskiej przy $sigma = 0.236$.
#figure(
  image("Images/pnn.png", width: 80%),
  caption: [Wykres dokładności dla różnych konfiguracji jądra i wartości sigma],
)

= Podsumowanie i wnioski

#bibliography("bibliography.bib")
