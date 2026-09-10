# Mapa skryptow projektu

Ten plik jest szybka mapa po projekcie: ktory skrypt co robi, z czego korzysta, co zapisuje i jak laczy sie z innymi etapami.

Najwazniejsza mysl: `main.py` jest orkiestratorem. Wiekszosc funkcji nie zwraca duzych danych w pamieci, tylko zapisuje wyniki do folderu `data/`, a nastepny etap czyta je z dysku.

## Pipeline w skrocie

```text
data/raw/<case>/
    |
    |  data_loading/prepare_dicom.py
    v
data/for_reconstruction/<case>/*.h5
data/ground_truth/<case>/*.npz
    |
    |  denoising/runner.py
    v
data/denoised/<case>/<method>/*.h5
data/for_reconstruction/<case>_denoised_<method>/*.h5
    |
    |  segmentation/runner.py
    v
data/segmentations/<case>/<model>/*.npz
data/for_reconstruction/<case>_seg_<model>/*.h5
    |
    |  unsupervised_segmentation/runner.py
    |  data_loading/unsupervised_output_loader.py
    v
data/segmentations/<case>/<unsup_model>/*.npz
data/for_reconstruction/<case>_unsup_.../*.h5
    |
    |  reconstruction/runner.py
    v
data/reconstructed/<case>/<method>/*.npz
data/reconstructed/<case>/<method>/*.vti
    |
    |  isosurface_extraction/extract.py
    v
data/reconstructed/<case>/segmentations/*.obj
    |
    |  evaluation/*.py
    v
data/evaluation/
    |
    |  visualization/*.py
    v
podglad 2D/3D
```

## Pliki glowne

| Plik | Rola | Czy uruchamiac bezposrednio? |
| --- | --- | --- |
| `main.py` | Glowny pipeline. Czyta flagi z `config.py` i odpala kolejne etapy. | Tak, zwykle `python3 main.py`. |
| `config.py` | Panel sterowania: sciezki, case'y, flagi `RUN_*`, parametry metod. | Nie, edytujesz przed uruchomieniem. |
| `README.md` | Opis projektu od autorow. | Nie. |
| `requirements.txt` | Zaleznosci Pythona. | Nie, uzywany przy instalacji. |
| `SCRIPT_MAP.md` | Ten plik: mapa skryptow i relacji. | Nie. |

## `main.py`

`main.py` laczy wszystkie moduly.

Kolejnosc etapow:

| Etap | Flaga | Funkcja | Modul |
| --- | --- | --- | --- |
| 0. Data loading | `RUN_DATA_LOADING` | `prepare_cases()` | `data_loading.prepare_dicom` |
| 1. Denoising | `RUN_DENOISING` | `denoise_cases()` | `denoising.runner` |
| 2. Supervised segmentation | `RUN_SEGMENTATION` | `segment_cases()` | `segmentation.runner` |
| 3. Unsupervised external pipeline | `RUN_UNSUPERVISED_SEGMENTATION` | `run_unsupervised_segmentation()` | `unsupervised_segmentation.runner` |
| 4. Import unsupervised output | `RUN_UNSUPERVISED_OUTPUT_LOADING` | `import_unsupervised_outputs()` | `data_loading.unsupervised_output_loader` |
| 5. Reconstruction | `RUN_RECONSTRUCTION` | `reconstruct_cases()` | `reconstruction.runner` |
| 6. Isosurface | `RUN_ISOSURFACE_EXTRACTION` | `extract_surfaces_for_cases()` | `isosurface_extraction.extract` |
| 7. Evaluation | `RUN_EVALUATION` | `evaluate_*()` | `evaluation.*` |
| 8. Visualization | `RUN_NAPARI_VIS`, `RUN_SIMPLE_VIS` | `visualize_case_*()` | `visualization.*` |

Wazne listy budowane w `main.py`:

| Zmienna | Znaczenie |
| --- | --- |
| `base_case_names` | Case'y z `CASE_NAMES`, np. `kosartur`. |
| `intensity_case_names` | Case'y z intensywnoscia USG. Po denoisingu moga byc np. `kosartur_denoised_rdpad`. |
| `supervised_segmentation_cases` | Case'y wygenerowane przez model supervised segmentation. |
| `unsupervised_segmentation_cases` | Case'y zaimportowane z wynikow unsupervised segmentation. |
| `intensity_reconstruction_cases` | Case'y intensywnosci wybrane do rekonstrukcji. |
| `label_reconstruction_cases` | Case'y masek/etykiet wybrane do rekonstrukcji. |

## `data_loading/`

### `data_loading/prepare_dicom.py`

Cel: zamienia dane surowe DICOM na format uzywany przez projekt.

Wejscie:

```text
data/raw/<case>/*Sweeps*.dcm
data/raw/<case>/*Volume*.dcm
data/raw/<case>/*Segmentation*.dcm
```

Wyjscie:

```text
data/for_reconstruction/<case>/<case>_from_dicom.h5
data/for_reconstruction/<case>/<case>_images.npy
data/for_reconstruction/<case>/<case>_poses.npy
data/ground_truth/<case>/<case>_volume.npz
data/ground_truth/<case>/<case>_segmentation_labels.npz
data/ground_truth/<case>/seg_classes/<case>_classX.npz
```

Najwazniejsze funkcje:

| Funkcja | Co robi |
| --- | --- |
| `prepare_cases()` | Przechodzi po case'ach z `data/raw`. |
| `prepare_case()` | Przygotowuje jeden case. |
| `extract_poses_from_private_tag_0051_1039()` | Wyciaga macierze pozycji klatek z prywatnego taga DICOM. |
| `save_h5()` | Zapisuje obrazy 2D i pozy do H5. |
| `save_npz_volume()` | Zapisuje wolumen referencyjny lub segmentacje do NPZ. |

Relacje:

```text
prepare_dicom.py -> data/for_reconstruction -> reconstruction/runner.py
prepare_dicom.py -> data/ground_truth -> evaluation/visualization
```

### `data_loading/h5_utils.py`

Pomocnicze funkcje do H5.

Uzywane przez:

```text
denoising/runner.py
segmentation/runner.py
reconstruction/runner.py
unsupervised_segmentation/runner.py
data_loading/unsupervised_output_loader.py
```

Typowe zadania:

| Funkcja/rola | Co robi |
| --- | --- |
| listowanie H5 | Znajduje pliki `.h5` w case. |
| czytanie metadanych | Sprawdza dostepne datasety i atrybuty. |
| zgadywanie kluczy | Pomaga znalezc `img` i `poses`, gdy klucze nie sa podane w configu. |

### `data_loading/unsupervised_output_loader.py`

Cel: bierze wyniki z zewnetrznego unsupervised segmentora i przerabia je na format projektu.

Wejscie:

```text
external/UnsupervisedSegmentor4Ultrasound/.../segmaps lub crf_segmaps
data/for_reconstruction/<source_case>/*.h5
```

Wyjscie:

```text
data/segmentations/<source_case>/<model_name>/*.npz
data/for_reconstruction/<output_case>/*.h5
```

Relacje:

```text
unsupervised_segmentation/runner.py tworzy wyniki zewnetrzne
unsupervised_output_loader.py importuje te wyniki do lokalnego formatu
reconstruction/runner.py rekonstruuje zaimportowane maski
```

## `denoising/`

### `denoising/runner.py`

Cel: odszumia klatki USG w H5.

Wejscie:

```text
data/for_reconstruction/<case>/*.h5
```

Wyjscie:

```text
data/denoised/<case>/<method>/*_<method>.h5
data/for_reconstruction/<case>_denoised_<method>/*_<method>.h5
```

Najwazniejsze funkcje:

| Funkcja | Co robi |
| --- | --- |
| `denoise_cases()` | Odpala denoising dla listy case'ow. |
| `denoise_case()` | Odszumia wszystkie H5 w jednym case. |
| `denoise_file()` | Odszumia jeden H5 i zapisuje wynik. |
| `get_denoise_function()` | Mapuje nazwe metody, np. `rdpad`, na prawdziwa funkcje. |

Relacje:

```text
data/for_reconstruction/<case> -> denoising -> data/for_reconstruction/<case>_denoised_<method>
```

### `denoising/common.py`

Pomocnicze funkcje do denoisingu.

Typowe zadania:

| Rola | Co robi |
| --- | --- |
| load H5 | Laduje `img`, `poses`, spacing. |
| save H5 | Zapisuje odszumione `img`, stare `poses`, stare spacing. |
| `apply_slice_wise()` | Odpala filtr osobno dla kazdej klatki 2D. |
| copy to for reconstruction | Kopiuje gotowy H5 jako nowy case do dalszego pipeline'u. |

### `denoising/rdp_ad.py`

Implementacja metody `rdpad`.

Rola: bardziej specjalistyczne odszumianie obrazow USG. Runner wywoluje ja przez `denoise_rdpad_2d()`.

### `denoising/classical.py`

Klasyczne filtry obrazu:

| Metoda | Sens |
| --- | --- |
| `gaussian` | Wygladzanie filtrem Gaussa. |
| `median` | Usuwanie punktowego szumu przez mediane lokalna. |
| `bilateral` | Wygladzanie z proba zachowania krawedzi. |
| `tv` | Total variation denoising. |

## `segmentation/`

### `segmentation/runner.py`

Cel: supervised segmentacja klatek 2D modelem wczytanym z checkpointu
PyTorch Lightning.

Wejscie:

```text
data/prepared/<case>/sweep.h5
models/segmentation/*.ckpt
```

Wyjscie:

```text
data/segmented/<case>/<source>/<model>/labels.h5
```

Najwazniejsze funkcje:

| Funkcja | Co robi |
| --- | --- |
| `segment_cases()` | Odpala segmentacje dla listy case'ow. |
| `load_segmentation_model()` | Laduje architekture i wagi z checkpointu Lightning. |
| `predict_masks_for_sequence()` | Przewiduje maske osobno dla kazdej klatki 2D. |
| `segment_case()` | Zachowuje pozy i spacing oraz zapisuje `labels.h5`. |

Relacje:

```text
data/prepared/<case>/sweep.h5 -> segmentation -> data/segmented/<case>/<source>/<model>/labels.h5
```

### `segmentation/common.py`

Pomocnicze funkcje dla modelu segmentacji:

| Rola | Co robi |
| --- | --- |
| normalizacja | Skaluje intensywnosc tak samo jak podczas treningu. |
| resize obrazu | Dopasowuje obraz do `image_size` z checkpointu. |
| resize maski | Przywraca maske do oryginalnego rozmiaru klatki. |

### `segmentation/model.py` i `segmentation/checkpoint.py`

`model.py` zawiera architekture U-Net zgodna z projektem treningowym.
`checkpoint.py` odczytuje wagi oraz pelny config zapisany przez
`save_hyperparameters()`. Brak wymaganych metadanych konczy etap bledem.

## `unsupervised_segmentation/`

### `unsupervised_segmentation/runner.py`

Cel: integruje projekt z zewnetrznym repo `external/UnsupervisedSegmentor4Ultrasound`.

Wejscie:

```text
data/for_reconstruction/<case>/*.h5
external/UnsupervisedSegmentor4Ultrasound/
```

Wyjscie po stronie external repo:

```text
external/UnsupervisedSegmentor4Ultrasound/data/<dataset>/<case>/images/*.png
external/UnsupervisedSegmentor4Ultrasound/configs/*.yaml
external/UnsupervisedSegmentor4Ultrasound/deep-spectral-segmentation/.../segmaps
external/UnsupervisedSegmentor4Ultrasound/deep-spectral-segmentation/.../crf_segmaps
```

Najwazniejsze funkcje:

| Funkcja | Co robi |
| --- | --- |
| `run_unsupervised_segmentation()` | Glowny runner tego etapu. |
| `ensure_external_repo()` | Sprawdza, czy external repo istnieje. |
| `prepare_dataset_from_h5()` | Eksportuje klatki H5 do PNG dla external pipeline'u. |
| `generate_*_config()` | Tworzy konfiguracje Hydry/YAML dla external pipeline'u. |
| `run_command()` | Odpala komendy external pipeline'u. |

Relacje:

```text
for_reconstruction H5 -> PNG dataset w external repo -> external segmentation outputs
```

Ten etap sam z siebie nie tworzy jeszcze lokalnych case'ow w `data/for_reconstruction`. Do tego sluzy `data_loading/unsupervised_output_loader.py`.

## `reconstruction/`

### `reconstruction/runner.py`

Cel: wybiera metode rekonstrukcji i odpala ja dla case'ow.

Wejscie:

```text
data/for_reconstruction/<case>/*.h5
```

Wyjscie:

```text
data/reconstructed/<case>/<method>/*.npz
data/reconstructed/<case>/<method>/*.vti
```

Najwazniejsze funkcje:

| Funkcja | Co robi |
| --- | --- |
| `reconstruct_cases()` | Odpala rekonstrukcje dla listy case'ow. |
| `reconstruct_case()` | Rekonstruuje wszystkie H5 w jednym case. |
| `reconstruct_file_with_method()` | Wybiera konkretna metode i przekazuje parametry z configu. |
| `get_out_dir()` | Buduje katalog wyniku, zwykle z podfolderem metody. |

Metody:

| Nazwa w configu | Funkcja |
| --- | --- |
| `voxel_nearest` | `reconstruction.vnn.reconstruct_voxel_nearest_for_file()` |
| `voxel_nearest_mean` | `reconstruction.vnn_mean.reconstruct_voxel_nearest_mean_for_file()` |
| `distance_weighted` | `reconstruction.vnn_distance_weighted.reconstruct_distance_weighted_for_file()` |
| `bspline` | `reconstruction.bspline.reconstruct_bspline_for_file()` |

### `reconstruction/common.py`

Wspolny kod geometrii, ladowania, zapisu i samplingu.

Najwazniejsze role:

| Rola | Co robi |
| --- | --- |
| load image/poses | Czyta `img`, `poses`, spacing z H5. |
| volume geometry | Liczy rozmiar outputowego wolumenu, origin i voxel size. |
| world points | Tworzy punkty 3D dla srodkow voxeli. |
| sampling | Pobiera wartosci z klatek 2D nearest/bilinear. |
| validity | Odrzuca tlo lub wartosci spoza obrazu. |
| save | Zapisuje NPZ i VTI. |

### `reconstruction/vnn.py`

Voxel nearest neighbor.

Sens: dla kazdego voxela wybiera najblizsza klatke USG i bierze z niej wartosc piksela.

### `reconstruction/vnn_mean.py`

Voxel nearest neighbor mean.

Sens: dla kazdego voxela bierze kilka najblizszych klatek i liczy srednia z poprawnych probek.

### `reconstruction/vnn_distance_weighted.py`

Distance weighted reconstruction.

Sens: bierze kilka najblizszych klatek i liczy srednia wazona odlegloscia od plaszczyzny klatki.

### `reconstruction/bspline.py`

B-spline reconstruction.

Sens: podobny do distance weighted, ale wartosci z klatek 2D pobiera interpolacja B-spline zamiast prostego nearest/bilinear.

## `isosurface_extraction/`

### `isosurface_extraction/extract.py`

Cel: zamienia wolumen maski/etykiety na powierzchnie 3D.

Wejscie:

```text
data/reconstructed/<case>/<ISO_SOURCE_SUBDIR>/*.npz
```

albo zalezne od configu:

```text
ISO_SOURCE_SUBDIR
ISO_PATTERN
ISO_LEVEL
```

Wyjscie:

```text
data/reconstructed/<case>/surfaces/*.npz
data/reconstructed/<case>/surfaces/*.obj
```

Relacje:

```text
reconstruction label/mask -> isosurface -> visualization
```

Najbardziej sensowne dla masek/segmentacji, bo dla intensywnosci USG wybor progu powierzchni jest mniej oczywisty.

## `evaluation/`

### `evaluation/reconstruction_with_gt.py`

Cel: porownuje zrekonstruowana intensywnosc z ground truth volume.

Wejscie:

```text
data/reconstructed/<case>/<method>/*.npz
data/ground_truth/<case>/*_volume.npz
```

Wyjscie:

```text
data/evaluation/
```

### `evaluation/segmentation_metrics.py`

Cel: liczy metryki segmentacji, gdy mamy rekonstrukcje maski i referencyjne segmentacje.

Wejscie:

```text
data/reconstructed/<case>/<method>/*.npz
data/ground_truth/<case>/seg_classes/*.npz
```

Wyjscie:

```text
data/evaluation/
```

### `evaluation/no_reference.py`

Cel: metryki bez referencji, czyli ocena samego zrekonstruowanego wolumenu/maski.

Wejscie:

```text
data/reconstructed/<case>/<method>/*.npz
```

Wyjscie:

```text
data/evaluation/
```

## `visualization/`

### `visualization/napari_viewer.py`

Cel: podglad 3D w Napari.

Czyta:

```text
data/reconstructed/<case>/
data/ground_truth/<case>/
```

Pokazuje m.in.:

| Warstwa | Zrodlo |
| --- | --- |
| reconstruction | `data/reconstructed/<case>/<method>/*.npz` |
| ground truth volume | `data/ground_truth/<case>/*_volume.npz` |
| segmentation labels | `data/ground_truth/<case>/*_segmentation_labels.npz` |
| segmentation classes | `data/ground_truth/<case>/seg_classes/*.npz` |
| surfaces | powierzchnie z isosurface |

### `visualization/simple_viewer.py`

Cel: prostszy podglad bez pelnego Napari.

Czyta podobne dane jak `napari_viewer.py`, ale sluzy raczej do szybkiego sprawdzenia wyniku.

### `visualization/common.py`

Pomocnicze funkcje ladowania i przygotowania danych do wizualizacji.

## `utils/`

Katalog pomocniczy. Obecnie zawiera glownie plik inicjalizacyjny `__init__.py`, czyli nie jest centralnym miejscem logiki.

## `external/`

### `external/UnsupervisedSegmentor4Ultrasound/`

Zewnetrzny projekt do unsupervised segmentation.

Ten katalog ma wlasna strukture, configi, zaleznosci i pipeline. Lokalny projekt nie powinien mieszac jego logiki z glownym kodem. Integracja odbywa sie przez:

```text
unsupervised_segmentation/runner.py
data_loading/unsupervised_output_loader.py
```

## Najwazniejsze relacje miedzy modulami

```text
main.py
  -> config.py
  -> data_loading.prepare_dicom
  -> denoising.runner
  -> segmentation.runner
  -> unsupervised_segmentation.runner
  -> data_loading.unsupervised_output_loader
  -> reconstruction.runner
  -> isosurface_extraction.extract
  -> evaluation.*
  -> visualization.*
```

```text
data_loading.prepare_dicom
  -> data/for_reconstruction
  -> data/ground_truth
```

```text
denoising.runner
  -> czyta data/for_reconstruction/<case>
  -> zapisuje data/denoised/<case>/<method>
  -> opcjonalnie zapisuje data/for_reconstruction/<case>_denoised_<method>
```

```text
segmentation.runner
  -> czyta data/for_reconstruction/<case>
  -> czyta models/segmentation/<model>
  -> zapisuje data/segmentations/<case>/<model>
  -> opcjonalnie zapisuje data/for_reconstruction/<case>_seg_<model>
```

```text
unsupervised_segmentation.runner
  -> czyta data/for_reconstruction/<case>
  -> zapisuje PNG/config/output w external repo
```

```text
data_loading.unsupervised_output_loader
  -> czyta output z external repo
  -> czyta oryginalne H5 z data/for_reconstruction
  -> zapisuje data/segmentations
  -> zapisuje data/for_reconstruction/<case>_unsup_...
```

```text
reconstruction.runner
  -> czyta data/for_reconstruction/<case>
  -> zapisuje data/reconstructed/<case>/<method>
```

```text
isosurface_extraction.extract
  -> czyta data/reconstructed/<case>
  -> zapisuje powierzchnie 3D
```

```text
evaluation.*
  -> czyta data/reconstructed i opcjonalnie data/ground_truth
  -> zapisuje data/evaluation
```

```text
visualization.*
  -> czyta data/reconstructed i data/ground_truth
  -> pokazuje wynik
```

## Jak czytac outputy

| Folder | Znaczenie |
| --- | --- |
| `data/raw/` | Surowe dane wejsciowe, najczesciej DICOM. |
| `data/for_reconstruction/` | Dane gotowe do rekonstrukcji. Kazdy podfolder to osobny case. |
| `data/denoised/` | Wyniki denoisingu jako osobny etap. |
| `data/segmentations/` | Wyniki segmentacji 2D przed rekonstrukcja 3D. |
| `data/reconstructed/` | Wolumeny 3D po rekonstrukcji. |
| `data/ground_truth/` | Dane referencyjne do porownan i wizualizacji. |
| `data/evaluation/` | Metryki i raporty z ewaluacji. |

## Najczestsza sciezka danych

Normalny flow intensywnosci:

```text
raw DICOM
  -> prepare_dicom.py
  -> for_reconstruction/<case>/*.h5
  -> reconstruction/runner.py
  -> reconstructed/<case>/<method>/*.npz
```

Flow z denoisingiem:

```text
for_reconstruction/<case>/*.h5
  -> denoising/runner.py
  -> for_reconstruction/<case>_denoised_<method>/*.h5
  -> kolejne etapy
```

Flow supervised segmentation:

```text
for_reconstruction/<case>/*.h5
  -> segmentation/runner.py
  -> segmentations/<case>/<model>/*.npz
  -> for_reconstruction/<case>_seg_<model>/*.h5
  -> reconstruction/runner.py
  -> reconstructed/<case>_seg_<model>/<method>/*.npz
```

Flow unsupervised segmentation:

```text
for_reconstruction/<case>/*.h5
  -> unsupervised_segmentation/runner.py
  -> external repo outputs
  -> unsupervised_output_loader.py
  -> for_reconstruction/<case>_unsup_.../*.h5
  -> reconstruction/runner.py
  -> reconstructed/<case>_unsup_.../<method>/*.npz
```
