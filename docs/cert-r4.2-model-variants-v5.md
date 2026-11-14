# CERT: пункт 4 — endpoint score LSTM и табличный boosting

Протокол зафиксирован 14 ноября 2026 года до запуска новой CERT-серии.
[Bounded plan](../configs/cert_r4_2_model_variants_v5.json) связывает daily source,
[пункт 2](cert-r4.2-group-cv-v3.md), [пункт 3](cert-r4.2-file-usb-v4.md) и все
конфигурации через SHA-256. Пункт 5 с иной рабочей точкой здесь не выполняется.

## Почему именно эти проверки

Пункты 1–3 исправили доступность истории и отдельно проверили source-specific
deviations. RF получил измеримый development-прирост; LSTM-AE остался слабым
при global-only, causal и обоих file/USB profiles. Это основание проверить
недостаток score и другой model family, а не наращивать LSTM вслепую.

1. **LSTM endpoint score.** Target — observable membership последнего дня окна,
   прежний score — MSE по всем пяти дням. Ошибка в предыдущих background днях
   может давать FP, а ошибка последнего дня — разбавляться другими днями.
   Отдельный `endpoint_mean_mse_v2` усредняет squared error только по признакам
   последнего дня. Это проверяемая гипотеза; error последнего дня не обязан
   хорошо разделять official labels.
2. **HistGradientBoostingClassifier (HGB).** Дневные агрегаты и прошлые robust
   deviations образуют табличную задачу, RF уже извлекает из них полезный сигнал.
   Проверяется один дополнительный supervised tree ensemble с shallow trees.
   Сравнение HGB с RF меняет model family при одинаковом доступе к размеченному
   inner fit. Это проверка альтернативы, а не автоматическая замена LSTM.

LSTM-AE остаётся в Argus как воспроизводимый reconstruction baseline. Его плохой
результат не доказывает непригодность всех LSTM/последовательных моделей.
Классификатор с sequence loss по меткам, другой autoencoder или Transformer
были бы иными гипотезами с собственным бюджетом; в этой серии они отсутствуют.

## Два фактора не смешиваются

Сохраняются четыре ранее проверенных feature policies: global-only v2, causal
v3, rolling file/USB v4 и frozen file/USB v4. Ни один вариант не выбирается
по новой outer-метрике для дополнительного обучения. Все 12 preparations уже
fit на своих inner fit и переиспользуются неизменными; новых feature fit нет.

На каждой policy/fold сравниваются пять методов:

| Метод | Model fit | Epoch/threshold | Вход/score |
|---|---|---|---|
| RF (контроль) | все labeled inner-fit days | сохранённый inner threshold | endpoint tabular probability |
| IF (контроль) | inner-fit background days | сохранённый inner threshold | endpoint tabular anomaly score |
| LSTM window (контроль) | all-background inner-fit windows | сохранённые epoch/threshold | пять дней, window mean MSE |
| LSTM endpoint | те же all-background windows, прежний full-window MSE loss | inner calibration с endpoint score | те же пять дней на входе, error последнего дня |
| HGB | все labeled inner-fit days, как RF | fixed 200 iterations; inner-calibration threshold | те же endpoint tabular inputs, positive-class probability |

Таким образом, **24 новых fit**: 4 policies × 3 folds × 2 новых метода;
**36 frozen controls** переиспользуются с сохранёнными assessments, всего
**60 assessment records**. Старые controls не переобучаются и не калибруются.
HGB/RF используют положительные labels при model fitting, IF/LSTM — нет;
превосходство classifier над reconstruction model не является сравнением
алгоритмов при равной supervision. Threshold calibration у всех использует
labels inner calibration, так что IF/LSTM здесь не полностью unsupervised.

LSTM сохраняет hidden32, Adam lr0,001, batch128, максимум 10 эпох, patience5
и mean-MSE optimization loss по всем пяти дням/признакам. Меняется только
score policy, включая основанный на ней выбор epoch и threshold. Это не
перенос старого threshold на новый score и не простая переоценка единственного
старого checkpoint. Early stopping и earliest exact epoch tie используют
только inner calibration. Dropout/weight decay/hidden size не меняются.

Endpoint reconstruction использует всё окно, включая наблюдаемый endpoint:
это end-of-day detection, не прогноз до наступления дня. Тест с единственной
эпохой проверяет одинаковые optimization loss и веса window/endpoint-вариантов:
новый scoring сам по себе не меняет градиенты. Absolute MSE разных score/feature
policies не является мерой сравнительного detection quality.

HGB имеет единственную объявленную конфигурацию: `loss=log_loss`, 200 iterations,
`learning_rate=0.05`, depth4, максимум 15 leaves/tree, minimum leaf20,
`l2_regularization=1`, max bins255, class weights balanced по inner-fit labels,
seed42. Histogram binning fit только на inner fit. `early_stopping=False`:
стандартный автоматический random validation split не должен разрывать
прежние connected groups. Число итераций фиксировано до scoring, без перебора.
Эти параметры не объявляются оптимальными для CERT.

## Оценка и защита границ

Только original primary train: 199 165 active account-days, 600 accounts,
592 группы; 196 765 endpoints, 646 observable-positive. Прежние roles/groups,
target, warmup, windows и все X сохранены; положительная история не очищается.
Original primary validation/test не оцениваются. Co-located daily CSV/context
проходит integrity verification и читается перед фильтрацией primary train;
это не утверждение об отсутствии любого чтения test-related source bytes.

Перед первым fit сохраняются plan/source/config/environment snapshots,
assignments всех folds, 36 controls, bindings всех 12 preparations и все 24 jobs.
Runtime source должен совпадать с snapshot до каждого worker и при завершении.
Workers отдельные, CPU/один поток. `test.csv` всегда закрыт; `assessment.csv`
закрыт до завершения candidate и фиксации checkpoint/threshold. Все numeric
inputs, memberships и scores сохраняются. Любая неудача остаётся в status/log;
она не разрешает удалить неудобный fold или выбрать новый seed.

Outer seed142, inner first split из пяти с seed242+fold, model seed42. Objective
и прежний percentile grid не меняются: F1, затем recall, lower FPR и lower
threshold; earliest epoch при точном tie. После фиксации candidate outer
assessment ничего не дообучает и не калибрует.

Отчёт публикует все folds/policies/methods, mean/sample SD/min/max, PR-AUC,
precision/recall/F1/FPR/confusion/support. Отдельно — парные endpoint−window и
HGB−RF разницы на одной policy/fold. Pooled decisions покрывают каждый endpoint
один раз на method/policy; pooled PR/ROC AUC намеренно undefined, потому что
scores разных fold models не обязаны иметь общую шкалу. Сценарные атрибуции
входят только в отчёт после всех новых обучений/inference.

Старые `experiment.inputs` и `experiment.score`, сам LSTM module и synthetic
reference scorer остаются прежними. Новые варианты используют отдельный
`configured_score`; отсутствие явной новой настройки делегирует прежнему
scorer. Старый frozen-final inference source guard должен продолжать проходить.

## Запуск

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 src/run_cert_model_variants.py --root "$CERT_PROCESSED_ROOT" \
  --days "$CERT_DAYS" --plan configs/cert_r4_2_model_variants_v5.json
```

Повтор команды создаёт другую серию, а не право искать красивый результат.
Здесь объявлена одна серия, один model seed и два отдельных новых метода.

## Как интерпретировать это в дипломе

Оставшиеся пункты могут повысить качество, но большого процента нельзя
гарантировать. В этой популяции всегда отвечающий «background» классификатор
получил бы около **99,67% accuracy при recall 0%**. Поэтому успех оценивается
по precision/recall, PR-AUC и цене FP, а не по одному высокому accuracy.
Сильная дипломная работа показывает воспроизводимые исходные результаты,
обоснованные изменения, отрицательные результаты, затраты и границы переноса.
Это дополняет задачу улучшения detection, а не отменяет слабость текущих моделей.

Пункт 5 сможет задать практическую рабочую точку и проверить устойчивость
выбранных development configurations. Он не может сам создать независимый
holdout из уже изученных дней. Нужны отдельно принятые независимые данные
для нового confirmatory утверждения. Те же folds остаются exploratory;
sample SD по трём зависимым folds не является confidence interval.

Выбор tree baseline поддерживает исследование
[Grinsztajn, Oyallon, Varoquaux, 2022](https://arxiv.org/abs/2207.08815): в их
табличном benchmark деревья часто превосходили нейросети. Оно не доказывает
превосходство HGB на CERT. Возможность хорошей реконструкции аномалий при слабом
разделении обсуждается в [Improved autoencoder for unsupervised anomaly detection](https://onlinelibrary.wiley.com/doi/10.1002/int.22582);
это возможный механизм, не установленный диагноз нашей LSTM. Параметры/early
stopping HGB сверены с [официальной документацией установленной версии sklearn 1.8](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).

## Выполненная серия

Серия `cert_model_variants_dd1281e93f8940c591264affbc572a07` завершена; manifest SHA-256:
`573e8e3cbebf2283ef232d8a49064733d70ac991dccfd22696905295dc846369`.
Все 24 новых fit завершены, вместе с 36 frozen controls опубликованы 60 assessment records. Технических неудач и дополнительных trials не было.

### Все новые outer assessments

| Fold | Features | Метод | Precision | Recall | F1 | PR-AUC | TP | FP | FN | Epoch selected/completed или iterations |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | v2 | LSTM endpoint | 0.020408 | 0.004505 | 0.007380 | 0.010972 | 1 | 48 | 221 | 5/10 |
| 1 | v2 | HGB | 0.438406 | 0.545045 | 0.485944 | 0.543372 | 121 | 155 | 101 | fixed 200 |
| 1 | v3 | LSTM endpoint | 0.028986 | 0.009009 | 0.013746 | 0.010574 | 2 | 67 | 220 | 9/10 |
| 1 | v3 | HGB | 0.855932 | 0.454955 | 0.594118 | 0.623649 | 101 | 17 | 121 | fixed 200 |
| 1 | rolling v4 | LSTM endpoint | 0.023256 | 0.009009 | 0.012987 | 0.010798 | 2 | 84 | 220 | 3/8 |
| 1 | rolling v4 | HGB | 0.791209 | 0.648649 | 0.712871 | 0.703217 | 144 | 38 | 78 | fixed 200 |
| 1 | frozen v4 | LSTM endpoint | 0.000000 | 0.000000 | 0.000000 | 0.010614 | 0 | 79 | 222 | 3/8 |
| 1 | frozen v4 | HGB | 0.627193 | 0.644144 | 0.635556 | 0.657008 | 143 | 85 | 79 | fixed 200 |
| 2 | v2 | LSTM endpoint | 0.035461 | 0.022624 | 0.027624 | 0.011653 | 5 | 136 | 216 | 4/9 |
| 2 | v2 | HGB | 0.561905 | 0.800905 | 0.660448 | 0.759125 | 177 | 138 | 44 | fixed 200 |
| 2 | v3 | LSTM endpoint | 0.047619 | 0.018100 | 0.026230 | 0.013859 | 4 | 80 | 217 | 9/10 |
| 2 | v3 | HGB | 0.880952 | 0.669683 | 0.760925 | 0.780766 | 148 | 20 | 73 | fixed 200 |
| 2 | rolling v4 | LSTM endpoint | 0.033898 | 0.027149 | 0.030151 | 0.014099 | 6 | 171 | 215 | 8/10 |
| 2 | rolling v4 | HGB | 0.923077 | 0.542986 | 0.683761 | 0.829188 | 120 | 10 | 101 | fixed 200 |
| 2 | frozen v4 | LSTM endpoint | 0.025510 | 0.022624 | 0.023981 | 0.013226 | 5 | 191 | 216 | 8/10 |
| 2 | frozen v4 | HGB | 0.732510 | 0.805430 | 0.767241 | 0.820809 | 178 | 65 | 43 | fixed 200 |
| 3 | v2 | LSTM endpoint | 0.046154 | 0.177340 | 0.073245 | 0.026824 | 36 | 744 | 167 | 5/10 |
| 3 | v2 | HGB | 0.195473 | 0.467980 | 0.275762 | 0.331330 | 95 | 391 | 108 | fixed 200 |
| 3 | v3 | LSTM endpoint | 0.052999 | 0.187192 | 0.082609 | 0.027372 | 38 | 679 | 165 | 9/10 |
| 3 | v3 | HGB | 0.312500 | 0.517241 | 0.389610 | 0.407821 | 105 | 231 | 98 | fixed 200 |
| 3 | rolling v4 | LSTM endpoint | 0.045392 | 0.162562 | 0.070968 | 0.025919 | 33 | 694 | 170 | 4/9 |
| 3 | rolling v4 | HGB | 0.369318 | 0.640394 | 0.468468 | 0.584523 | 130 | 222 | 73 | fixed 200 |
| 3 | frozen v4 | LSTM endpoint | 0.041026 | 0.157635 | 0.065107 | 0.026669 | 32 | 748 | 171 | 8/10 |
| 3 | frozen v4 | HGB | 0.235437 | 0.477833 | 0.315447 | 0.498323 | 97 | 315 | 106 | fixed 200 |

Все 36 контрольных fold records сохранены в `reused_controls.json`/`summary.json` и исходных отчётах пунктов 2–3. Они не переобучались и не калибровались.

### Mean outer F1

| Метод | v2 | v3 | rolling v4 | frozen v4 |
|---|---|---|---|---|
| RF | 0.226609 | 0.234597 | 0.274402 | 0.291851 |
| IF | 0.034802 | 0.016395 | 0.032686 | 0.042091 |
| LSTM window | 0.030564 | 0.031327 | 0.037770 | 0.031995 |
| LSTM endpoint | 0.036083 | 0.040861 | 0.038035 | 0.029696 |
| HGB | 0.474051 | 0.581551 | 0.621700 | 0.572748 |

### Mean outer PR-AUC

| Метод | v2 | v3 | rolling v4 | frozen v4 |
|---|---|---|---|---|
| RF | 0.231314 | 0.274137 | 0.289437 | 0.304638 |
| IF | 0.017415 | 0.017342 | 0.018200 | 0.019487 |
| LSTM window | 0.016320 | 0.015712 | 0.017616 | 0.017077 |
| LSTM endpoint | 0.016483 | 0.017268 | 0.016939 | 0.016836 |
| HGB | 0.544609 | 0.604079 | 0.705642 | 0.658714 |

Mean/sample SD/min/max по всем метрикам, включая ROC-AUC и FPR, сохранены в `summary.json`; SD зависимых folds не является confidence interval.

### Pooled OOF decisions

| Features | Метод | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| v2 | RF | 224 | 1105 | 422 | 195014 | 0.168548 | 0.346749 | 0.226835 |
| v2 | IF | 46 | 1345 | 600 | 194774 | 0.033070 | 0.071207 | 0.045164 |
| v2 | LSTM window | 79 | 3017 | 567 | 193102 | 0.025517 | 0.122291 | 0.042223 |
| v2 | LSTM endpoint | 42 | 928 | 604 | 195191 | 0.043299 | 0.065015 | 0.051980 |
| v2 | HGB | 393 | 684 | 253 | 195435 | 0.364903 | 0.608359 | 0.456181 |
| v3 | RF | 249 | 1255 | 397 | 194864 | 0.165559 | 0.385449 | 0.231628 |
| v3 | IF | 13 | 495 | 633 | 195624 | 0.025591 | 0.020124 | 0.022530 |
| v3 | LSTM window | 79 | 2917 | 567 | 193202 | 0.026368 | 0.122291 | 0.043383 |
| v3 | LSTM endpoint | 44 | 826 | 602 | 195293 | 0.050575 | 0.068111 | 0.058047 |
| v3 | HGB | 354 | 268 | 292 | 195851 | 0.569132 | 0.547988 | 0.558360 |
| rolling v4 | RF | 303 | 1393 | 343 | 194726 | 0.178656 | 0.469040 | 0.258753 |
| rolling v4 | IF | 28 | 1079 | 618 | 195040 | 0.025294 | 0.043344 | 0.031945 |
| rolling v4 | LSTM window | 81 | 3180 | 565 | 192939 | 0.024839 | 0.125387 | 0.041464 |
| rolling v4 | LSTM endpoint | 41 | 949 | 605 | 195170 | 0.041414 | 0.063467 | 0.050122 |
| rolling v4 | HGB | 394 | 270 | 252 | 195849 | 0.593373 | 0.609907 | 0.601527 |
| frozen v4 | RF | 256 | 905 | 390 | 195214 | 0.220500 | 0.396285 | 0.283343 |
| frozen v4 | IF | 58 | 1973 | 588 | 194146 | 0.028557 | 0.089783 | 0.043332 |
| frozen v4 | LSTM window | 78 | 2837 | 568 | 193282 | 0.026758 | 0.120743 | 0.043808 |
| frozen v4 | LSTM endpoint | 37 | 1018 | 609 | 195101 | 0.035071 | 0.057276 | 0.043504 |
| frozen v4 | HGB | 418 | 465 | 228 | 195654 | 0.473386 | 0.647059 | 0.546763 |

По каждой строке — те же 196 765 endpoints, включая 646 positives и 196 119 background. Pooled PR/ROC AUC не вычисляется.

### Сценарный recall

| Features | Метод | Сценарий 1 | Сценарий 2 | Сценарий 3 |
|---|---|---|---|---|
| v2 | RF | 5/58 (0.086207) | 219/556 (0.393885) | 0/32 (0.000000) |
| v2 | LSTM window | 39/58 (0.672414) | 40/556 (0.071942) | 0/32 (0.000000) |
| v2 | LSTM endpoint | 18/58 (0.310345) | 24/556 (0.043165) | 0/32 (0.000000) |
| v2 | HGB | 33/58 (0.568966) | 351/556 (0.631295) | 9/32 (0.281250) |
| v3 | RF | 10/58 (0.172414) | 239/556 (0.429856) | 0/32 (0.000000) |
| v3 | LSTM window | 40/58 (0.689655) | 39/556 (0.070144) | 0/32 (0.000000) |
| v3 | LSTM endpoint | 14/58 (0.241379) | 29/556 (0.052158) | 1/32 (0.031250) |
| v3 | HGB | 27/58 (0.465517) | 323/556 (0.580935) | 4/32 (0.125000) |
| rolling v4 | RF | 7/58 (0.120690) | 296/556 (0.532374) | 0/32 (0.000000) |
| rolling v4 | LSTM window | 43/58 (0.741379) | 35/556 (0.062950) | 3/32 (0.093750) |
| rolling v4 | LSTM endpoint | 14/58 (0.241379) | 27/556 (0.048561) | 0/32 (0.000000) |
| rolling v4 | HGB | 34/58 (0.586207) | 356/556 (0.640288) | 4/32 (0.125000) |
| frozen v4 | RF | 8/58 (0.137931) | 248/556 (0.446043) | 0/32 (0.000000) |
| frozen v4 | LSTM window | 39/58 (0.672414) | 39/556 (0.070144) | 0/32 (0.000000) |
| frozen v4 | LSTM endpoint | 14/58 (0.241379) | 23/556 (0.041367) | 0/32 (0.000000) |
| frozen v4 | HGB | 38/58 (0.655172) | 373/556 (0.670863) | 7/32 (0.218750) |

Сценарий 1: 58 дней/18 групп; сценарий 2: 556 дней/19 групп; сценарий 3: 32 дня/одна группа. IF также полностью сохранена в summary/reports.

### Парные эффекты двух гипотез

| Features | Сравнение | Δ mean F1 | Δ mean PR-AUC | Δ TP | Δ FP |
|---|---|---|---|---|---|
| v2 | LSTM endpoint − LSTM window | 0.005519 | 0.000163 | -37 | -2089 |
| v2 | HGB − RF | 0.247442 | 0.313295 | 169 | -421 |
| v3 | LSTM endpoint − LSTM window | 0.009534 | 0.001556 | -35 | -2091 |
| v3 | HGB − RF | 0.346954 | 0.329942 | 105 | -987 |
| rolling v4 | LSTM endpoint − LSTM window | 0.000265 | -0.000677 | -40 | -2231 |
| rolling v4 | HGB − RF | 0.347298 | 0.416206 | 91 | -1123 |
| frozen v4 | LSTM endpoint − LSTM window | -0.002299 | -0.000241 | -41 | -1819 |
| frozen v4 | HGB − RF | 0.280897 | 0.354076 | 162 | -440 |

Все individual fold deltas также опубликованы; по этой серии не добавляются новые trials и не выбирается новый outer threshold.

## Вывод и выбор дальнейшего направления

**HGB — обоснованный кандидат на основную supervised-модель Argus.** На каждой
из четырёх feature policies mean outer F1/PR-AUC выше RF; обе метрики выше и во
всех 12 парных policy/fold сравнениях. Это не результат единственного удобного
fold или смены labels. Все варианты и их неудачи сохранены.

Наибольшие средние показатели этой development-серии — у rolling HGB:
F1 **0,621700 ± 0,133498**, PR-AUC **0,705642 ± 0,122350**. Pooled decisions:
TP394, FP270, FN252; precision **59,34%**, recall **60,99%**, F1 **0,601527**.
При тех же rolling inputs у RF TP303/FP1393/FN343, precision17,87%, recall46,90%,
pooled F1 0,258753. Новый model family даёт **91 дополнительный TP и на 1 123
меньше FP**, сохраняя cohort. На 196 119 background endpoints приходится
примерно 1,377 FP на тысячу. Это доля на данном benchmark, не обещанная
частота тревог при эксплуатации.

Для связи с пунктом 3: ранее наибольший mean RF F1 0,291851 был у frozen v4;
её pooled TP256/FP905/FN390. Rolling HGB даёт TP394/FP270/FN252. Здесь вместе
меняются model family и выбранная feature policy, поэтому изолированный эффект
HGB следует брать из парного rolling HGB−RF сравнения выше. Оптимальная на
development policy зависит от модели; переносить frozen как универсального
победителя пункта 3 было бы неверно.

**Не все показатели оптимальны одновременно.** Frozen HGB имеет больший pooled
recall 64,71% (418 TP), но FP465, precision47,34% и pooled F1 0,546763 — хуже
rolling по precision/F1/FP. Его третий fold F1 0,315447 при 0,767241 во втором:
широкий разброс не скрывается. Даже rolling HGB в третьем fold имеет F1
0,468468 и precision36,93%, а во втором precision92,31%; отдельно выбранные
92% нельзя представлять как общую точность модели. В rolling HGB остаются
**252 пропущенных positive days**, включая 24/58 сценария 1, 200/556 сценария 2
и 28/32 сценария 3. По последнему одна независимая группа, сильных выводов о
переносе на новые инциденты нет.

**Endpoint score не спасает эту LSTM-AE.** На v3 mean F1 0,031327 → 0,040861,
но pooled TP79 → 44, recall12,23% → 6,81%, FP2917 → 826. На rolling mean F1
практически прежний (0,037770 → 0,038035), PR-AUC снижается; TP81 → 41.
На frozen mean F1 ухудшается, первый fold сохраняет нулевой recall. Во всех
policies существенная часть прежних TP теряется. Даже лучший endpoint precision
лишь около 5%: изменение target alignment само по себе не создаёт хорошую
detector model. Старый score и checkpoints не заменяются автоматически.

Это даёт основание **рассматривать HGB как основную supervised-модель, а
LSTM-AE как контрольный baseline**, сохранив RF, IF и оба LSTM scores для сравнения.
Это рекомендация по development evidence, а не автоматическое удаление LSTM
или объявление HGB production-ready. Плохое качество данного reconstruction
baseline не означает, что все нейросети или все LSTM непригодны; другой
supervised sequence model потребовал бы отдельной гипотезы и контроля
переобучения на небольшом числе positive accounts/groups.

**Дальнейшие пункты могут помочь, но не гарантируют нового скачка.** Здесь
измеримый прогресс уже получен. Пункт 5 должен сначала объявить практический
FP/recall objective, зафиксировать кандидатов и проверить устойчивость.
Другая рабочая точка может повысить precision ценой recall; нельзя ожидать
одновременного роста всех метрик. Для диплома теперь есть конкретный пример
обоснованной смены model family, положительные/отрицательные ablations и
сохранённые воспроизводимые артефакты. Новое независимое подтверждение всё ещё
требует отдельно принятых данных. Новой оценки использованного test не было.
На завершение пункта 4 пункт 5 не запускался. Теперь
[проверка рабочей точки и пяти seeds](cert-r4.2-operating-point-v6.md) выполнена
отдельной серией: основной FP/recall budget не подтвердился, независимых новых
CERT данных нет. При сверке исправлена текстовая описка выше: scenario1 FN24,
а не28; TP34/support58 и исходные JSON/веса/thresholds не менялись.

## Проверки реализации

Все **72 теста** прошли. Пять новых проверяют формулу endpoint score и
исключение prefix errors, точное legacy-delegation без новой настройки,
неизменность loss/весов после одной эпохи и inner-only selection, HGB fit по
всем inner-fit labels без random validation, полный mini-run всех четырёх
policies с 40 records и отказом при дополнительном факторе.
Compileall, shell syntax и whitespace checks пройдены. Полный `run_all.sh` и
`build_reference_snapshot.py` в отдельном временном проекте точно воспроизвели
прежние RF/IF/LSTM synthetic reference metrics; release reference не обновлялся.

Независимая повторная проверка всех 60 records загрузила каждый model checkpoint
и воспроизвела точные saved scores, decisions, inputs и window memberships.
Для 12 endpoint LSTM scores дополнительно рассчитаны непосредственно из
`model(batch)`/последнего дня; для HGB проверены все inner-fit IDs/labels/X,
fixed iterations и отсутствие early stopping. Config hashes совпали с plan
и settings snapshots, внутренние epoch/threshold grids перепроверены.
Все 12 preparation manifests и candidate/assessment manifests валидны.

SHA-256 всех **66 старых checkpoints** (30 первоначальной серии, 18 пункта 2,
18 новых пункта 3) и всех prerequisite manifests сохранились. Старый
frozen-final inference source guard проходит; `experiment.inputs/score` AST,
LSTM module, feature builders и прежние configs не менялись. Новые named
variants доступны в isolated API; default synthetic workflow остаётся прежним.

Работа продолжена в ветке `codex/cert-causal-history-v3`. Перед пунктом 4
сохранён полный snapshot исходных изменений; incremental patch пункта 4
отделён от прежней незакоммиченной работы. Смешанного commit/push не было.
