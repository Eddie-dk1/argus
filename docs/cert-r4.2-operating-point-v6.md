# CERT: пункт 5 — рабочая точка и устойчивость

Протокол объявлен 14 ноября 2026 года до новой CERT-серии.
[Bounded plan](../configs/cert_r4_2_operating_point_v6.json) фиксирует исходную
[серию пункта 4](cert-r4.2-model-variants-v5.md), hashes трёх конфигураций,
популяцию, folds, seeds, бюджеты и правило отказа. Это завершение пяти
[quality increments](cert-r4.2-improvement-plan.md), а не новая независимая
проверка использованного CERT test.

## Что фиксируется перед запуском

Основной development-кандидат — HGB с rolling file/USB v4. Контроли:
RF rolling на тех же входах и RF frozen, лучший RF-вариант пункта 3.
Список выбран по уже известным development результатам, а не на независимых
данных. IF и оба LSTM scores сохраняют свои результаты и код; их повторное
обучение в этой серии не объявляется. Новых features, architecture choices,
гиперпараметров или preprocessing fit нет.

Все три конфигурации проверяются с seeds **7, 21, 42, 84, 168** на прежних
трёх outer folds. Девять seed42 checkpoints используются неизменными, 36
новых моделей fit только на соответствующем inner fit. Нельзя выбирать лучший
seed, исключать неудачный fold или менять модель по результатам этой серии.
Seed42 — заранее выбранный reference, не победитель. Повторные seeds не
создают новых людей/инцидентов или независимых данных; у HGB результат может
оказаться детерминированным при этих размерах и настройках.

## Инженерная рабочая точка

Основной профиль: **не более 1 FP на 1000 background active account-days**
на inner calibration; требуемый recall — **не ниже 50%**. Это выбранный
инженерный ориентир для исследования, а не утверждённый пользователем SLA.
Он относится к фоновой единице «аккаунт × активный день», не к пакетам,
потокам, событиям, инцидентам или числу SOC-уведомлений. Background означает
отсутствие official observable, а не независимо подтверждённую безопасность.

Заранее объявлен второй профиль — **2 FP на 1000** при том же recall.
Он показывает чувствительность к бюджету; основным при неудаче первого не
становится. Третий контроль — прежний inner-calibration F1 percentile threshold
без изменений. Все 135 config/seed/fold/profile решений сохраняются.

Для каждого нового профиля:

1. Восстановить scores неизменной модели на inner calibration и проверить
   исходные X, y, endpoint IDs, exact window members и F1 threshold.
2. Построить конечную таблицу всех unique calibration scores и finite
   `nextafter(max_score,+inf)` для reject-all. Решение — `score >= threshold`;
   одинаковые scores всегда принимаются/отклоняются вместе. Это новая явно
   именованная selection policy, прежние percentile grids не переписываются.
3. Допустимый FP count — `floor(N_background × max_fp_per_1000 / 1000)`.
   Выбрать максимальный recall при этом ограничении, затем меньше FP,
   затем больший threshold. Никакой calibration по outer labels нет.
4. Если лучший бюджетный threshold не достигает recall50%, сохранить его
   с `eligible_on_calibration=False`. Бюджет не расширять, seed не менять,
   модель не переобучать ради прохождения. Даже reject-all — допустимый
   диагностический результат с нулевым recall, не пригодный detector.
5. До outer inference сохранить отдельный sealed inference bundle с весами,
   state, config, calibration inputs/scores, полным threshold search и всеми
   численными thresholds. Только затем открыть outer assessment.

Ограничение на calibration **не гарантирует** такой же FPR/recall на новых
аккаунтах. Outer-нарушения отдельно показываются и не вызывают новую selection.
HGB class weights не превращают raw probabilities в откалиброванную вероятность
атаки; здесь меняется decision threshold, не probability calibration.

## Границы и проверки

Используется только original primary train: те же 600 аккаунтов, 592 группы,
196 765 endpoints и 646 observable positives. Inner/outer connected groups
не пересекаются, первые четыре активных дня каждого аккаунта исключены из
endpoint cohort у всех моделей одинаково. Global state/scaler уже fit на
inner fit, personal history остаётся past-only без удаления positive days.
Это offline account transfer, не временной прогноз.

Worker запрещает prepared test reads постоянно и outer inputs/predictions
до завершения freeze всех рабочих точек. Полный source/config/environment
snapshot и список всех jobs сохраняются до fit. Parent integrity проверяет
источники/manifest bytes, в том числе co-located daily source с прежними test
данными; это не используется для нового test scoring или threshold selection.

Все profiles одной модели используют одни outer scores. Per-fold PR/ROC AUC
сохраняются; pooled cross-fold rank metrics не определяются. Pooled decisions
считаются отдельно для каждого seed: один endpoint не превращается в пять
наблюдений. Model-seed variation и fold variation разделены; SD не является
доверительным интервалом. Scenario attribution подключается только после
завершения всех fit/calibration/inference.

## Независимое подтверждение

Сейчас нет вновь принятого независимого CERT holdout. Старый final test уже
использован, development folds неоднократно изучены. Поэтому в пункте 5 нет
нового test scoring или confirmatory утверждения. Зафиксировать это ограничение
явно — предусмотренная планом ветка, а не переименование старого test.

Для независимого подтверждения понадобятся отдельно принятый совместимый
источник, audit сущностей/дубликатов/связанных эпизодов и frozen protocol до
оценки. CICIDS2017/CSE-CIC-IDS2018 проверят другой network-flow сценарий с
новым feature contract и обучением; они не являются прямым независимым test
уже обученного CERT account-day detector.

## Запуск

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 src/run_cert_operating_point.py \
  --root "$ARTIFACT_ROOT" \
  --days "$CERT_DAYS" \
  --plan configs/cert_r4_2_operating_point_v6.json
```

Ниже добавлены результаты фактической серии после завершения всех объявленных
jobs. Исходные objective, shortlist, budgets, seeds и правила отказа не менялись.

## Выполненная серия

Все 45 объявленных config/seed/fold jobs завершены: 36 новых fit, девять reused seed42 checkpoints, 135 decision profiles. Новых preparations нет; используются шесть прежних rolling/frozen preparations. Артефакт на SSD: `cert_operating_point_1677f9343b73418ea6887db0b680d458`. Основной кандидат и профиль не переизбирались после outer assessment.

### Pooled OOF: среднее по пяти model seeds

Каждый seed покрывает те же 196 765 endpoints ровно один раз. Здесь усредняются пять отдельных pooled metrics; строки/labels разных seeds не объединяются. `±` — sample SD по model seeds при фиксированных folds, не доверительный интервал.

| Модель | Профиль | Precision | Recall | F1 ± SD | FP mean [min,max] | TP mean [min,max] |
|---|---|---:|---:|---:|---:|---:|
| HGB rolling | прежний F1 | 59.34% | 60.99% | 0.6015 ± 0.0000 | 270.0 [270,270] | 394.0 [394,394] |
| HGB rolling | 1 FP / 1000 (основной) | 52.23% | 59.91% | 0.5580 ± 0.0000 | 354.0 [354,354] | 387.0 [387,387] |
| HGB rolling | 2 FP / 1000 (sensitivity) | 38.92% | 66.10% | 0.4900 ± 0.0000 | 670.0 [670,670] | 427.0 [427,427] |
| RF rolling | прежний F1 | 23.00% | 36.53% | 0.2772 ± 0.0197 | 833.2 [570,1393] | 236.0 [203,303] |
| RF rolling | 1 FP / 1000 (основной) | 44.41% | 18.11% | 0.2546 ± 0.0380 | 148.4 [109,225] | 117.0 [92,161] |
| RF rolling | 2 FP / 1000 (sensitivity) | 34.05% | 23.56% | 0.2774 ± 0.0318 | 293.4 [245,334] | 152.2 [120,182] |
| RF frozen | прежний F1 | 21.09% | 38.70% | 0.2720 ± 0.0188 | 948.6 [788,1153] | 250.0 [222,271] |
| RF frozen | 1 FP / 1000 (основной) | 46.09% | 21.27% | 0.2902 ± 0.0136 | 163.4 [111,197] | 137.4 [129,152] |
| RF frozen | 2 FP / 1000 (sensitivity) | 36.08% | 25.54% | 0.2977 ± 0.0229 | 295.0 [209,325] | 165.0 [137,184] |

### Достижение объявленных ограничений

Оба условия означают FP budget и recall ≥50%. Calibration feasibility и outer agreement считаются отдельно. Число из 15 = три fixed folds × пять seeds; это не 15 независимых экспериментов.

| Модель | Профиль | Calibration: оба / 15 | Outer: бюджет / 15 | Outer: recall / 15 | Outer: оба / 15 |
|---|---|---:|---:|---:|---:|
| HGB rolling | 1 FP / 1000 (основной) | 10 | 5 | 10 | 0 |
| HGB rolling | 2 FP / 1000 (sensitivity) | 10 | 10 | 15 | 10 |
| RF rolling | 1 FP / 1000 (основной) | 0 | 10 | 0 | 0 |
| RF rolling | 2 FP / 1000 (sensitivity) | 0 | 10 | 0 | 0 |
| RF frozen | 1 FP / 1000 (основной) | 0 | 10 | 0 | 0 |
| RF frozen | 2 FP / 1000 (sensitivity) | 2 | 11 | 0 | 0 |

### HGB: все folds и profiles, заранее выбранный reference seed42

| Fold | Профиль | Threshold | Calibration recall | Cal FP / допустимо | Cal eligible | Outer P | Outer R | Outer F1 | Outer FP | Outer FP/1000 | Outer оба |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 1 | прежний F1 | 0.824686519 | 87.01% | 12 / — | — | 79.12% | 64.86% | 0.7129 | 38 | 0.581 | — |
| 1 | 1 FP / 1000 (основной) | 0.761190566 | 89.61% | 16 / 26 | True | 67.43% | 66.22% | 0.6682 | 71 | 1.086 | False |
| 1 | 2 FP / 1000 (sensitivity) | 0.761190566 | 89.61% | 16 / 52 | True | 67.43% | 66.22% | 0.6682 | 71 | 1.086 | True |
| 2 | прежний F1 | 0.977675641 | 18.68% | 36 / — | — | 92.31% | 54.30% | 0.6838 | 10 | 0.153 | — |
| 2 | 1 FP / 1000 (основной) | 0.981441353 | 16.48% | 25 / 26 | False | 93.10% | 48.87% | 0.6409 | 8 | 0.122 | False |
| 2 | 2 FP / 1000 (sensitivity) | 0.972352928 | 19.78% | 48 / 52 | False | 92.31% | 59.73% | 0.7253 | 11 | 0.168 | True |
| 3 | прежний F1 | 0.637395429 | 58.70% | 25 / — | — | 36.93% | 64.04% | 0.4685 | 222 | 3.395 | — |
| 3 | 1 FP / 1000 (основной) | 0.617097746 | 61.96% | 26 / 26 | True | 32.43% | 65.02% | 0.4328 | 275 | 4.206 | False |
| 3 | 2 FP / 1000 (sensitivity) | 0.492662638 | 65.22% | 44 / 52 | True | 20.11% | 72.91% | 0.3152 | 588 | 8.993 | False |

### Все seeds: pooled decisions

| Модель | Seed | Профиль | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| HGB rolling | 7 | прежний F1 | 394 | 270 | 252 | 59.34% | 60.99% | 0.6015 |
| HGB rolling | 7 | 1 FP / 1000 (основной) | 387 | 354 | 259 | 52.23% | 59.91% | 0.5580 |
| HGB rolling | 7 | 2 FP / 1000 (sensitivity) | 427 | 670 | 219 | 38.92% | 66.10% | 0.4900 |
| HGB rolling | 21 | прежний F1 | 394 | 270 | 252 | 59.34% | 60.99% | 0.6015 |
| HGB rolling | 21 | 1 FP / 1000 (основной) | 387 | 354 | 259 | 52.23% | 59.91% | 0.5580 |
| HGB rolling | 21 | 2 FP / 1000 (sensitivity) | 427 | 670 | 219 | 38.92% | 66.10% | 0.4900 |
| HGB rolling | 42 | прежний F1 | 394 | 270 | 252 | 59.34% | 60.99% | 0.6015 |
| HGB rolling | 42 | 1 FP / 1000 (основной) | 387 | 354 | 259 | 52.23% | 59.91% | 0.5580 |
| HGB rolling | 42 | 2 FP / 1000 (sensitivity) | 427 | 670 | 219 | 38.92% | 66.10% | 0.4900 |
| HGB rolling | 84 | прежний F1 | 394 | 270 | 252 | 59.34% | 60.99% | 0.6015 |
| HGB rolling | 84 | 1 FP / 1000 (основной) | 387 | 354 | 259 | 52.23% | 59.91% | 0.5580 |
| HGB rolling | 84 | 2 FP / 1000 (sensitivity) | 427 | 670 | 219 | 38.92% | 66.10% | 0.4900 |
| HGB rolling | 168 | прежний F1 | 394 | 270 | 252 | 59.34% | 60.99% | 0.6015 |
| HGB rolling | 168 | 1 FP / 1000 (основной) | 387 | 354 | 259 | 52.23% | 59.91% | 0.5580 |
| HGB rolling | 168 | 2 FP / 1000 (sensitivity) | 427 | 670 | 219 | 38.92% | 66.10% | 0.4900 |
| RF rolling | 7 | прежний F1 | 206 | 616 | 440 | 25.06% | 31.89% | 0.2807 |
| RF rolling | 7 | 1 FP / 1000 (основной) | 161 | 225 | 485 | 41.71% | 24.92% | 0.3120 |
| RF rolling | 7 | 2 FP / 1000 (sensitivity) | 176 | 329 | 470 | 34.85% | 27.24% | 0.3058 |
| RF rolling | 21 | прежний F1 | 220 | 570 | 426 | 27.85% | 34.06% | 0.3064 |
| RF rolling | 21 | 1 FP / 1000 (основной) | 122 | 126 | 524 | 49.19% | 18.89% | 0.2729 |
| RF rolling | 21 | 2 FP / 1000 (sensitivity) | 146 | 290 | 500 | 33.49% | 22.60% | 0.2699 |
| RF rolling | 42 | прежний F1 | 303 | 1393 | 343 | 17.87% | 46.90% | 0.2588 |
| RF rolling | 42 | 1 FP / 1000 (основной) | 104 | 142 | 542 | 42.28% | 16.10% | 0.2332 |
| RF rolling | 42 | 2 FP / 1000 (sensitivity) | 182 | 334 | 464 | 35.27% | 28.17% | 0.3133 |
| RF rolling | 84 | прежний F1 | 248 | 869 | 398 | 22.20% | 38.39% | 0.2813 |
| RF rolling | 84 | 1 FP / 1000 (основной) | 106 | 140 | 540 | 43.09% | 16.41% | 0.2377 |
| RF rolling | 84 | 2 FP / 1000 (sensitivity) | 137 | 269 | 509 | 33.74% | 21.21% | 0.2605 |
| RF rolling | 168 | прежний F1 | 203 | 718 | 443 | 22.04% | 31.42% | 0.2591 |
| RF rolling | 168 | 1 FP / 1000 (основной) | 92 | 109 | 554 | 45.77% | 14.24% | 0.2172 |
| RF rolling | 168 | 2 FP / 1000 (sensitivity) | 120 | 245 | 526 | 32.88% | 18.58% | 0.2374 |
| RF frozen | 7 | прежний F1 | 250 | 1153 | 396 | 17.82% | 38.70% | 0.2440 |
| RF frozen | 7 | 1 FP / 1000 (основной) | 138 | 171 | 508 | 44.66% | 21.36% | 0.2890 |
| RF frozen | 7 | 2 FP / 1000 (sensitivity) | 168 | 324 | 478 | 34.15% | 26.01% | 0.2953 |
| RF frozen | 21 | прежний F1 | 222 | 788 | 424 | 21.98% | 34.37% | 0.2681 |
| RF frozen | 21 | 1 FP / 1000 (основной) | 134 | 111 | 512 | 54.69% | 20.74% | 0.3008 |
| RF frozen | 21 | 2 FP / 1000 (sensitivity) | 182 | 297 | 464 | 38.00% | 28.17% | 0.3236 |
| RF frozen | 42 | прежний F1 | 256 | 905 | 390 | 22.05% | 39.63% | 0.2833 |
| RF frozen | 42 | 1 FP / 1000 (основной) | 129 | 176 | 517 | 42.30% | 19.97% | 0.2713 |
| RF frozen | 42 | 2 FP / 1000 (sensitivity) | 154 | 320 | 492 | 32.49% | 23.84% | 0.2750 |
| RF frozen | 84 | прежний F1 | 251 | 811 | 395 | 23.63% | 38.85% | 0.2939 |
| RF frozen | 84 | 1 FP / 1000 (основной) | 134 | 162 | 512 | 45.27% | 20.74% | 0.2845 |
| RF frozen | 84 | 2 FP / 1000 (sensitivity) | 137 | 209 | 509 | 39.60% | 21.21% | 0.2762 |
| RF frozen | 168 | прежний F1 | 271 | 1086 | 375 | 19.97% | 41.95% | 0.2706 |
| RF frozen | 168 | 1 FP / 1000 (основной) | 152 | 197 | 494 | 43.55% | 23.53% | 0.3055 |
| RF frozen | 168 | 2 FP / 1000 (sensitivity) | 184 | 325 | 462 | 36.15% | 28.48% | 0.3186 |

### Сценарии, reference seed42

| Модель | Профиль | Scenario 1 | Scenario 2 | Scenario 3 |
|---|---|---|---|---|
| HGB rolling | прежний F1 | 34/58 (58.62%); groups=18 | 356/556 (64.03%); groups=19 | 4/32 (12.50%); groups=1 |
| HGB rolling | 1 FP / 1000 (основной) | 35/58 (60.34%); groups=18 | 346/556 (62.23%); groups=19 | 6/32 (18.75%); groups=1 |
| HGB rolling | 2 FP / 1000 (sensitivity) | 35/58 (60.34%); groups=18 | 386/556 (69.42%); groups=19 | 6/32 (18.75%); groups=1 |
| RF rolling | прежний F1 | 7/58 (12.07%); groups=18 | 296/556 (53.24%); groups=19 | 0/32 (0.00%); groups=1 |
| RF rolling | 1 FP / 1000 (основной) | 0/58 (0.00%); groups=18 | 104/556 (18.71%); groups=19 | 0/32 (0.00%); groups=1 |
| RF rolling | 2 FP / 1000 (sensitivity) | 0/58 (0.00%); groups=18 | 182/556 (32.73%); groups=19 | 0/32 (0.00%); groups=1 |
| RF frozen | прежний F1 | 8/58 (13.79%); groups=18 | 248/556 (44.60%); groups=19 | 0/32 (0.00%); groups=1 |
| RF frozen | 1 FP / 1000 (основной) | 0/58 (0.00%); groups=18 | 129/556 (23.20%); groups=19 | 0/32 (0.00%); groups=1 |
| RF frozen | 2 FP / 1000 (sensitivity) | 0/58 (0.00%); groups=18 | 154/556 (27.70%); groups=19 | 0/32 (0.00%); groups=1 |

У сценария 3 только одна incident-connected group в этом development cohort; все seeds повторяют ту же группу. Support не становится в пять раз больше.

### Устойчивость и разброс

- HGB rolling: exact outer score equality между всеми seeds в 3 из 3 folds.
- RF rolling: exact outer score equality между всеми seeds в 0 из 3 folds.
- RF frozen: exact outer score equality между всеми seeds в 0 из 3 folds.

Все numeric thresholds заморожены до соответствующего outer score. `frozen_shortlist.json` сохраняет выбор до первого fit, `declared_jobs.json` — весь бюджет, `frozen_candidates.json` — все веса/thresholds и hashes. Полные per-fold показатели, в том числе PR/ROC AUC и violations, находятся в `summary.json`; exact arrays — в каждом assessment. Между folds PR/ROC AUC не агрегируются через общий score ranking.


## Оценка результата пункта 5

**Объявленная основная рабочая точка не подтвердилась.** Ни HGB, ни один из
двух RF-вариантов не выполнил одновременно outer FP budget1/1000 и recall50%
ни в одном из 15 config-specific fold/seed случаев. Для HGB это три unique
fold outcomes, повторённые с пятью seeds, а не 15 независимых неудач.
Calibration eligible у HGB только два folds; в fold2 best budget-respecting
recall16,48%. В outer fold1 FP rate1,086/1000, в fold3 —4,206/1000 превышают
бюджет; fold2 выполняет budget, но recall48,87% ниже требования. Это показывает
ограничение переноса calibration на другие аккаунты. Причина не установлена
одним этим сравнением; нельзя выдавать такое наблюдение за доказанный механизм.

HGB primary profile даёт pooled TP387/FP354/FN259, precision52,23%, recall59,91%,
F1 0,558039 и **1,805 FP/1000**. Прежний F1-контроль даёт TP394/FP270/FN252,
precision59,34%, recall60,99%, pooled F1 0,601527. Пункт 5 не улучшил эти
показатели HGB. Контроль сохраняется как прежний исследовательский reference;
он не переизбирается задним числом в качестве победившей основной рабочей точки.

Бюджет2/1000 повышает HGB pooled recall до66,10%, но FP670, precision38,92%,
F1 0,489960 и actual FPR3,416/1000. В fold3 FP rate8,993/1000; sensitivity
профиль тоже не даёт общего выполнения условий. Он остаётся secondary analysis.
У RF strict profiles заметно меньше FP, но mean pooled recall18,11%/21,27%
для rolling/frozen. Подмена основного бюджета более удобным результатом или
исключение fold3 не производились.

**HGB не зависит от этих model seeds в данном режиме.** Все exact outer score
arrays совпадают между пятью seeds во всех трёх folds. При фиксированных
данных/настройках здесь нет измеренной seed variation; пять повторов не
доказывают устойчивость к другим группам или новым источникам. У RF scores
различаются во всех folds: pooled F1 старого контроля rolling0,2772±0,0197,
frozen0,2720±0,0188. При seed42 fold variation HGB F1-контроля по-прежнему
0,621700±0,133498; большой group-related разброс не исчезает от смены seed.

Сценарий3 остаётся слабым: HGB F1-контроль4/32, оба новых профиля6/32 при одной
independent group. Рост на двух днях не устанавливает хороший перенос.
У основного нового профиля scenario2 TP356→346, FN200→210; этот регресс сохранён.

**Все пять quality increments выполнены в development-границах.** Основной
измеримый прирост дал пункт 4: на одинаковых rolling inputs/targets у RF и HGB
mean outer F1 0,274402→0,621700 и PR-AUC 0,289437→0,705642 при seed42.
Пункт 5 зафиксировал shortlist, численные thresholds и реальное ограничение
FP/recall, а не обеспечил обещанный operational performance. Сохраняем HGB
как основной supervised research candidate, RF/IF/LSTM как сравнения. Готового
единого deployment threshold или новой подтверждённой test-метрики нет.
Приоритет дальнейшего quality work — разнообразие независимых групп,
представительность calibration и независимая проверка по отдельному протоколу.

Сейчас нельзя получить новое confirmatory утверждение из старого test: вновь
принятых независимых CERT данных нет. Это явно сохранено в `summary.json`.
CICIDS2017/CSE-CIC-IDS2018 потребуют отдельных adapters, fit и frozen evaluation
для network-flow задачи. Общая оценка дипломной готовности, оставшиеся S3–S8
и предложение четырёх последовательных задач S3 приведены
[в отдельной карте](cert-r4.2-stage-assessment.md). Следующий этап не запускался.

## Provenance фактической серии

- Артефакт: `cert_operating_point_1677f9343b73418ea6887db0b680d458`.
- SHA-256 `complete.json`: `9f417f3c02b55865bb27cb8015b619fc9d97d1b15d87e32f84604061506312c9`.
- SHA-256 исходного plan config: `f7c41140534a903bb2a770bcf0415b08acd4ba62091ca1465de2c7357f21f273`.
- SHA-256 serialized `plan.json` в artifact: `a8c8099542c638fef944fc16876da67e0141fa8da8d4744ec2bbc438ba896d86`.
  JSON values совпадают; в исходном config есть завершающий newline, в saved
  JSON — нет. Это различие сериализации, не изменение протокола.
- Все 36 новых fit и девять reuses завершены с первого объявленного запуска
  без дополнительных CERT trials или скрытых исключений.

## Проверки и сохранность

Все **77 tests** прошли. Пять новых проверяют score ties/integer budgets,
reject-all и infeasible recall без ослабления условий, совпадение selector
с brute-force oracle, invalid inputs, freeze при отсутствующем outer file,
запрет fit/selection в assessment, отличие calibration budget от outer
результата и полный mini-run пяти seeds с сохранением всех controls/profiles.
После уточнения lineage metadata повторно прошёл freeze/assessment test.
Compileall, CLI help, shell syntax, links/anchors и whitespace checks пройдены.

Полный `run_all.sh` и `build_reference_snapshot.py` выполнены в отдельной
временной копии проекта. RF/IF/LSTM synthetic metrics JSON точно совпали
с прежним reference; публичные reference artifacts не обновлялись.

Отдельный verification script повторно загрузил все **45** моделей и
воспроизвёл exact calibration/outer scores, решения всех **135** профилей,
численные inputs, endpoint IDs и все window members. Обучающие X/IDs/y
совпали со всеми labeled inner-fit rows. Numeric operating thresholds
независимо пересчитаны через `sklearn.metrics.roc_curve` с сохранением всех
score ties и тем же integer FP budget; выбранные thresholds совпали.
Проверены прежние F1 percentile thresholds и точное совпадение девяти seed42
outer score arrays/решений с controls пункта 4. Все шесть preparations и
новые manifests валидны; blocked test/assessment access attempts отсутствуют.

Все **97 прежних checkpoints** и **331** ранее существовавший manifest/weight
SHA-256 совпали с snapshot до серии. Старый frozen-final inference source guard
проходит. Прежние feature builders, training/score код и configs не менялись.
Исправлена только описка FN в прозе отчёта пункта 4 (scenario1: 24 из58,
при неизменных TP34); исходные численные artifacts остались прежними.

Ветка остаётся `codex/cert-causal-history-v3`. Перед пунктом 5 сохранён snapshot
предыдущего состояния; отдельный incremental patch включает только этот
пункт. Общего commit/push с прежними незакоммиченными изменениями не было.
Большие inputs/models/predictions сохранены на SSD, а не в Git.
