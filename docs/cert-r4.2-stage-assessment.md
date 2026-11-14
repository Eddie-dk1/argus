# Argus: оценка CERT-работы и оставшиеся этапы

Состояние на 14 ноября 2026 года. Итог пяти quality increments и рабочей
точки — в [отчёте пункта 5](cert-r4.2-operating-point-v6.md); общая очередь —
в [roadmap](roadmap.md), требования S0–S8 — в
[плане Spectra](spectra-lab-implementation-plan.md#10-порядок-реализации-и-приёмка).

## Что уже имеет исследовательскую ценность

- Приняты исходные CERT r4.2 и answers; target — точное observable membership,
  а не все дни аккаунта инсайдера. Данные, labels, source mapping и группы
  связаны hashes и проверками. CERT сам является публичной синтетической
  коллекцией; эти результаты не являются production evidence.
- Построены active account-days, exact five-active-day windows и общая
  оценочная популяция. Train-fitted transforms и past-only personal history
  отделены от разметки и от разрешённой калибровки. Incident-connected groups
  не разрываются ради удобного результата.
- Есть RF/IF/LSTM-AE baselines, отрицательные результаты, frozen final первой
  серии и controlled development ablations. Исправление истории, rolling/frozen
  file/USB profile, endpoint reconstruction score и смена model family
  проверялись отдельно, а не одним необъяснимым изменением.
- HGB дал существенный прирост на тех же development endpoints. Разница
  supervised fit у HGB/RF и background-only fit у IF/LSTM явно записана.
  LSTM baseline сохранён, из его неудачи не делается вывод о всех нейросетях.
- Все seeds/folds/ошибки, exact numerical inputs/scores/decisions, checkpoints,
  thresholds, configs, source и environment сохраняются. Есть исполняемые
  проверки reload и сохранности старых моделей.

Это сильная основа экспериментальной части диплома: есть обоснование решений,
контроли, измеримый эффект и воспроизводимые неудачи. Это не основание обещать
90–99% качества на любых данных. При 646 positives из 196 765 endpoints
классификатор «всегда background» имел бы accuracy99,67% и recall0%; поэтому
важны precision/recall/F1, FP и ограничения по сценариям.

## Что ограничивает текущий вывод

Новая серия остаётся development evidence. Те же folds многократно изучены,
а старый primary test уже использован. Новые model seeds проверяют случайность
обучения, но не создают независимую выборку. Сценарий 3 представлен только одной
connected group в новой CV; хорошее обобщение на него не установлено. Все
оценки относятся к offline переносу между аккаунтами, не к будущим дням и не
к реальному потоку SOC.

Нельзя напрямую сравнивать старый test F1 и новый development F1 как один
процент улучшения: это разные популяции и протоколы. Честное количественное
сравнение HGB с RF берётся из парных assessment на одинаковых inputs/endpoints.
Независимая CERT-проверка требует отдельно принятого совместимого источника с
entity/duplicate/episode audit до оценивания. Новый каталог или новый split
из уже изученных данных этого не заменяют.

## Что осталось по этапам

| Этап | Состояние и оставшаяся работа |
|---|---|
| S0: SSD и принятие CERT | Выполнено для текущего CERT snapshot. Любой новый источник проходит собственный acceptance gate. |
| S1: Контракты и изоляция | Основной isolated experiment workflow выполнен. Его контракты нужно использовать в будущем runtime/UI; legacy root-команды всё ещё не study runner. |
| S2: Публичный baseline | CERT preparation, первая frozen final и quality experiments выполнены. Независимое подтверждение улучшенного detector не получено; релиз v0.3.0 не объявлен автоматически. |
| S3: Frozen inference и сценарий | Все три пункта выполнены: общий inference, [exact replay/диагностика](s3-frozen-diagnostics.md), [synthetic raw patches и единый CLI](s3-synthetic-scenarios.md). Проверены no-op, пересчёт истории против full-split oracle и сохранность источников. Это программная основа, показатели детекции не изменились. |
| S4: Разбор и улучшение моделей | [Согласованы семь пунктов](s4-model-quality-plan.md): source/input, timeline, train-only references, residuals и два bounded цикла HGB/RF/LSTM с frozen controls. Ещё не начат; качество приоритетнее UI. CERT raw what-if требует отдельного контракта. |
| S5: Spectra Lab | Один backend и UI: обзор, сравнение моделей, разбор и сценарии, jobs/history и capabilities. Готового объединённого продукта пока нет. |
| S6: Подбор моделей | Bounded random search, затем честное сравнение с TPE при одинаковых budget/cohort; quality/latency/memory measurements. Текущие fixed ablations не являются готовым AutoML/Spectra search. Не превращать development reuse в независимое подтверждение. |
| S7: Поставка и диплом | Offline bundle, установка и smoke на clean checkout, компактный release snapshot, Git/release history, итоговые таблицы/рисунки/ограничения, эксперимент статьи и пользовательская проверка по плану. Текущие checks — часть, не весь этап. |
| S8 / v0.4–v0.5: network datasets | Принять CICIDS2017, затем CSE-CIC-IDS2018; audit schemas/labels/duplicates/non-finite values, freeze capture-day split, обучить отдельные кандидаты, сравнить attack-family errors и 2017→2018 transfer при совместимых features. Оба источника пока не оценены. |

S8 в implementation plan — следующая очередь adapters; v0.4/v0.5 в roadmap —
data milestones. Это разные способы группировать работу, а не два разных
обещания о готовности. Для первой Spectra UI не нужно ждать всех трёх источников;
network experiments остаются отдельной частью согласованного дипломного scope.

HGB сохраняется сильным кандидатом для будущих flow tables, но CERT веса и
user-day features туда не переносятся автоматически. На каждом новом сценарии
снова нужны свой feature contract, train/calibration, bounded comparison и
frozen final. Победитель CERT не объявляется заранее победителем CICIDS.
Для LSTM на flow data нужна обоснованная последовательность с доступными
entity/time полями и проверяемым membership. Порядок строк CSV не заменяет
этот контракт; отсутствие нужных полей ограничивает возможности сценария.

## История согласования S3

Обновление 2026-11-14: это предварительное предложение заменено
[согласованным планом из трёх пунктов](s3-frozen-inference.md). Пункт 1 реализует
постоянные правила и общий inference API; [пункт 2](s3-frozen-diagnostics.md)
завершил полные replay/диагностику. [Пункт 3](s3-synthetic-scenarios.md) завершил
synthetic scenarios/CLI. S3 выполнен; следующий implementation stage — S4.

1. Единый загрузчик frozen artifacts и inference contract для сохранённых
   моделей: порядок features, score semantics, numeric threshold, provenance
   и отказ при несовместимом state/schema. Поддержать HGB как tabular candidate,
   сохранив исследовательские baselines.
2. No-op replay на fixed synthetic примере: exact inputs/members и совпадение
   scores/decisions; ничего не fit и не calibrate в inference.
3. Один разрешённый synthetic `bytes_sent` raw patch с пересчётом зависимостей
   через прежний builder и проверкой против полного recompute. Не приписывать
   изменённому примеру прежний ground truth автоматически.
4. Сохранение base/variant lineage и простой Python/CLI маршрут для будущего
   UI. CERT включить для просмотра frozen результатов; raw what-if не включать
   без собственного source-field/reaggregation контракта.

Этот список сохраняет первоначальное предложение; фактическая приёмка S3
выполнена по трём согласованным пунктам. Для дальнейшего улучшения метрик стоит уменьшить
неопределённость независимой оценкой/качеством данных, а не бесконечно подбирать
параметры на уже известных folds.

Пункт 5 подтвердил важность этого приоритета: primary FP/recall budget не
выполнен ни в одном unique outer fold HGB, хотя его scores не меняются от
model seed. Это не повод скрывать результат или ослаблять условия. Для
диплома завершён обоснованный development-цикл; новая независимая проверка,
воспроизводимая поставка и единый пользовательский маршрут остаются работой.
