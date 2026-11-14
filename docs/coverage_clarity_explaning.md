# Часть 2. Как считать coverage

## Шаг 1. Привести скоры к [0,1]

| Режим | Источник скора | Преобразование |
|---|---|---|
| с реранкером | логит Qwen3 Reranker | `sigmoid(logit)` |
| без реранкера | fusion score | min-max по кандидатам |

**Пороги для двух режимов разные.** Нельзя использовать одни числа — шкалы не совпадают.

Стартовые значения (калибровать на своём наборе, не брать на веру):

| Порог | с реранкером | без |
|---|---|---|
| `T_floor` — «в базе нет ничего» | 0.15 | 0.20 |
| `T_chunk` — чанк годится как доказательство | 0.35 | 0.40 |
| `T_high` — уверенное попадание | 0.60 | 0.65 |
| `delta` — минимальный отрыв top1−top3 | 0.05 | 0.07 |

## Шаг 2. Сила покрытия одного аспекта

Главная идея: **искать слово атрибута, а не слово темы.** Вопрос «какая гарантия на X» похож на чанк про X, но покрытием считается только наличие слова «гарантия».

```python
ATTR_TERMS = {
 "цена":        ["цена","стоимость","тариф","руб","₽","прайс","оплата"],
 "срок":        ["срок","дней","недель","месяц","внедрение","подключ"],
 "применимость":["подходит","для","применяется","рассчитан","сегмент"],
 "механизм":    ["как","за счёт","позволяет","выявля","снижа","переда"],
 "наличие":     ["есть","поддерживает","возможность","функция"],
 "гарантия":    ["гарантия","срок службы","обслуживание"],
}

def aspect_strength(a, evidence):
    cand = [c for c in evidence if c.score >= T_chunk]
    topic = any(lemma_overlap(a.keywords, c.text) for c in cand)
    attr  = any(lemma_overlap(ATTR_TERMS[a.attr], c.text) for c in cand)

    s = 1.0 if (topic and attr) else 0.5 if topic else 0.0

    # числовой атрибут требует числа рядом со словом атрибута
    if a.attr in ("цена","срок","гарантия") and s > 0:
        if not number_near_attr(cand, ATTR_TERMS[a.attr], window=120):
            s = min(s, 0.5)
    return s
```

`number_near_attr` — regex `\d[\d\s.,]*\s*(руб|₽|дн|мес|кВт|А|шт)` в окне ±120 символов от найденного слова атрибута. Это отсекает «про цену написали словами, но цифры нет».

## Шаг 3. Агрегация

```python
coverage_raw = sum(w[a] * strength[a] for a in aspects) / sum(w.values())
# weights: главный аспект 2, второстепенные 1
# aspects пусты (P5 упал) → один аспект из standalone_query, weight 1
```

## Шаг 4. Жёсткие гейты (перекрывают число)

```python
if top1 < T_floor:                                  return NONE
if named_product and named_product not in catalog:  return NONE
if not linked_entities and top1 < T_mid:            return NONE

level = HIGH    if coverage_raw >= 0.80 and top1 >= T_high else \
        PARTIAL if coverage_raw >= 0.35 else NONE

if level == HIGH and (top1 - top3) < delta:
    level = PARTIAL          # плоский профиль = неуверенность
```

## Шаг 5. Опциональный LLM-авторейтер

Только когда `0.35 ≤ coverage_raw < 0.80`, интент не простой и остался бюджет. Один вызов, по аспектам yes/no:

```
Дан вопрос и фрагменты. Для каждого аспекта ответь, есть ли во
фрагментах данные для ответа на него. Только JSON:
{"аспект": "yes|partial|no"}
Не оценивай стиль. Не используй свои знания об энергетике.
```

---

# Часть 3. Как считать clarity

## Шаг 1. Блокирующие слоты по интенту

| Интент | Блокирующие (вес) | Уточняющие |
|---|---|---|
| `pricing` | — | scale, segment |
| `product_consult` | pain (2), object_type (2) | scale, region |
| `task_diagnosis` | object_type (2) | metering_state, scale |
| `comparison` | use_case (2) | scale |
| `lead_intent` | name, phone, company (по 1) | — |
| прочее | — | — |

Пустой список → `clarity = 1.0` автоматически. Это защищает от «анкеты» на простых вопросах.

## Шаг 2. Значение слота

```python
VALUE = {"stated": 1.0, "inferred": 0.6, "conflict": 0.0, "missing": 0.0}
clarity_raw = sum(w[s]*VALUE[state[s]] for s in blocking) / sum(w.values())
```

`conflict` — пользователь назвал два разных значения («у нас и ТЦ, и склад»). Это не «заполнено», это повод переспросить.

## Шаг 3. Жёсткие блокеры

```python
blocked = unresolved_reference \
       or any(state[s]=="conflict" for s in blocking) \
       or (topic_switch and new_object_unknown)

level = CLEAR   if clarity_raw >= 0.99 and not blocked else \
        PARTIAL if clarity_raw >= 0.50 else UNCLEAR
```

## Шаг 4. Область действия слотов

| Слот | Живёт |
|---|---|
| segment, region, контакты | весь диалог |
| object_type, pain, scale, metering_state | текущая тема |

При `topic_switch=true` тематические слоты **не переносятся** в новую тему, а складываются в стек тем. При возврате — восстанавливаются. Иначе бот применит «СНТ» к вопросу про завод.

## Шаг 5. Стоит ли задавать вопрос

```python
def worth_asking(intent, slots, history, coverage):
    if coverage == NONE:                 return False   # спрашивать нечего
    if clarify_streak(history) >= 2:     return False
    missing = [s for s in blocking(intent)
               if VALUE[state[s]] < 1.0 and not declined(s, history)]
    if not missing:                      return False
    s = max(missing, key=weight)

    # опционально: проверка «изменится ли ответ»
    vals = plausible_values(s)                 # 2–3 значения
    if len(vals) >= 2:
        sets = [top5_doc_ids(retrieve(query, s=v)) for v in vals]
        if max_jaccard_distance(sets) < 0.2:
            return False                       # ответ одинаков → не спрашиваем
    return s
```

`clarify_streak` и `declined` считаются **из переданной истории** — состояние между кейсами не сохраняется. `declined` — regex по репликам пользователя: «не знаю», «просто скажите», «не скажу».

---

# Часть 4. Два прогона на числах

**Кейс А.** «Какая гарантия на счётчик X?»
- top1 = 0.72 (чанк про X), но слова «гарантия» нет ни в одном чанке ≥ 0.35
- `strength = 0.5` (тема есть, атрибута нет) → `coverage_raw = 0.5` → **PARTIAL**
- clarity: `pricing`-подобный, блокирующих нет → **CLEAR**
- Клетка «partial + ясно» →covered часть + «про гарантию в базе нет» + специалист. **Не выдумывать срок.**

**Кейс Б.** «Хочу снизить счета за электричество», история пуста
- нашлись карточки по `tasks`, top1 = 0.78, отрыв 0.12, атрибут «механизм» есть → `coverage_raw = 1.0` → **HIGH**
- блокирующий `object_type` пуст → `clarity_raw = 0` → **UNCLEAR**
- worth_asking: разные object_type ведут к разным карточкам → расстояние 0.6 → спрашиваем
- Клетка «high + неясно» → ответ на общем уровне + **один** вопрос про объект.