# Сценарій запису скринкаста для захисту

Тривалість запису ~2-3 хв. Мета: показати повний цикл preventive blocking (push → predictor → BLOCK → downstream skipped → override → re-run → merge unlocks).

---

## Перед записом — preflight checklist

> **NB:** після старту запису не варто переключатися між вікнами зайвий раз — лишити лише браузер + 1 термінал для пушу.

### 1. Інфраструктура (запустити `./scripts/demo_start.sh`)
- [ ] PostgreSQL + Redis (`docker compose up -d postgres redis`)
- [ ] Backend (`uv run uvicorn app.main:app --host 127.0.0.1 --port 8000`) — порт 8000
- [ ] Frontend (`npm run dev` в `frontend/`) — порт 3000
- [ ] Cloudflared named tunnel: `cloudflared tunnel run cicd-predictor` — стабільний URL **https://cicd-predictor.spedymax.org**
- [ ] Webhook URL у GitHub постійний — менять не треба (один раз вже налаштовано через `gh api -X PATCH repos/Spedymax/cicd-predictor-demo/hooks/623494949 -f config[url]=…`)

### 2. Перевірити, що все живе
```bash
curl -s http://127.0.0.1:8000/health                    # → {"status":"ok"}
curl -s http://127.0.0.1:3000 | head -1                  # → <!doctype html>
curl -sf "$TUNNEL_URL/health"                            # → {"status":"ok"} (через інет)
gh api "repos/Spedymax/cicd-predictor-demo/hooks" --jq '.[0].config.url'
# повинно повернути актуальний trycloudflare.com URL
```

### 3. Бекап-репозиторій у чистому стані
```bash
cd demo-repo
git checkout main && git pull origin main
git branch --no-merged main | xargs -I{} git branch -D {} 2>/dev/null  # local cleanup
```

### 4. Підготувати вікна
- **Browser tab 1:** https://github.com/Spedymax/cicd-predictor-demo/pulls (порожній або з закритими PR)
- **Browser tab 2:** http://localhost:3000 (Predictions list)
- **Terminal:** в директорії `demo-repo/`, готовий до пушу
- **OBS / QuickTime:** курсор поверх Terminal, готовий до Start Recording

---

## Сценарій (читати по пунктах вголос)

### Сцена 0 — вступ (10 сек)
> «Демонструю систему превентивного блокування CI/CD-пайплайнів на основі ML-прогнозу. Студент: Соколов, ІП-з21.»

Показати **dashboard list** (http://localhost:3000) — кілька real-прогнозів. Скрол вниз.

---

### Сцена 1 — push ризикованого коміту (30 сек)
У терміналі:
```bash
git checkout -b demo-defence-$(date +%s)
```
Сказати: «Створюю гілку, додаю ризиковану зміну в Dockerfile — переписую базовий образ на Ubuntu з CUDA-стеком, ~30 нових рядків.»

Швидкий редагувальник Dockerfile (можна копіпастити готовий зразок):
```bash
cat > Dockerfile <<'EOF'
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip build-essential \
    nvidia-cuda-toolkit nvidia-cuda-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install tensorflow-gpu==2.15.0 torch==2.1.0
COPY src/ src/
CMD ["python3", "-m", "src.app"]
EOF

git add Dockerfile && git commit -m "feat: GPU training stack" && git push -u origin HEAD
```

---

### Сцена 2 — прогноз з'являється в dashboard (15 сек)
Переключитися на **http://localhost:3000**, дочекатися появи нового рядка в списку (1-2 секунди).
Сказати: «Webhook доставлено, backend витягнув diff через GitHub API, побудував feature-vector, прогнав через v26 two-stage модель.»

Клікнути на новий prediction. Показати:
- **decision: BLOCK**
- **predicted_class: docker_build_failed**
- **risk_score ≈ 0.5+**
- **Class probabilities chart** (success ще може бути high — `пояснити пізніше`)
- **SHAP explanation (top 10)** — підсвітити червоні стовпці (has_dockerfile_change, avg_lines_per_file).
  Сказати: «SHAP TreeExplainer показує signed внесок кожної ознаки в P(failure). Червоне підвищує ризик, зелене знижує. base_value 0.5 → predicted 0.55.»
- **Recommendations** — switch to slim image, multi-stage build.

---

### Сцена 3 — PR відкривається, merge заблоковано (25 сек)
Переключитися на **GitHub** → відкрити автоматично створений (`gh pr create`) або вже існуючий PR.
Сказати: «Бранч-protection вимагає check `cicd-failure-predictor`. Predictor запостив status=failure → merge button disabled.»

Показати:
- Червоний X у Predictor gate (push + pull_request)
- Required badge біля `cicd-failure-predictor`
- 6× Skipped checks у docker-build / lint / test
- Кнопка Merge заблокована з підказкою

Сказати: «6 jobs з 8 _навіть не стартували_ — runner-хвилини збережено. Це **preventive gating**, не reactive.»

---

### Сцена 4 — клікнути на Predictor gate, показати log (15 сек)
GitHub → Actions → останній run → Predictor gate job → Wait for predictor verdict step.
Показати лог:
```
Polling predictor for ea34e4…
::notice title=Predictor verdict::decision=block class=docker_build_failed risk=0.505
::error title=Predictor BLOCK::High CI failure risk (docker_build_failed, risk=0.505).
##[error]Process completed with exit code 1.
```
Сказати: «Workflow gate polls REST endpoint, отримує verdict, exits 1 — downstream jobs з `needs: predictor-gate` GitHub-ом помічаються як Skipped.»

---

### Сцена 5 — override через dashboard (20 сек)
Повернутися в **dashboard** → prediction detail. Клікнути **Override...** → вибрати **Auto approve** → reason: «Ризик прийнятний для prototype-фази, схвалено».
Сказати: «Оператор може override-нути рішення. Backend:
1. Оновлює decision у БД (audit trail).
2. Постить новий status=success на GitHub.
3. Викликає GitHub Actions API для re-run failed jobs — все автоматично.»

Submit.

---

### Сцена 6 — workflow auto-reruns (30 сек)
Швидко переключитися на **GitHub Actions** → побачити нову иконку (in_progress).

Сказати: «Жодного ручного кліку на GitHub-боці. Backend сам тригернув re-run.»

Дочекатися завершення (~30 сек). Очікувано:
- **Predictor gate: success** ✅
- **lint: success** ✅
- **docker-build: failure** ❌ (бо Dockerfile реально не збирається)
- **test: failure** ❌ (немає тестів у demo-репо)

Сказати: «Predictor дозволив pipeline пройти, downstream запустилися — і **docker-build реально впав**. ML-прогноз підтверджено реальністю: модель була права ще до того, як runner стартував. Це reactive validation prevention-у.»

---

### Сцена 7 — підсумок (10 сек)
Сказати: «Три рівні захисту в одній архітектурі:
1. Workflow-level gate — preventive, не витрачає runner-хвилин.
2. Branch protection check — final merge gate.
3. Override-loop з audit trail — human-in-the-loop без втрати швидкості.

Запис закінчено.»

---

## Артефакти, які варто залишити в OBS-сцені (для скріншотів)

- PR #3 mergeStatus з 3 failing + 6 skipped
- SHAP chart на prediction detail
- Workflow run з skipped jobs
- Workflow run після override з downstream-failures
- Override modal у dashboard

## Backup-план

Якщо під час захисту cloudflared/інтернет впаде:
- Підготувати MP4-запис цього сценарію.
- Тримати локальний скріншот PR #3 у power-point слайді — як fallback.
