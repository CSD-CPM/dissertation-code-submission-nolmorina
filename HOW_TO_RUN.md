# How to Run the Fraud Detection App

This app can be used in two ways: **locally on your computer** or via a hosted version online. Running it locally is strongly recommended as it is significantly faster and gives you full control over training and data. The online version is available for a quick preview but may be slow due to free-tier resource limits.

---

## Option 1 — Run Locally (Recommended)

### What Is Inside the ZIP

Extract the ZIP file. You will get one folder called **Gradio**:

```
Gradio/
├── app.py                      ← the application
├── requirements.txt            ← packages to install
├── artifacts/                  ← pre-trained model (ready to use immediately)
├── src/                        ← source code
```

### Requirements

- **Python 3.9 or newer** — check with `python3 --version`
- **pip** — usually installed with Python
- An internet connection the first time (to download packages)

---

### Step 1 — Open a Terminal

**Mac:** `Command + Space` → type `Terminal` → Enter  
**Windows:** `Windows key` → type `cmd` or `PowerShell` → Enter  
**Linux:** `Ctrl + Alt + T`

### Step 2 — Navigate to the Gradio Folder

```bash
cd ~/Desktop/Gradio
```

Confirm you are in the right place by running `ls` — you should see `app.py` listed.

### Step 3 — Install Packages

Run this once:

```bash
pip install -r requirements.txt
```

This takes 1–3 minutes. Wait until the terminal prompt returns before continuing.

> If `pip` is not found, try `pip3`. On Mac/Linux, add `--user` if you get a permissions error.

### Step 4 — Start the App

```bash
python app.py
```

> If `python` is not found, try `python3`.

You will see:

```
* Running on local URL:  http://127.0.0.1:7860
```

### Step 5 — Open in Your Browser

Go to `http://127.0.0.1:7860` in Chrome, Firefox, Safari, or Edge. The app loads and is ready to use.

**The terminal must stay open while you use the app.** To stop it, press `Ctrl + C`.

### Restarting Later

Packages only need to be installed once. Next time, just repeat Steps 1–2 and run `python app.py`.

---

## Using the App

### Tab 1 — Fraud Scoring

Upload any transaction CSV. The app scores every row and returns a fraud probability (0–1), a binary flag, and a risk label — Low, Review, or High — sorted highest risk first. A download link for the full scored CSV is provided.

> `is_fraud` is not required for scoring. If present, it is carried through to the output unchanged.

### Tab 2 — Data Analysis

Upload a CSV to run schema validation, quality checks, and a feature engineering preview. Use this before training to check your data for missing columns, duplicates, or leakage.

### Tab 3 — Visualizations

Upload a CSV to generate 25 interactive charts covering fraud patterns across time, amount, geography, merchant category, and customer profile. Charts that need `is_fraud` show a notice when the label column is absent.

### Tab 4 — Model Training

Upload a labelled training CSV (must include `is_fraud`) and optionally a holdout test CSV. Choose your model families, set an alert budget, and click **Train Models**. The winning model is saved automatically and becomes active in the Scoring tab.

| Setting | What it does |
|---|---|
| Model families | Which algorithms to train |
| Alert budget | Top X% of transactions to flag — e.g. `0.01` = top 1% |
| Max training rows | Cap data size — set `0` to use all rows |
| Fast mode | Fewer trees, skips linear models — good for a quick test run |

### Tab 5 — Model Management

Browse all previously trained model versions and activate any of them with one click. Use this to roll back if a new training run underperforms.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `python: command not found` | Use `python3` instead of `python` |
| `pip: command not found` | Use `pip3` instead of `pip` |
| Port already in use | Another app is using port 7860 — close it or restart your computer |
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` |
| Page not loading | Check the terminal still shows `Running on local URL` — it must stay open |
| Upload fails with missing columns error | Check the CSV has all 13 required columns listed below |

---

## Required CSV Columns

All uploaded CSVs must contain these 13 columns (comma or pipe delimited):

```
trans_date, trans_time, amt, category, gender, state,
dob, lat, long, merch_lat, merch_long, city_pop, profile
```

The `is_fraud` column is **optional for scoring** and **required for training**.

---

## Option 2 — Online Preview

A hosted version is available at:

**https://huggingface.co/spaces/nolinjoo/fraud-detection**

No installation or account needed. Note that this version runs on free-tier hardware and can be noticeably slow, especially during model training. It is best suited for a quick look at the interface rather than full use.
