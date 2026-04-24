# Shorts Auto Editor

Automatically finds and cuts scenes from a full movie that match your shorts clip.

---

## Installation

> ⚠️ Only needs to be done once. Requires internet connection / takes about 15–20 minutes.

### Step 1 — Download

Click the green **Code** button at the top right of this page → **Download ZIP**

Extract the downloaded ZIP file.

---

### Step 2 — Install

Inside the extracted folder, **double-click `setup.bat`**.

A black window will appear and install everything automatically. It closes when done.

> 💡 If a "Windows protected your PC" warning appears, click **More info → Run anyway**.

---

### Step 3 — Launch

After installation, a **Shorts Auto Editor** shortcut will appear on your **Desktop** (and inside the folder).

Double-click it to launch.

---

## How to Use

### 1. Select Files

- Drag your short clip into the **📱 Shorts** area on the left (or click to browse)
- Drag the original movie file into the **🎬 Movie** area on the right
  - You can add multiple movie files — each will produce its own output

### 2. Set Output Folder Name

Enter a name for the output folder (e.g. `my_movie`)

### 3. Options

| Option | When to use |
|--------|-------------|
| **Visual Only** | Check this if your shorts clip has background music (BGM) |
| **Chronological** | Check this to prevent duplicate scenes in the output (recommended) |

### 4. Run

Click **▶ Run**.

The progress bar will fill as processing continues. Time depends on movie length.

| Environment | Time (2-hour movie) |
|-------------|---------------------|
| NVIDIA GPU | 15 ~ 25 min |
| No GPU (CPU only) | 1 hour or more |

> 💡 On first run, the AI model (~85MB) will be downloaded automatically. Internet connection required.

### 5. View Results

When done, click **📁 Open Output Folder** to find your results.

| File | Contents |
|------|----------|
| `(name)_final.mp4` | Final output video |
| `(name)_report.html` | Matching report (open in browser) |

---

### Other Buttons

- **■ Stop** — Stop processing at any time
- **↺ Reset** — Clear all inputs and start over
